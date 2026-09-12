// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

#![deny(deprecated)]

mod caro;
mod configuration;
mod free_style;
mod gomoku;
mod mcts;
mod openings;
mod ortcommon;
mod ortopt;
mod play;
mod renju;
mod rule;
mod standard;

use configuration::cfg;
use gomoku::{GameStage, Gomoku};
use mcts::MCTS;
use openings::{Opening, ScheduledGame, load_openings, pair_count, schedule, sign_test_p_value};
use ortopt::NeuralNetwork;
use play::SelfPlay;
use rule::Color;

use std::{
    cell::RefCell,
    collections::HashMap,
    env, fs,
    io::Write,
    path::Path,
    rc::Rc,
    sync::atomic::AtomicUsize,
    time::{Duration, Instant},
};

pub fn sims_for_weight(weight_id: u16) -> usize {
    let boosted = cfg::DEFAULT_SIMULATION_NUM
        + (weight_id as usize / cfg::SIMS_BOOST_EVERY as usize) * cfg::SIMS_BOOST_STEP;
    boosted.min(cfg::SIMS_CAP)
}

/// Simulation budget for acceptance evaluation, returned with the source for the log.
///
/// `EVAL_SIMS` pins the evaluation to a fixed budget: the generation schedule
/// (`sims_for_weight`) grows to `SIMS_CAP`, so evaluating a late generation costs up to
/// three times a generation-0 evaluation for the same number of games. Pinning the
/// budget keeps an evaluation round inside a short Colab session, and both sides always
/// get the same value.
fn eval_sims_for(weight_id: u16) -> (u16, &'static str) {
    match env::var("EVAL_SIMS")
        .ok()
        .as_deref()
        .and_then(parse_positive::<u16>)
    {
        Some(sims) => (sims, "EVAL_SIMS"),
        None => (sims_for_weight(weight_id) as u16, "generation schedule"),
    }
}

/// Parse a positive override value; zero, negative and unparsable values are ignored so
/// the caller falls back to its default. Shared by `EVAL_SIMS` and `EVAL_WORKERS`.
fn parse_positive<T>(value: &str) -> Option<T>
where
    T: std::str::FromStr + PartialOrd + From<u8>,
{
    let parsed = value.trim().parse::<T>().ok()?;
    (parsed > T::from(0u8)).then_some(parsed)
}

pub async fn generate_data_for_train(cur_weight_id: u16, start_batch_id: u16) {
    if let Ok(cur_path) = env::current_dir() {
        println!("Current folder: {:?}", cur_path);
        let model_path = cur_path
            .join("weights")
            .join(cur_weight_id.to_string() + ".onnx");

        println!("Current training model path: {:?}", model_path);

        let thread_num = cfg::NUM_2_SELF_PLAY_THREADS as usize;
        let total_games = cfg::NUM_2_SELF_PLAY as usize;
        let base = total_games / thread_num;
        let remain = total_games % thread_num;
        // lower the per-instance intra-op thread count as instances grow to avoid
        // a multiplied total thread count
        let intra_thread_num = ((cfg::DEFAULT_INTRA_THREAD_NUM as usize) / thread_num).max(2) as u8;

        // simulation count grows with weight generation
        let sims = sims_for_weight(cur_weight_id);

        let mut handles = Vec::with_capacity(thread_num);
        let mut offset = 0usize;
        for t in 0..thread_num {
            let model_path = model_path.clone();
            let game_num = (base + if t < remain { 1 } else { 0 }) as u16;
            let start_id = start_batch_id + offset as u16;
            offset += game_num as usize;
            handles.push(tokio::task::spawn_blocking(move || {
                if let Ok(m) = NeuralNetwork::new(
                    &model_path,
                    cfg::DEFAULT_BATCH_SIZE as usize,
                    intra_thread_num,
                ) {
                    // Rc is only used within this thread; NeuralNetwork only holds an
                    // UnboundedSender, so it can move across threads
                    let model_ref = Rc::new(RefCell::new(m));
                    let sp = SelfPlay::new(model_ref);
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build();
                    match rt {
                        Ok(rt) => rt.block_on(sp.self_play_for_train(game_num, start_id, sims)),
                        Err(error) => eprintln!("Create runtime error: {}", error),
                    }
                } else {
                    eprintln!("Load model error!!!");
                }
            }));
        }
        for handle in handles {
            if let Err(error) = handle.await {
                eprintln!("Self play thread error: {}", error);
            }
        }
    } else {
        eprintln!("Can not find current folder!!!");
    }
}

/// Build the starting position of one evaluation game.
///
/// The opening is injected before the MCTS instances are created, so both searches
/// start from the book position. With `opening == None` this is exactly the legacy
/// empty-board position.
pub fn build_eval_position(opening: Option<&Opening>) -> Option<Gomoku> {
    let mut game = Gomoku::new(cfg::BOARD_SIZE, cfg::N_IN_ROW)?;
    if let Some(opening) = opening
        && !game.load_position(opening.stones(), Color::Black)
    {
        eprintln!("Failed to load opening {:?}", opening.stones());
        return None;
    }
    Some(game)
}

/// Everything one evaluation game needs, grouped so the call site stays readable
/// now that the opening book is part of the setup.
struct EvalGame<'a> {
    nn_a: Option<Rc<RefCell<NeuralNetwork>>>,
    nn_b: Option<Rc<RefCell<NeuralNetwork>>>,
    a_first: bool,
    do_render: bool,
    num_mcts_sim_a: u16,
    num_mcts_sim_b: u16,
    opening: Option<&'a Opening>,
}

/// Play one game from the scheduled position, returning net A's result counts.
///
/// `None` means the game could not be played at all (the position failed to load or was
/// already decided). That must stay distinct from a played game, otherwise an unusable
/// opening would silently enter the score as a 0-0 result.
async fn play_eval_game(setup: EvalGame<'_>) -> Option<(u16, u16, u16)> {
    let EvalGame {
        nn_a,
        nn_b,
        a_first,
        do_render,
        num_mcts_sim_a,
        num_mcts_sim_b,
        opening,
    } = setup;

    let mut a_win = 0;
    let mut b_win = 0;
    let mut draw = 0;
    let mut step = 0u16;
    let game = build_eval_position(opening)?;
    let g_ref = Rc::new(RefCell::new(game));

    let mut game_state = {
        let mut game = g_ref.borrow_mut();
        *game.get_game_status()
    };
    if game_state.0 != GameStage::Running {
        // an opening that is already decided would pick the pair winner from the book
        // file instead of from play, so it must never be counted as a game
        eprintln!("Evaluation opening is already terminal, skipping the game");
        return None;
    }

    let mut ma = MCTS::new(
        nn_a,
        cfg::C_PUCT as f64,
        cfg::C_VIRTUAL_LOSS,
        AtomicUsize::new(num_mcts_sim_a as usize),
        cfg::DEFAULT_SIM_PER_BATCH_NUM,
        g_ref.borrow().get_action_size(),
    );
    let mut mb = MCTS::new(
        nn_b,
        cfg::C_PUCT as f64,
        cfg::C_VIRTUAL_LOSS,
        AtomicUsize::new(num_mcts_sim_b as usize),
        cfg::DEFAULT_SIM_PER_BATCH_NUM,
        g_ref.borrow().get_action_size(),
    );

    while game_state.0 == GameStage::Running {
        // a book opening always has an even ply count, so the side to move at step 0
        // is Black and the plain step parity below still identifies the colours
        let is_a_turn = if a_first {
            step % 2 == 0
        } else {
            step % 2 != 0
        };
        let best_action = if is_a_turn {
            ma.get_best_action(&g_ref.borrow()).await
        } else {
            mb.get_best_action(&g_ref.borrow()).await
        };
        let is_update_succeed_a = ma.update_root_with_action(&g_ref.borrow(), best_action);
        let is_update_succeed_b = mb.update_root_with_action(&g_ref.borrow(), best_action);
        if is_update_succeed_a && is_update_succeed_b {
            g_ref.borrow_mut().execute_move(best_action);
        } else {
            eprintln!("May be wrong with MCTS!!!");
        }

        if do_render {
            println!("step: {}", step);
            g_ref.borrow().render();
            println!();
        }
        game_state = {
            let mut game = g_ref.borrow_mut();
            *game.get_game_status()
        };

        step += 1;
    }
    println!(
        "eval: total step num = {} (opening ply {})",
        step,
        opening.map_or(0, Opening::plies)
    );

    if (game_state.1 == Color::Black && a_first) || (game_state.1 == Color::White && !a_first) {
        println!("winner = a");
        a_win += 1;
    } else if (game_state.1 == Color::Black && !a_first)
        || (game_state.1 == Color::White && a_first)
    {
        println!("winner = b");
        b_win += 1;
    } else {
        draw += 1
    }

    Some((a_win, b_win, draw))
}

/// Outcome of one evaluation: raw game counts plus the score of every complete
/// colour-swapped pair (0 = A lost both, 0.5 = even, 1 = A won both).
pub struct EvalOutcome {
    pub a_win: u16,
    pub b_win: u16,
    pub draw: u16,
    pub pair_scores: Vec<f64>,
    /// Wall-clock of the whole evaluation (session creation included). Budget with this:
    /// unlike the games it does not scale with the game count, so it is worth pinning
    /// down separately from the per-pair rate.
    pub elapsed: Duration,
    /// Wall-clock spent creating the ONNX sessions (the slowest worker). A large value
    /// here is a cold runtime or a heavy weight file, not an expensive search — this is
    /// what a logging-only view of the total would hide.
    pub load_elapsed: Duration,
    /// Whether every scheduled game produced a result. A failed evaluation reports
    /// zero games and must never be read as a score for net A.
    pub complete: bool,
}

impl Default for EvalOutcome {
    fn default() -> Self {
        Self {
            a_win: 0,
            b_win: 0,
            draw: 0,
            pair_scores: Vec::new(),
            elapsed: Duration::ZERO,
            load_elapsed: Duration::ZERO,
            complete: false,
        }
    }
}

impl EvalOutcome {
    /// Score of net A with draws counted as `DRAW_SCORE` (0 when no game was played).
    pub fn win_ratio(&self) -> f64 {
        let total = self.a_win + self.b_win + self.draw;
        if total == 0 {
            return 0.0;
        }
        (f64::from(self.a_win) + cfg::DRAW_SCORE * f64::from(self.draw)) / f64::from(total)
    }

    /// Pair decisions for the sign test: wins, losses and ties (pairs scoring exactly
    /// 0.5, which carry no information about which side is stronger).
    pub fn pair_decisions(&self) -> (u32, u32, u32) {
        let mut wins = 0;
        let mut losses = 0;
        let mut ties = 0;
        for score in &self.pair_scores {
            if *score > 0.5 {
                wins += 1;
            } else if *score < 0.5 {
                losses += 1;
            } else {
                ties += 1;
            }
        }
        (wins, losses, ties)
    }

    /// Wall-clock actually spent playing, i.e. the total minus session creation.
    pub fn games_elapsed(&self) -> Duration {
        self.elapsed.saturating_sub(self.load_elapsed)
    }

    /// Mean wall-clock seconds per colour-swapped pair over the whole evaluation: the
    /// unit to budget an evaluation in. Session creation is per-process, so it weighs
    /// less per pair as the game count grows.
    pub fn seconds_per_pair(&self) -> f64 {
        if self.pair_scores.is_empty() {
            return 0.0;
        }
        self.elapsed.as_secs_f64() / self.pair_scores.len() as f64
    }

    /// Mean wall-clock seconds per pair of actual play, which is the rate that scales
    /// with the game count when sizing a screen.
    pub fn games_seconds_per_pair(&self) -> f64 {
        if self.pair_scores.is_empty() {
            return 0.0;
        }
        self.games_elapsed().as_secs_f64() / self.pair_scores.len() as f64
    }
}

/// Describe a scheduled opening for the log without dumping the whole book.
fn describe_opening(opening: Option<&Opening>) -> String {
    match opening {
        None => "empty board".to_string(),
        Some(opening) => format!("book, {} ply", opening.plies()),
    }
}

/// Fold the per-game score of net A into colour-swapped pair scores.
///
/// A pair is one opening played with each colour, so the first-move advantage cancels
/// inside it and the pair score is the pair's fair estimate of strength. A trailing
/// game without a partner is dropped: it has nothing to cancel against.
pub fn pair_scores_from_games(game_scores: &[f64]) -> Vec<f64> {
    game_scores
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .collect()
}

/// One finished game as reported by an evaluation worker.
struct GameResult {
    index: usize,
    a_win: u16,
    b_win: u16,
    draw: u16,
    elapsed: Duration,
}

/// Load one side's network for a worker; `weight_id < 0` means "no network" (random
/// playout). Each worker loads its own sessions: an inference session is bound to the
/// thread that owns it, so sharing one across workers would serialize them.
fn load_eval_model(
    weights_dir: &Path,
    weight_id: i32,
    intra_thread_num: u8,
) -> Result<Option<Rc<RefCell<NeuralNetwork>>>, String> {
    if weight_id < 0 {
        return Ok(None);
    }
    let model_path = weights_dir.join(format!("{weight_id}.onnx"));
    match NeuralNetwork::new(
        &model_path,
        cfg::MAX_BATCH_SIZE as usize,
        intra_thread_num,
    ) {
        Ok(model) => Ok(Some(Rc::new(RefCell::new(model)))),
        Err(error) => Err(format!(
            "load weight {weight_id} from {} error: {error}",
            model_path.display()
        )),
    }
}

/// Play the assigned games inside one worker thread, reporting how long its own ONNX
/// session creation took.
///
/// Each worker builds its own current-thread runtime and owns its ONNX sessions, so
/// nothing `Rc`-based crosses a thread boundary. Games are independent, so the merged
/// result is identical to running them one at a time; only the wall-clock differs.
#[allow(clippy::too_many_arguments)]
fn worker_eval_games(
    worker: usize,
    weights_dir: &Path,
    weight_a_id: i32,
    weight_b_id: i32,
    assigned: &[(usize, ScheduledGame)],
    num_mcts_sim_a: u16,
    num_mcts_sim_b: u16,
    do_render: bool,
    intra_thread_num: u8,
) -> Result<(Vec<GameResult>, Duration), String> {
    // session creation is reported separately: a cold runtime or a heavy weight shows up
    // here, and lumping it into the game time would hide it
    let load_started = Instant::now();
    let model_a = load_eval_model(weights_dir, weight_a_id, intra_thread_num)?;
    let model_b = load_eval_model(weights_dir, weight_b_id, intra_thread_num)?;
    let load_elapsed = load_started.elapsed();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("create worker runtime error: {error}"))?;

    let mut results = Vec::with_capacity(assigned.len());
    for (index, scheduled) in assigned {
        println!(
            "Eval game {} start... (worker {}, a_first={}, opening: {})",
            index + 1,
            worker,
            scheduled.a_first,
            describe_opening(scheduled.opening.as_ref())
        );
        let started = Instant::now();
        let played = runtime.block_on(play_eval_game(EvalGame {
            nn_a: model_a.clone(),
            nn_b: model_b.clone(),
            a_first: scheduled.a_first,
            do_render,
            num_mcts_sim_a,
            num_mcts_sim_b,
            opening: scheduled.opening.as_ref(),
        }));
        if let Some((a_win, b_win, draw)) = played {
            results.push(GameResult {
                index: *index,
                a_win,
                b_win,
                draw,
                elapsed: started.elapsed(),
            });
        }
    }
    Ok((results, load_elapsed))
}

/// How many games to play concurrently (`EVAL_WORKERS`, default 2).
///
/// Every worker owns its own inference sessions, so this is a wall-clock lever only:
/// the games are independent and the merged result is the same for any worker count.
fn eval_workers() -> usize {
    env::var("EVAL_WORKERS")
        .ok()
        .as_deref()
        .and_then(parse_positive::<usize>)
        .unwrap_or(2)
}

/// Gather the game results reported by the workers, in game order, together with the
/// slowest worker's session-creation time (workers load in parallel, so the maximum —
/// not the sum — is the wall-clock they added).
async fn collect_eval_games(
    weights_dir: &Path,
    weight_a_id: i32,
    weight_b_id: i32,
    games: &[ScheduledGame],
    num_mcts_sim_a: u16,
    num_mcts_sim_b: u16,
    workers: usize,
) -> Result<(Vec<GameResult>, Duration), String> {
    // fair share of the CPU threads when several workers run at once, mirroring the
    // self-play generator; never below two so a single session is not left single-threaded
    let intra_thread_num = ((cfg::DEFAULT_INTRA_THREAD_NUM as usize) / workers).max(2) as u8;
    // interleaved board dumps from concurrent games are unreadable and cost I/O, so
    // rendering only survives the single-worker path
    let do_render = cfg::RENDER_AT_EVAL && workers == 1;
    if cfg::RENDER_AT_EVAL && workers > 1 {
        println!("Eval: board rendering disabled while running {workers} workers (set EVAL_WORKERS=1 to see the boards)");
    }

    let mut handles = Vec::with_capacity(workers);
    for worker in 0..workers {
        let assigned: Vec<(usize, ScheduledGame)> = games
            .iter()
            .cloned()
            .enumerate()
            .filter(|(index, _)| index % workers == worker)
            .collect();
        if assigned.is_empty() {
            continue;
        }
        let weights_dir = weights_dir.to_path_buf();
        handles.push(tokio::task::spawn_blocking(move || {
            worker_eval_games(
                worker,
                &weights_dir,
                weight_a_id,
                weight_b_id,
                &assigned,
                num_mcts_sim_a,
                num_mcts_sim_b,
                do_render,
                intra_thread_num,
            )
        }));
    }

    let mut results = Vec::with_capacity(games.len());
    let mut load_elapsed = Duration::ZERO;
    for handle in handles {
        match handle.await {
            Ok(Ok((mut worker_results, worker_load))) => {
                results.append(&mut worker_results);
                load_elapsed = load_elapsed.max(worker_load);
            }
            Ok(Err(error)) => return Err(error),
            Err(error) => return Err(format!("evaluation worker panicked: {error}")),
        }
    }
    results.sort_by_key(|result| result.index);
    Ok((results, load_elapsed))
}

/// Play the scheduled games and fold them into an [`EvalOutcome`].
///
/// The games are played by `workers` threads that each own their ONNX sessions; the
/// results are merged in game order, so the outcome does not depend on the worker count.
async fn run_eval_games(
    weights_dir: &Path,
    weight_a_id: i32,
    weight_b_id: i32,
    games: &[ScheduledGame],
    num_mcts_sim_a: u16,
    num_mcts_sim_b: u16,
) -> EvalOutcome {
    let started = Instant::now();
    let workers = eval_workers().min(games.len().max(1));
    let (results, load_elapsed) = match collect_eval_games(
        weights_dir,
        weight_a_id,
        weight_b_id,
        games,
        num_mcts_sim_a,
        num_mcts_sim_b,
        workers,
    )
    .await
    {
        Ok(collected) => collected,
        Err(error) => {
            // never report a partial evaluation as a score for net A
            eprintln!("Evaluation failed, treating it as no result: {error}");
            return EvalOutcome::default();
        }
    };

    let mut outcome = EvalOutcome {
        elapsed: started.elapsed(),
        load_elapsed,
        ..EvalOutcome::default()
    };
    let mut game_scores = Vec::with_capacity(results.len());
    for result in &results {
        outcome.a_win += result.a_win;
        outcome.b_win += result.b_win;
        outcome.draw += result.draw;
        game_scores.push(f64::from(result.a_win) + cfg::DRAW_SCORE * f64::from(result.draw));
        println!(
            "Eval game {} end. Current result: a_win={}, b_win={}, draw={} ({:.1}s)",
            result.index + 1,
            outcome.a_win,
            outcome.b_win,
            outcome.draw,
            result.elapsed.as_secs_f64()
        );
        if game_scores.len().is_multiple_of(2) {
            println!(
                "Eval pair {} end: a score {:.2} (colour swapped, {:.1}s so far)",
                game_scores.len() / 2,
                pair_scores_from_games(&game_scores)
                    .last()
                    .copied()
                    .unwrap_or_default(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    outcome.pair_scores = pair_scores_from_games(&game_scores);
    // a game that produced no result at all (an unusable opening) leaves the outcome
    // short of the schedule, which the caller must not read as a score
    outcome.complete = results.len() == games.len();
    if !outcome.complete {
        eprintln!(
            "Evaluation incomplete: {} of {} games produced a result",
            results.len(),
            games.len()
        );
    }

    outcome
}

pub async fn eval(
    weight_a_id: i32,
    weight_b_id: i32,
    game_num: u16,
    num_mcts_sim_a: u16,
    num_mcts_sim_b: u16,
    openings: &[Opening],
) -> EvalOutcome {
    if game_num == 0 {
        return EvalOutcome::default();
    }
    if !openings.is_empty() && !game_num.is_multiple_of(2) {
        // the odd game cannot complete a colour-swapped pair, so it would bias the
        // per-pair statistic: report it and leave it to the game count only
        println!(
            "Eval: game_num {} is odd, the last game does not complete a pair \
             (use an even NUM_CONTEST for the pair statistic)",
            game_num
        );
    }

    let cur_path = env::current_dir().expect("Unable to get current folder");
    let weights_dir = cur_path.join("weights");
    // preflight on the file rather than on a loaded session: the workers load their own
    // sessions, and a missing weight must not be reported as a score for net A
    for weight_id in [weight_a_id, weight_b_id] {
        if weight_id >= 0 && !weights_dir.join(format!("{weight_id}.onnx")).exists() {
            eprintln!(
                "Weight {weight_id}.onnx not found in {}, no evaluation result",
                weights_dir.display()
            );
            return EvalOutcome::default();
        }
    }

    let games = schedule(openings, game_num, cfg::BOARD_SIZE);
    println!(
        "Eval: {} games, {} colour-swapped pairs, {} opening(s) in the book",
        games.len(),
        pair_count(games.len()),
        openings.len()
    );
    run_eval_games(
        &weights_dir,
        weight_a_id,
        weight_b_id,
        &games,
        num_mcts_sim_a,
        num_mcts_sim_b,
    )
    .await
}

/// Load the persisted Elo ratings from elo.txt (weight_id -> elo)
fn load_elo() -> HashMap<i32, f64> {
    let mut ratings = HashMap::new();
    if let Ok(content) = fs::read_to_string("elo.txt") {
        for line in content.lines() {
            let mut iter = line.split_whitespace();
            if let (Some(id), Some(elo)) = (iter.next(), iter.next()) {
                if let (Ok(id), Ok(elo)) = (id.parse::<i32>(), elo.parse::<f64>()) {
                    ratings.insert(id, elo);
                }
            }
        }
    }
    ratings
}

/// Write the Elo ratings back to elo.txt
fn save_elo(ratings: &HashMap<i32, f64>) {
    let mut ids: Vec<i32> = ratings.keys().copied().collect();
    ids.sort_unstable();
    let content = ids
        .iter()
        .map(|id| format!("{} {:.1}", id, ratings[id]))
        .collect::<Vec<_>>()
        .join("\n");
    if !content.is_empty() {
        _ = fs::write("elo.txt", content + "\n");
    }
}

/// A new candidate weight inherits its parent's (current best) Elo rating as the starting point.
/// The candidate is trained from best's games, so its floor is best; if it always restarted
/// from ELO_INITIAL, the candidate's rating after beating best could be lower than best's
/// original rating, leaving elo.txt stagnant for a long time. Overwrite semantics: the
/// candidate id may reuse an old id from a rejected round; that old model is discarded,
/// so its leftover rating must not affect the new model.
fn inherit_elo(new_weight: i32, parent_weight: i32) {
    let mut ratings = load_elo();
    let parent_rating = ratings
        .get(&parent_weight)
        .copied()
        .unwrap_or(cfg::ELO_INITIAL);
    ratings.insert(new_weight, parent_rating);
    save_elo(&ratings);
}

/// Update both sides' Elo from one evaluation match result, returning a log description
fn update_elo(weight_a: i32, weight_b: i32, outcome: &EvalOutcome) -> String {
    let total = outcome.a_win + outcome.b_win + outcome.draw;
    if total == 0 {
        return String::new();
    }
    // score rate of a: 1 point per win, DRAW_SCORE per draw
    let score_a = outcome.win_ratio();

    let mut ratings = load_elo();
    let rating_a = ratings.get(&weight_a).copied().unwrap_or(cfg::ELO_INITIAL);
    let rating_b = ratings.get(&weight_b).copied().unwrap_or(cfg::ELO_INITIAL);

    // standard Elo: expected win rate = 1 / (1 + 10^((Rb - Ra) / 400))
    let expected_a = 1.0 / (1.0 + 10f64.powf((rating_b - rating_a) / 400.0));
    let delta = cfg::ELO_K * (score_a - expected_a);
    let new_a = rating_a + delta;
    let new_b = rating_b - delta;
    ratings.insert(weight_a, new_a);
    ratings.insert(weight_b, new_b);
    save_elo(&ratings);

    // match performance Elo diff implied by this score rate (a relative to b),
    // clamped to avoid 0/1 extremes
    let s = score_a.clamp(1e-3, 1.0 - 1e-3);
    let perf_diff = -400.0 * (1.0 / s - 1.0).log10();

    format!(
        "Elo: {}-th {:.1}->{:.1} ({:+0.1}), {}-th {:.1}->{:.1} ({:+0.1}), match diff {:+0.1}\n",
        weight_a, rating_a, new_a, delta, weight_b, rating_b, new_b, -delta, perf_diff
    )
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args[1] == "prepare" {
        println!("Prepare for training.");
        _ = fs::create_dir("data");
        _ = fs::create_dir("weights");

        let mut f_1 =
            fs::File::create("current_and_best_weight.txt").expect("Unable to create file");
        _ = f_1.write_all("0 0".as_bytes());

        let mut f_2 = fs::File::create("random_mcts_number.txt").expect("Unable to create file");
        _ = f_2.write_all(cfg::DEFAULT_SIMULATION_NUM.to_string().as_bytes());

        // seed the colour-paired opening book so the file documents itself; an
        // existing book is never overwritten, and a missing one falls back to the
        // built-in book at evaluation time
        if !std::path::Path::new(openings::OPENING_FILE).exists() {
            let book = openings::default_openings(cfg::BOARD_SIZE, cfg::N_IN_ROW);
            _ = fs::write(openings::OPENING_FILE, openings::format_book(&book));
            println!("Wrote {} with {} opening(s).", openings::OPENING_FILE, book.len());
        }
        println!("Next: Generate initial weight by python.");
    } else if args[1] == "generate" && args.len() == 3 {
        let start_batch_id: u16 = args[2].parse().expect("Parameter Error!!!");
        println!(
            "Generate {}-{} -th batch.",
            start_batch_id,
            start_batch_id + cfg::NUM_2_SELF_PLAY - 1
        );

        if let Ok(content) = fs::read_to_string("current_and_best_weight.txt") {
            let mut iter = content.split_whitespace();
            let cur_weight: i32 = iter.next().and_then(|s| s.parse().ok()).unwrap_or(-1);
            let best_weight: i32 = iter.next().and_then(|s| s.parse().ok()).unwrap_or(-1);

            if best_weight < 0 {
                println!("LOAD error,check current_and_best_weight.txt");
                return;
            } else {
                // self-play data should be generated by the accepted best weight (AlphaZero flow)
                println!(
                    "Generating... best_weight = {} current_weight = {} start batch id: {}",
                    best_weight, cur_weight, start_batch_id
                );
                generate_data_for_train(best_weight as u16, start_batch_id).await;
            }
        } else {
            println!("Read current_and_best_weight.txt error!!!");
        }
    } else if args[1] == "eval_with_winner" && args.len() == 3 {
        let mut current_weight = 0;
        let mut best_weight = 0;
        if let Ok(content) = fs::read_to_string("current_and_best_weight.txt") {
            let mut iter = content.split_whitespace();
            if let Some(item) = iter.next() {
                current_weight = item.parse().unwrap();
            }
            if let Some(item) = iter.next() {
                best_weight = item.parse().unwrap();
            }

            println!(
                "Current weight: {}, Best weight: {}",
                current_weight, best_weight
            );

            if current_weight >= 0 && current_weight == best_weight {
                // Current and best are the same weight (a re-run right after a rollback, or
                // a round that was accepted without training in between). Playing it against
                // itself would burn a round, report a meaningless result out of search
                // noise, and jitter the single shared Elo rating.
                let message = format!(
                    "current weight {current_weight} == best weight {best_weight}: \
                     nothing to evaluate, best kept\n"
                );
                println!("{}", message.trim_end());
                fs::write(
                    "current_and_best_weight.txt",
                    best_weight.to_string() + " " + &best_weight.to_string(),
                )
                .expect("Unable to write file");
                fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("eval_result.log")
                    .expect("Unable to open file")
                    .write_all(message.as_bytes())
                    .expect("Unable to write data");
                return;
            }

            let game_num: u16 = args[2].parse().expect("Parameter Error!!!");
            let (sims, sims_source) = eval_sims_for(current_weight.max(0) as u16);
            println!("Eval sims: {} per move ({})", sims, sims_source);
            let num_mcts_sim_a: u16 = sims;
            let num_mcts_sim_b: u16 = sims;
            let book = load_openings(
                std::path::Path::new(openings::OPENING_FILE),
                cfg::BOARD_SIZE,
                cfg::N_IN_ROW,
            );
            let result = eval(
                current_weight,
                best_weight,
                game_num,
                num_mcts_sim_a,
                num_mcts_sim_b,
                &book,
            )
            .await;

            let mut result_log_info = current_weight.to_string()
                + "-th weight win: "
                + &result.a_win.to_string()
                + " "
                + &best_weight.to_string()
                + "-th weight win: "
                + &result.b_win.to_string()
                + " tie:"
                + &result.draw.to_string()
                + "\n";
            // the candidate inherits best's rating as its starting point, so lineage ratings
            // accumulate monotonically across iterations
            inherit_elo(current_weight, best_weight);
            let elo_info = update_elo(current_weight, best_weight, &result);
            result_log_info.push_str(&elo_info);
            let win_ratio = result.win_ratio();
            // Paired, colour-swapped openings: the first-move advantage cancels inside
            // each pair, so a sign test over pair scores is a much sharper statement
            // than the raw win rate over the same number of independent games.
            let (pair_wins, pair_losses, pair_ties) = result.pair_decisions();
            if !result.pair_scores.is_empty() {
                let p_value = sign_test_p_value(pair_wins, pair_wins + pair_losses);
                result_log_info.push_str(&format!(
                    "paired openings: {} pairs ({} W / {} L / {} D), sign test p = {:.4} \
                     over {} decisive pair(s)\n",
                    result.pair_scores.len(),
                    pair_wins,
                    pair_losses,
                    pair_ties,
                    p_value,
                    pair_wins + pair_losses
                ));
                println!(
                    "Paired openings: {} pairs ({} W / {} L / {} D), sign test p = {:.4}",
                    result.pair_scores.len(),
                    pair_wins,
                    pair_losses,
                    pair_ties,
                    p_value
                );
            }
            // wall-clock accounting: the number to budget a Colab session with. Session
            // creation is per-process, so it is reported apart from the play rate that
            // scales with the game count.
            let cost_info = format!(
                "eval cost: {} pairs / {} games, load {:.1}s + games {:.1}s = {:.1}s, \
                 {:.1}s per pair total / {:.1}s per pair of play, {:.0} sims/move ({}), \
                 {} worker(s)\n",
                result.pair_scores.len(),
                result.a_win + result.b_win + result.draw,
                result.load_elapsed.as_secs_f64(),
                result.games_elapsed().as_secs_f64(),
                result.elapsed.as_secs_f64(),
                result.seconds_per_pair(),
                result.games_seconds_per_pair(),
                sims,
                sims_source,
                eval_workers(),
            );
            result_log_info.push_str(&cost_info);
            println!("{}", cost_info.trim_end());

            if !result.complete || result.a_win + result.b_win + result.draw == 0 {
                // no usable result: never promote on a failed evaluation
                result_log_info.push_str("evaluation produced no usable result, candidate rejected\n");
                fs::write(
                    "current_and_best_weight.txt",
                    best_weight.to_string() + " " + &best_weight.to_string(),
                )
                .expect("Unable to write file");
            } else if win_ratio > cfg::UPDATE_THRESHOLD {
                result_log_info = result_log_info
                    + "new best weight: "
                    + &current_weight.to_string()
                    + " generated!!!\n";
                fs::write(
                    "current_and_best_weight.txt",
                    current_weight.to_string() + " " + &current_weight.to_string(),
                )
                .expect("Unable to write file");
            } else {
                // candidate rejected: roll back to best and retrain from best next round (AlphaZero flow)
                result_log_info = result_log_info
                    + "candidate rejected, rollback to best weight: "
                    + &best_weight.to_string()
                    + "\n";
                fs::write(
                    "current_and_best_weight.txt",
                    best_weight.to_string() + " " + &best_weight.to_string(),
                )
                .expect("Unable to write file");
            }
            println!("{}", result_log_info);
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("eval_result.log")
                .expect("Unable to open file")
                .write_all(result_log_info.as_bytes())
                .expect("Unable to write data");
        } else {
            eprintln!("Failed to read current and best weights!!!");
        }
    } else if args[1] == "eval_with_random" && args.len() == 3 {
        let mut current_weight_id = 0;
        let mut best_weight_id = 0;
        if let Ok(content) = fs::read_to_string("current_and_best_weight.txt") {
            let mut iter = content.split_whitespace();
            if let Some(item) = iter.next() {
                current_weight_id = item.parse().unwrap();
            }
            if let Some(item) = iter.next() {
                best_weight_id = item.parse().unwrap();
            }

            let mut num_random_mcts_sim = 0;
            if let Ok(content) = fs::read_to_string("random_mcts_number.txt") {
                num_random_mcts_sim = content.trim().parse().unwrap();
            } else {
                eprintln!("Failed to read random MCTS number!!!");
            }

            let game_num: u16 = args[2].parse().expect("Parameter Error!!!");
            let (sims, sims_source) = eval_sims_for(current_weight_id.max(0) as u16);
            println!("Eval sims: {} per move ({})", sims, sims_source);
            let num_mcts_sim_a: u16 = sims;
            let num_mcts_sim_b: u16 = num_random_mcts_sim as u16;
            // the random opponent is not a weight, so no opening book is meaningful here
            let result = eval(
                current_weight_id,
                -1,
                game_num,
                num_mcts_sim_a,
                num_mcts_sim_b,
                &[],
            )
            .await;

            let mut result_log_info = current_weight_id.to_string()
                + "-th weight with mcts ["
                + &num_mcts_sim_a.to_string()
                + "] win: "
                + &result.a_win.to_string()
                + " Random with mcts ["
                + &num_mcts_sim_b.to_string()
                + "] win: "
                + &result.b_win.to_string()
                + " tie: "
                + &result.draw.to_string()
                + "\n";
            // the candidate likewise inherits best's rating, keeping the same rating scale as
            // eval_with_winner; the random baseline (-1) drifts naturally as a fixed-strength anchor
            inherit_elo(current_weight_id, best_weight_id);
            let elo_info = update_elo(current_weight_id, -1, &result);
            result_log_info.push_str(&elo_info);
            let win_ratio = result.win_ratio();
            // wall-clock accounting, same as eval_with_winner (no pair statistic here:
            // the random anchor is not a colour-paired weight)
            let cost_info = format!(
                "eval cost: {} games, load {:.1}s + games {:.1}s = {:.1}s, \
                 {} sims/move ({}), {} worker(s)\n",
                result.a_win + result.b_win + result.draw,
                result.load_elapsed.as_secs_f64(),
                result.games_elapsed().as_secs_f64(),
                result.elapsed.as_secs_f64(),
                sims,
                sims_source,
                eval_workers(),
            );
            result_log_info.push_str(&cost_info);
            println!("{}", cost_info.trim_end());

            if !result.complete || result.a_win + result.b_win + result.draw == 0 {
                result_log_info
                    .push_str("evaluation produced no usable result, candidate rejected\n");
            } else if win_ratio > cfg::UPDATE_THRESHOLD {
                result_log_info = result_log_info
                    + "new best weight: "
                    + &current_weight_id.to_string()
                    + " generated!!!\n";
                fs::write(
                    "current_and_best_weight.txt",
                    current_weight_id.to_string() + " " + &current_weight_id.to_string(),
                )
                .expect("Unable to write file");
            }
            println!("{}", result_log_info);
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("eval_result.log")
                .expect("Unable to open file")
                .write_all(result_log_info.as_bytes())
                .expect("Unable to write data");
        } else {
            eprintln!("Failed to read current and best weights!!!");
        }
    } else {
        println!("Hello, world!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Board points currently occupied, in `(index, color)` form.
    fn occupied(game: &Gomoku) -> Vec<(u16, Color)> {
        let mut stones = Vec::new();
        for row in 0..game.get_board_size() {
            for col in 0..game.get_board_size() {
                let color = game.get_board()[row as usize][col as usize];
                if color != Color::Blank {
                    stones.push((
                        u16::from(row) * u16::from(game.get_board_size()) + u16::from(col),
                        color,
                    ));
                }
            }
        }
        stones
    }

    #[test]
    fn empty_board_position_is_unchanged() {
        let mut game = build_eval_position(None).expect("the default board must build");
        assert_eq!(game.get_last_move(), -1);
        assert_eq!(*game.get_cur_color(), Color::Black);
        assert!(occupied(&game).is_empty());
        assert_eq!(game.get_game_status().0, GameStage::Running);
    }

    #[test]
    fn opening_is_loaded_with_black_to_move() {
        let opening = openings::default_openings(cfg::BOARD_SIZE, cfg::N_IN_ROW)
            .into_iter()
            .next()
            .expect("the built-in book must not be empty");
        let mut game = build_eval_position(Some(&opening)).expect("the opening must load");
        assert_eq!(*game.get_cur_color(), Color::Black);
        assert_eq!(
            game.get_last_move(),
            opening.stones().last().expect("opening has stones").0 as i16
        );
        let mut placed = occupied(&game);
        let mut expected = opening.stones().to_vec();
        placed.sort_by_key(|&(index, _)| index);
        expected.sort_by_key(|&(index, _)| index);
        assert_eq!(placed, expected, "every opening stone must be on the board");
        assert_eq!(game.get_game_status().0, GameStage::Running);
    }

    #[test]
    fn a_pair_starts_from_colour_swapped_positions() {
        let book = openings::default_openings(cfg::BOARD_SIZE, cfg::N_IN_ROW);
        let games = schedule(&book, 4, cfg::BOARD_SIZE);
        let first = build_eval_position(games[0].opening.as_ref()).expect("first game builds");
        let second = build_eval_position(games[1].opening.as_ref()).expect("second game builds");

        let mut first_stones = occupied(&first);
        let mut second_stones = occupied(&second);
        first_stones.sort_by_key(|&(index, _)| index);
        second_stones.sort_by_key(|&(index, _)| index);
        assert_eq!(first_stones.len(), second_stones.len());
        for ((index_a, color_a), (index_b, color_b)) in first_stones.iter().zip(second_stones.iter())
        {
            assert_eq!(index_a, index_b, "the pair uses the same points");
            assert_ne!(color_a, color_b, "the pair must exchange the colours");
        }
        // and the game whose index is odd gives the White stones to net A
        assert!(games[0].a_first);
        assert!(!games[1].a_first);
    }

    #[test]
    fn win_ratio_and_pair_decisions_follow_the_scoring_convention() {
        let outcome = EvalOutcome {
            a_win: 3,
            b_win: 1,
            draw: 0,
            pair_scores: vec![1.0, 0.5, 0.25, 0.0],
            ..EvalOutcome::default()
        };
        assert!((outcome.win_ratio() - 0.75).abs() < 1e-12);
        // 1.0 counts as a pair win, 0.0 as a loss, 0.5 as a tie; 0.25 is a loss
        assert_eq!(outcome.pair_decisions(), (1, 2, 1));
        // a failed evaluation reports zero games and must never look like a score
        let failed = EvalOutcome::default();
        assert_eq!(failed.win_ratio(), 0.0);
        assert!(!failed.complete);
        assert_eq!(failed.seconds_per_pair(), 0.0);

        let empty = EvalOutcome {
            a_win: 0,
            b_win: 0,
            draw: 0,
            pair_scores: Vec::new(),
            ..EvalOutcome::default()
        };
        assert_eq!(empty.win_ratio(), 0.0);
        assert_eq!(empty.pair_decisions(), (0, 0, 0));

        let all_draws = EvalOutcome {
            a_win: 0,
            b_win: 0,
            draw: 4,
            pair_scores: vec![0.5, 0.5],
            ..EvalOutcome::default()
        };
        assert!((all_draws.win_ratio() - 0.5).abs() < 1e-12);
        assert_eq!(all_draws.pair_decisions(), (0, 0, 2));
    }

    #[test]
    fn eval_cost_reports_seconds_per_pair() {
        let outcome = EvalOutcome {
            a_win: 2,
            b_win: 2,
            draw: 0,
            pair_scores: vec![0.5, 0.5],
            elapsed: Duration::from_secs_f64(120.0),
            load_elapsed: Duration::from_secs_f64(0.5),
            complete: true,
        };
        // the unit to budget an evaluation in: wall-clock per colour-swapped pair
        assert!((outcome.seconds_per_pair() - 60.0).abs() < 1e-9);
        // session creation is per-process, so the play rate is reported apart from it:
        // this is the rate that scales when a screen adds more pairs
        assert!((outcome.games_elapsed().as_secs_f64() - 119.5).abs() < 1e-9);
        assert!((outcome.games_seconds_per_pair() - 59.75).abs() < 1e-9);
    }

    #[test]
    fn positive_overrides_ignore_zero_and_garbage() {
        assert_eq!(parse_positive::<u16>("400"), Some(400));
        assert_eq!(parse_positive::<u16>("  384 "), Some(384));
        assert_eq!(parse_positive::<u16>("0"), None);
        assert_eq!(parse_positive::<u16>("-5"), None);
        assert_eq!(parse_positive::<u16>(""), None);
        assert_eq!(parse_positive::<u16>("many"), None);
        assert_eq!(parse_positive::<usize>("2"), Some(2));
        assert_eq!(parse_positive::<usize>("0"), None);
    }

    #[test]
    fn game_scores_fold_into_colour_swapped_pairs() {
        // a pair is (A as Black, A as White); an unpaired last game is dropped
        let scores = pair_scores_from_games(&[1.0, 0.0, 0.5, 0.5, 1.0]);
        assert_eq!(scores, vec![0.5, 0.5]);
        assert!(pair_scores_from_games(&[]).is_empty());
        assert!(pair_scores_from_games(&[1.0]).is_empty());
        // 0.5 for a draw, so a drawn pair scores 0.5 for both games
        assert_eq!(pair_scores_from_games(&[0.5, 0.5]), vec![0.5]);
        // one win and one loss is an even pair
        assert_eq!(pair_scores_from_games(&[1.0, 0.0]), vec![0.5]);
        // the pairing must be colour balanced, so hitting UPDATE_THRESHOLD needs both
        // games: a single win cannot carry a pair over 0.55
        assert_eq!(pair_scores_from_games(&[1.0, 0.5]), vec![0.75]);
    }

    /// Opt-in end-to-end check of the paired evaluation with a real ONNX weight:
    ///   cargo test --bin train_and_eval -- --ignored --nocapture
    /// The weight must match the current `INPUT_CHANNEL_SIZE` (the older local
    /// `1204.onnx` is a 3-channel model and would be rejected by ONNX Runtime), and
    /// the self-play data plus ONNX weights are Colab artifacts, hence the ignore.
    #[tokio::test]
    #[ignore = "needs a matching local ONNX weight and a few seconds per game"]
    async fn paired_schedule_runs_with_a_real_model() {
        let weights_dir = Path::new("build/weights");
        if !weights_dir.join("1205.onnx").exists() {
            eprintln!("skipping: {}/1205.onnx not found", weights_dir.display());
            return;
        }
        let book = openings::default_openings(cfg::BOARD_SIZE, cfg::N_IN_ROW);
        let games = schedule(&book, 2, cfg::BOARD_SIZE);
        // one batch is the floor, so a small budget still exercises the search; the
        // workers load their own sessions, which is what the parallel path relies on
        let outcome = run_eval_games(weights_dir, 1205, 1205, &games, 4, 4).await;

        assert!(outcome.complete, "both games of the pair must produce a result");
        assert_eq!(
            outcome.a_win + outcome.b_win + outcome.draw,
            2,
            "both games of the pair must produce a result"
        );
        assert_eq!(outcome.pair_scores.len(), 1);
        // the pair is a mirror, so a model playing itself is expected to split it
        // (0.5); with only a few simulations the two searches are not exactly
        // symmetric, so only the range is asserted
        assert!(
            (0.0..=1.0).contains(&outcome.pair_scores[0]),
            "pair score {} is out of range",
            outcome.pair_scores[0]
        );
        println!(
            "real-model pair: a_win={}, b_win={}, draw={}, pair score {:.2}, \
             {:.1}s with {} worker(s)",
            outcome.a_win,
            outcome.b_win,
            outcome.draw,
            outcome.pair_scores[0],
            outcome.elapsed.as_secs_f64(),
            eval_workers()
        );
    }

    #[test]
    fn an_already_decided_opening_is_rejected() {
        // see openings::tests for the parser-level check; here the contract that
        // matters is that the schedule never hands a decided position to the search
        let book = openings::default_openings(cfg::BOARD_SIZE, cfg::N_IN_ROW);
        for scheduled in schedule(&book, 6, cfg::BOARD_SIZE) {
            let Some(opening) = scheduled.opening.as_ref() else {
                panic!("a book schedule must always carry an opening");
            };
            let mut game = build_eval_position(Some(opening)).expect("book openings must load");
            assert_eq!(
                game.get_game_status().0,
                GameStage::Running,
                "an evaluation must start from a running position"
            );
        }
    }

    /// End-to-end smoke test of the changed code path: a game that starts from a book
    /// opening must still play out to exactly one result. No network is attached, so
    /// the search uses uniform priors and finishes quickly.
    #[tokio::test]
    async fn a_book_opening_plays_out_to_one_result() {
        let opening = openings::default_openings(cfg::BOARD_SIZE, cfg::N_IN_ROW)
            .into_iter()
            .next()
            .expect("the built-in book must not be empty");
        let result = play_eval_game(EvalGame {
            nn_a: None,
            nn_b: None,
            a_first: true,
            do_render: false,
            num_mcts_sim_a: 1,
            num_mcts_sim_b: 1,
            opening: Some(&opening),
        })
        .await
        .expect("a book opening must be playable");

        assert_eq!(
            result.0 + result.1 + result.2,
            1,
            "one game must produce exactly one result, got {result:?}"
        );
    }
}
