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
    cell::RefCell, collections::HashMap, env, fs, io::Write, rc::Rc, sync::atomic::AtomicUsize,
};

pub fn sims_for_weight(weight_id: u16) -> usize {
    let boosted = cfg::DEFAULT_SIMULATION_NUM
        + (weight_id as usize / cfg::SIMS_BOOST_EVERY as usize) * cfg::SIMS_BOOST_STEP;
    boosted.min(cfg::SIMS_CAP)
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

async fn play_eval_game(setup: EvalGame<'_>) -> (u16, u16, u16) {
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
    let Some(game) = build_eval_position(opening) else {
        return (0, 0, 0);
    };
    let g_ref = Rc::new(RefCell::new(game));

    let mut game_state = {
        let mut game = g_ref.borrow_mut();
        *game.get_game_status()
    };
    if game_state.0 != GameStage::Running {
        // an opening that is already decided would pick the pair winner from the book
        // file instead of from play, so it must never be counted as a game
        eprintln!("Evaluation opening is already terminal, skipping the game");
        return (0, 0, 0);
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

    (a_win, b_win, draw)
}

/// Outcome of one evaluation: raw game counts plus the score of every complete
/// colour-swapped pair (0 = A lost both, 0.5 = even, 1 = A won both).
pub struct EvalOutcome {
    pub a_win: u16,
    pub b_win: u16,
    pub draw: u16,
    pub pair_scores: Vec<f64>,
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

async fn run_eval_games(
    model_a: Option<Rc<RefCell<NeuralNetwork>>>,
    model_b: Option<Rc<RefCell<NeuralNetwork>>>,
    games: &[ScheduledGame],
    num_mcts_sim_a: u16,
    num_mcts_sim_b: u16,
) -> EvalOutcome {
    let mut outcome = EvalOutcome {
        a_win: 0,
        b_win: 0,
        draw: 0,
        pair_scores: Vec::new(),
    };

    let a = model_a.clone();
    let b = model_b.clone();
    let mut game_scores = Vec::with_capacity(games.len());
    for (game_index, scheduled) in games.iter().enumerate() {
        println!(
            "Eval game {} start... (a_first={}, opening: {})",
            game_index + 1,
            scheduled.a_first,
            describe_opening(scheduled.opening.as_ref())
        );
        let ma = a.clone();
        let mb = b.clone();

        let (a_win, b_win, draw) = play_eval_game(EvalGame {
            nn_a: ma,
            nn_b: mb,
            a_first: scheduled.a_first,
            do_render: cfg::RENDER_AT_EVAL,
            num_mcts_sim_a,
            num_mcts_sim_b,
            opening: scheduled.opening.as_ref(),
        })
        .await;

        outcome.a_win += a_win;
        outcome.b_win += b_win;
        outcome.draw += draw;
        game_scores.push(f64::from(a_win) + cfg::DRAW_SCORE * f64::from(draw));
        if game_scores.len().is_multiple_of(2) {
            println!(
                "Eval pair {} end: a score {:.2} (colour swapped)",
                game_scores.len() / 2,
                pair_scores_from_games(&game_scores)
                    .last()
                    .copied()
                    .unwrap_or_default()
            );
        }
        println!(
            "Eval game {} end. Current result: a_win={}, b_win={}, draw={}",
            game_index + 1,
            outcome.a_win,
            outcome.b_win,
            outcome.draw
        );
    }
    outcome.pair_scores = pair_scores_from_games(&game_scores);

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
    let outcome = EvalOutcome {
        a_win: 0,
        b_win: 0,
        draw: 0,
        pair_scores: Vec::new(),
    };
    if game_num == 0 {
        return outcome;
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
    let load_model = |weight_id: i32| {
        if weight_id < 0 {
            None
        } else {
            let model_path = cur_path
                .join("weights")
                .join(weight_id.to_string() + ".onnx");
            match NeuralNetwork::new(
                &model_path,
                cfg::MAX_BATCH_SIZE as usize,
                cfg::DEFAULT_INTRA_THREAD_NUM,
            ) {
                Ok(model) => Some(model),
                Err(error) => {
                    eprintln!("Load model {} error: {}", model_path.display(), error);
                    None
                }
            }
        }
    };
    let model_a = load_model(weight_a_id).map(|m| Rc::new(RefCell::new(m)));
    let model_b = load_model(weight_b_id).map(|m| Rc::new(RefCell::new(m)));
    if weight_a_id >= 0 && model_a.is_none() || weight_b_id >= 0 && model_b.is_none() {
        return outcome;
    }

    let games = schedule(openings, game_num, cfg::BOARD_SIZE);
    println!(
        "Eval: {} games, {} colour-swapped pairs, {} opening(s) in the book",
        games.len(),
        pair_count(games.len()),
        openings.len()
    );
    run_eval_games(model_a, model_b, &games, num_mcts_sim_a, num_mcts_sim_b).await
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

            let game_num: u16 = args[2].parse().expect("Parameter Error!!!");
            let sims: u16 = sims_for_weight(current_weight.max(0) as u16) as u16;
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
            if win_ratio > cfg::UPDATE_THRESHOLD {
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
            let num_mcts_sim_a: u16 = sims_for_weight(current_weight_id.max(0) as u16) as u16;
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
            if win_ratio > cfg::UPDATE_THRESHOLD {
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
        };
        assert!((outcome.win_ratio() - 0.75).abs() < 1e-12);
        // 1.0 counts as a pair win, 0.0 as a loss, 0.5 as a tie; 0.25 is a loss
        assert_eq!(outcome.pair_decisions(), (1, 2, 1));

        let empty = EvalOutcome {
            a_win: 0,
            b_win: 0,
            draw: 0,
            pair_scores: Vec::new(),
        };
        assert_eq!(empty.win_ratio(), 0.0);
        assert_eq!(empty.pair_decisions(), (0, 0, 0));

        let all_draws = EvalOutcome {
            a_win: 0,
            b_win: 0,
            draw: 4,
            pair_scores: vec![0.5, 0.5],
        };
        assert!((all_draws.win_ratio() - 0.5).abs() < 1e-12);
        assert_eq!(all_draws.pair_decisions(), (0, 0, 2));
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
        let model_path = std::path::Path::new("build/weights/1205.onnx");
        if !model_path.exists() {
            eprintln!("skipping: {} not found", model_path.display());
            return;
        }
        let load = || {
            NeuralNetwork::new(model_path, 16, 2)
                .ok()
                .map(|model| Rc::new(RefCell::new(model)))
        };
        let book = openings::default_openings(cfg::BOARD_SIZE, cfg::N_IN_ROW);
        let games = schedule(&book, 2, cfg::BOARD_SIZE);
        // one batch is the floor, so a small budget still exercises the search
        let outcome = run_eval_games(load(), load(), &games, 4, 4).await;

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
            "real-model pair: a_win={}, b_win={}, draw={}, pair score {:.2}",
            outcome.a_win, outcome.b_win, outcome.draw, outcome.pair_scores[0]
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
        .await;

        assert_eq!(
            result.0 + result.1 + result.2,
            1,
            "one game must produce exactly one result, got {result:?}"
        );
    }
}
