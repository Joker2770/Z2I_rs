// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use rand;
use rand_distr::{Distribution, multi::Dirichlet};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    env, fs,
    io::Write,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    configuration::cfg,
    gomoku::{GameStage, Gomoku},
    mcts::MCTS,
    ortopt::NeuralNetwork,
    rule::Color,
};

/// Move-selection temperature after `step` plies of a self-play game.
///
/// The decay length is derived from the profile's own anchors instead of being
/// hand-tuned, which pins both ends of the schedule: `temp_at(EXPLORE_STEP)` is
/// `EXPLORE_TEMP`, and `temp_at(GREEDY_FROM_STEP)` is exactly `GREEDY_TEMP`, so the
/// schedule is always spent exactly by the ply the profile is expected to reach and
/// every later ply selects greedily. Changing `GREEDY_FROM_STEP` therefore only moves
/// the ply after which the game stops exploring, it does not change how much the opening
/// explores.
pub fn temp_at(step: u16) -> f64 {
    if step >= cfg::GREEDY_FROM_STEP {
        return cfg::GREEDY_TEMP;
    }

    let decay_len = f64::from(cfg::GREEDY_FROM_STEP - cfg::EXPLORE_STEP)
        / (cfg::EXPLORE_TEMP / cfg::GREEDY_TEMP).ln();

    cfg::GREEDY_TEMP.max(
        cfg::EXPLORE_TEMP * (-f64::from(step.saturating_sub(cfg::EXPLORE_STEP)) / decay_len).exp(),
    )
}

fn noise_candidates(board: &[Vec<Color>], legal_moves: &[u8]) -> Vec<usize> {
    let size = board.len();
    if size == 0 {
        return Vec::new();
    }
    assert!(board.iter().all(|row| row.len() == size));
    assert_eq!(legal_moves.len(), size * size);

    let has_stones = board.iter().flatten().any(|color| *color != Color::Blank);
    let mut candidates = Vec::new();
    for (index, legal) in legal_moves.iter().enumerate() {
        if *legal != 1 {
            continue;
        }
        let (row, col) = (index / size, index % size);
        let near_stone = if has_stones {
            let radius = 2usize;
            let row_start = row.saturating_sub(radius);
            let row_end = (row + radius).min(size - 1);
            let col_start = col.saturating_sub(radius);
            let col_end = (col + radius).min(size - 1);
            (row_start..=row_end)
                .any(|r| (col_start..=col_end).any(|c| board[r][c] != Color::Blank))
        } else {
            let center = (size - 1) / 2;
            let radius = (size / 4).clamp(1, 2);
            row.abs_diff(center) <= radius && col.abs_diff(center) <= radius
        };
        if near_stone {
            candidates.push(index);
        }
    }

    // A small or unusual board can leave no point in the preferred region. Do not
    // turn a valid position into a no-noise position solely because of that filter.
    if candidates.is_empty() {
        candidates.extend(
            legal_moves
                .iter()
                .enumerate()
                .filter_map(|(index, legal)| (*legal == 1).then_some(index)),
        );
    }
    candidates
}

fn normalized_policy_entropy(raw_probs: &[f64], legal_moves: &[u8]) -> f64 {
    assert_eq!(raw_probs.len(), legal_moves.len());
    let legal_count = legal_moves.iter().filter(|legal| **legal == 1).count();
    if legal_count <= 1 {
        return 0.0;
    }

    let legal_mass: f64 = raw_probs
        .iter()
        .zip(legal_moves)
        .filter_map(|(prob, legal)| {
            (*legal == 1 && prob.is_finite() && *prob > 0.0).then_some(*prob)
        })
        .sum();
    if !legal_mass.is_finite() || legal_mass <= f64::EPSILON {
        return 0.0;
    }

    let entropy: f64 = raw_probs
        .iter()
        .zip(legal_moves)
        .filter_map(|(prob, legal)| {
            (*legal == 1 && prob.is_finite() && *prob > 0.0).then_some(prob / legal_mass)
        })
        .map(|prob| -prob * prob.ln())
        .sum();
    (entropy / (legal_count as f64).ln()).clamp(0.0, 1.0)
}

fn noise_epsilon(step: u16, raw_probs: &[f64], legal_moves: &[u8]) -> f64 {
    let legal_count = legal_moves.iter().filter(|legal| **legal == 1).count();
    if legal_count <= 2 || step >= cfg::GREEDY_FROM_STEP {
        return 0.0;
    }

    let phase_scale = if step <= cfg::EXPLORE_STEP {
        1.0
    } else {
        f64::from(cfg::GREEDY_FROM_STEP - step)
            / f64::from(cfg::GREEDY_FROM_STEP - cfg::EXPLORE_STEP)
    };

    // A diffuse search policy already explores many moves. Concentrated search
    // policies (often forced tactical replies) should not be overturned by noise.
    cfg::DIRI * phase_scale * normalized_policy_entropy(raw_probs, legal_moves)
}

fn add_self_play_noise(
    action_probs: &mut [f64],
    raw_probs: &[f64],
    legal_moves: &[u8],
    board: &[Vec<Color>],
    step: u16,
    rng: &mut impl rand::Rng,
) {
    assert_eq!(action_probs.len(), raw_probs.len());
    assert_eq!(action_probs.len(), legal_moves.len());
    let epsilon = noise_epsilon(step, raw_probs, legal_moves);
    if epsilon <= 0.0 {
        return;
    }

    let candidates = noise_candidates(board, legal_moves);
    if candidates.len() < 2 {
        return;
    }
    let dirichlet = Dirichlet::new(&vec![cfg::DIRICHLET_ALPHA; candidates.len()]).unwrap();
    let noise = dirichlet.sample(rng);
    let mut mixed_noise = vec![0.0; action_probs.len()];
    for (index, probability) in candidates.into_iter().zip(noise) {
        mixed_noise[index] = probability;
    }
    for (index, legal) in legal_moves.iter().enumerate() {
        if *legal == 1 {
            action_probs[index] =
                (1.0 - epsilon) * action_probs[index] + epsilon * mixed_noise[index];
        }
    }
}

pub struct SelfPlay {
    neural_network: Rc<RefCell<NeuralNetwork>>,
}

/// Forbidden-move endings seen by this process since the last read.
///
/// Self-play is spread over threads, so the counter is process-global; `generate` reports and
/// resets it once per round. With `Gomoku::refresh_playable_moves` in place the engine cannot
/// play a forbidden point any more, so this should stay 0: a non-zero value means the search
/// and the judge disagree again (or the games came from files written before the fix).
static FORBIDDEN_MOVE_ENDINGS: AtomicUsize = AtomicUsize::new(0);

/// How many games ended by the mover's own forbidden move since the last call, resetting it.
pub fn take_forbidden_move_endings() -> usize {
    FORBIDDEN_MOVE_ENDINGS.swap(0, Ordering::SeqCst)
}

/// Whether self-play games that end by a forbidden move are dropped instead of saved.
///
/// Off by default: the ending should no longer happen, and the counter above is the signal to
/// watch. Turn it on (`SELFPLAY_DROP_FORBIDDEN=1`) if a lineage still produces them and the
/// value targets must stay clean while the cause is being fixed -- note that a window made
/// mostly of such games will then starve instead of training.
fn drop_forbidden_games() -> bool {
    matches!(
        env::var("SELFPLAY_DROP_FORBIDDEN").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

impl SelfPlay {
    pub fn new(nn: Rc<RefCell<NeuralNetwork>>) -> Self {
        Self { neural_network: nn }
    }

    pub async fn play(&self, save_id: u16, simulation_num: usize, board_size: u8, n_in_row: u8) {
        let buffer_len: u16 = board_size as u16 * board_size as u16 + 1;

        let game = Gomoku::new(board_size, n_in_row);
        if let Some(gg) = game {
            let game_ref = Rc::new(RefCell::new(gg));
            let action_size = game_ref.borrow().get_action_size();
            let nn = Some(Rc::clone(&self.neural_network));
            let mut mcts = MCTS::new(
                nn,
                cfg::C_PUCT as f64,
                cfg::C_VIRTUAL_LOSS,
                AtomicUsize::new(simulation_num),
                cfg::DEFAULT_SIM_PER_BATCH_NUM,
                action_size,
            );

            let mut game_status = {
                let mut game = game_ref.borrow_mut();
                *game.get_game_status()
            };
            println!("Game rule: {}", game_ref.borrow().get_rule().bits());

            let mut step = 0u16;
            let mut board_buffer =
                vec![vec![vec![0; board_size as usize]; board_size as usize]; buffer_len as usize];
            let mut v_buffer = vec![0; buffer_len as usize];
            let mut p_buffer =
                vec![vec![0.0; board_size as usize * board_size as usize]; buffer_len as usize];
            let mut color_buffer = vec![0i8; buffer_len as usize];
            let mut last_move_buffer = vec![0; buffer_len as usize];

            let mut rng = rand::rng();

            let mut hasher = Sha256::new();
            while game_status.0 == GameStage::Running {
                let temp = temp_at(step);
                if cfg::RENDER_AT_SELF_PLAY {
                    println!("Step: {}", step);
                    println!("temp: {}", temp);
                }
                // The side to move may have forbidden points (Renju Black): refresh what the
                // search, the Dirichlet noise and the stored target are allowed to use, so none
                // of them can disagree with the judge that ends the game. A no-op for every
                // other rule (see `Gomoku::refresh_playable_moves`).
                game_ref.borrow_mut().refresh_playable_moves();
                let (raw_probs, mut action_probs) = mcts
                    .get_raw_and_tempered_probs(&game_ref.borrow(), temp)
                    .await;
                let board = game_ref.borrow().get_board().clone();
                // the training target stores the raw τ = 1 visit distribution π(a) ∝ N(a)
                // (the AlphaZero convention, decoupled from move-selection temperature);
                // the temperature-sharpened policy and the Dirichlet noise below only affect
                // move selection and never leak into the training target
                for (i, p) in raw_probs.iter().enumerate() {
                    p_buffer[step as usize][i] = *p;
                }
                for i in 0..board.len() {
                    for j in 0..board[i].len() {
                        board_buffer[step as usize][i][j] = if board[i][j] == Color::Black {
                            1
                        } else if board[i][j] == Color::White {
                            -1
                        } else {
                            0
                        };
                    }
                }
                let cur_color = *game_ref.borrow().get_cur_color();
                color_buffer[step as usize] = if cur_color == Color::Black {
                    1
                } else if cur_color == Color::White {
                    -1
                } else {
                    0
                };
                last_move_buffer[step as usize] = game_ref.borrow().get_last_move();

                let lm = game_ref.borrow().get_legal_moves().to_vec();
                // Keep opening exploration broad, focus perturbations around played stones,
                // taper them with game phase and avoid disturbing concentrated tactical search.
                add_self_play_noise(&mut action_probs, &raw_probs, &lm, &board, step, &mut rng);

                let rst = mcts.get_action_by_sample(&action_probs);
                // canonical game fingerprint: the chosen move sequence uniquely determines the
                // stored board/π/v content, so identical games map to the same file (deterministic
                // dedup) and distinct games to distinct files; no random suffix is needed
                hasher.update(rst.to_ne_bytes());
                mcts.update_root_with_action(&game_ref.borrow(), rst);
                if !game_ref.borrow_mut().execute_move(rst) {
                    // defensive: refresh the terminal status before bailing out of a failed move,
                    // so the later win_col_2_i is not computed from a stale status
                    game_status = {
                        let mut game = game_ref.borrow_mut();
                        *game.get_game_status()
                    };
                    break;
                }
                game_status = {
                    let mut game = game_ref.borrow_mut();
                    *game.get_game_status()
                };
                if cfg::RENDER_AT_SELF_PLAY {
                    game_ref.borrow().render();
                    println!();
                }
                step = step.saturating_add(1);
            }

            let win_col_2_i = if game_status.1 == Color::Black {
                1
            } else if game_status.1 == Color::White {
                -1
            } else {
                0
            };
            // Renju's forbidden move ends the game with White winning and the mover losing on
            // its own move. Such a game labels every Black-to-move ply as "losing", which is
            // what collapses a value head onto the colour plane, so it is counted here -- and
            // with SELFPLAY_DROP_FORBIDDEN=1 it is not written at all.
            let forbidden_ending = {
                let mut game = game_ref.borrow_mut();
                let _ = *game.get_game_status();
                game.ended_by_forbidden_move()
            };
            if forbidden_ending {
                FORBIDDEN_MOVE_ENDINGS.fetch_add(1, Ordering::SeqCst);
            }
            println!(
                "Self play: total step num = {} winner = {}{}",
                step,
                win_col_2_i,
                if forbidden_ending {
                    " (forbidden-move ending)"
                } else {
                    ""
                }
            );
            if forbidden_ending && drop_forbidden_games() {
                println!(
                    "Self play: game dropped (SELFPLAY_DROP_FORBIDDEN=1): its labels would all be \
                     decided by the side to move"
                );
                return;
            }
            // length marker guards the degenerate zero-move case; the winner is implied by moves;
            // the rule flag is hashed too so that identical move sequences played under different
            // rules map to distinct data files
            let rule_bits = game_ref.borrow().get_rule().bits();
            hasher.update(&[rule_bits]);
            hasher.update(step.to_ne_bytes());
            let hash_rst = hasher.finalize();
            let hex_string = hex::encode(hash_rst);

            let path = env::current_dir().unwrap();
            let new_path = path
                .join("data")
                .join("data_".to_string() + &save_id.to_string() + "_" + &hex_string);
            println!("Save path: {:?}", new_path);
            new_path
                .parent()
                .map(fs::create_dir_all)
                .transpose()
                .expect("Unable to create dirs");
            // write to a temp file first and rename atomically after finishing,
            // so the training side never reads a half-written data file
            let tmp_path = new_path.with_extension("part");
            let mut file = fs::File::create(&tmp_path).expect("Unable to create file");
            _ = file.write_all(&(step as i32).to_ne_bytes());
            // header carries the rule flag actually used for self-play so training
            // sides can reject samples generated under a different rule
            _ = file.write_all(&(rule_bits as i32).to_ne_bytes());

            for i in 0..step {
                for j in 0..board_size {
                    for k in 0..board_size {
                        _ = file.write_all(
                            &(board_buffer[i as usize][j as usize][k as usize] as i32)
                                .to_ne_bytes(),
                        );
                    }
                }
            }

            for i in 0..step {
                for j in 0..action_size {
                    _ = file.write_all(&(p_buffer[i as usize][j as usize] as f32).to_ne_bytes());
                }
            }

            for i in 0..step {
                let new_v = color_buffer[i as usize] as i32 * win_col_2_i;
                v_buffer[i as usize] = new_v;
                _ = file.write_all(&v_buffer[i as usize].to_ne_bytes());
            }
            for i in 0..step {
                _ = file.write_all(&(color_buffer[i as usize] as i32).to_ne_bytes());
            }
            for i in 0..step {
                _ = file.write_all(&(last_move_buffer[i as usize] as i32).to_ne_bytes());
            }
            drop(file);
            if let Err(error) = fs::rename(&tmp_path, &new_path) {
                eprintln!("Rename data file error: {}", error);
            }
        }
    }

    pub async fn self_play_for_train(
        &self,
        game_num: u16,
        start_batch_id: u16,
        simulation_num: usize,
    ) {
        for i in 0..game_num {
            self.play(
                start_batch_id + i,
                simulation_num,
                cfg::BOARD_SIZE,
                cfg::N_IN_ROW,
            )
            .await;
        }
    }
}

#[cfg(test)]
mod tests {

    use super::{cfg, noise_candidates, noise_epsilon, normalized_policy_entropy};
    use crate::play;
    use crate::rule::Color;

    use rand_distr::{Distribution, multi::Dirichlet};

    #[test]
    fn dirichlet_sample_alpha_0_3() {
        const ALPHA: f64 = 0.3;
        const N: usize = 10;

        let dirichlet = Dirichlet::new(&[ALPHA; N]).unwrap();
        let mut rng = rand::rng();

        let n_samples = 10_000usize;
        let mut sums = vec![0.0; N];
        for _ in 0..n_samples {
            let sample = dirichlet.sample(&mut rng);
            let mut s = 0.0;
            for (idx, x) in sample.iter().enumerate() {
                assert!(x.is_finite(), "dirichlet sample must be finite");
                assert!(*x >= 0.0, "dirichlet sample must be non-negative");
                sums[idx] += *x;
                s += *x;
            }
            assert!((s - 1.0).abs() < 1e-9, "dirichlet sample must sum to 1");
        }

        let expected = 1.0 / N as f64;
        for mean in sums.iter().map(|s| s / n_samples as f64) {
            assert!(
                (mean - expected).abs() < 0.01,
                "component mean {} should be close to expected {}",
                mean,
                expected
            );
        }
    }

    #[test]
    fn noise_schedule_fades_to_zero_before_greedy_moves() {
        let probs = vec![1.0 / 9.0; 9];
        let legal = vec![1; 9];
        assert!((noise_epsilon(cfg::EXPLORE_STEP, &probs, &legal) - cfg::DIRI).abs() < 1e-12);
        assert!(
            noise_epsilon(
                (cfg::EXPLORE_STEP + cfg::GREEDY_FROM_STEP) / 2,
                &probs,
                &legal
            ) < cfg::DIRI
        );
        assert_eq!(noise_epsilon(cfg::GREEDY_FROM_STEP, &probs, &legal), 0.0);
        assert_eq!(
            noise_epsilon(cfg::GREEDY_FROM_STEP.saturating_add(1), &probs, &legal),
            0.0
        );
    }

    #[test]
    fn concentrated_search_and_near_terminal_legal_sets_suppress_noise() {
        let legal = vec![1; 9];
        let one_hot = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(normalized_policy_entropy(&one_hot, &legal), 0.0);
        assert_eq!(noise_epsilon(0, &one_hot, &legal), 0.0);

        let two_legal = vec![1, 1, 0, 0, 0, 0, 0, 0, 0];
        let uncertain = vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(noise_epsilon(0, &uncertain, &two_legal), 0.0);
        assert_eq!(normalized_policy_entropy(&[0.5, 0.5], &[1, 1]), 1.0);
    }

    #[test]
    fn opening_noise_candidates_are_centered_and_include_small_boards() {
        let board = vec![vec![Color::Blank; 15]; 15];
        let legal = vec![1; 15 * 15];
        let candidates = noise_candidates(&board, &legal);
        assert_eq!(candidates.len(), 25);
        assert!(candidates.contains(&(7 * 15 + 7)));
        assert!(!candidates.contains(&0));

        let small_board = vec![vec![Color::Blank; 3]; 3];
        let small_legal = vec![1; 9];
        assert_eq!(noise_candidates(&small_board, &small_legal).len(), 9);
    }

    #[test]
    fn midgame_noise_candidates_stay_near_stones_and_respect_legality() {
        let mut board = vec![vec![Color::Blank; 15]; 15];
        board[7][7] = Color::Black;
        let mut legal = vec![1; 15 * 15];
        legal[7 * 15 + 7] = 0;
        legal[7 * 15 + 8] = 0;

        let candidates = noise_candidates(&board, &legal);
        assert!(candidates.contains(&(5 * 15 + 5)));
        assert!(!candidates.contains(&(7 * 15 + 7)));
        assert!(!candidates.contains(&(7 * 15 + 8)));
        assert!(!candidates.contains(&0));

        legal.fill(0);
        assert!(noise_candidates(&board, &legal).is_empty());
        assert_eq!(noise_epsilon(0, &vec![0.0; 15 * 15], &legal), 0.0);
    }

    #[test]
    fn edge_stone_candidates_use_saturating_board_bounds() {
        let mut board = vec![vec![Color::Blank; 5]; 5];
        board[0][0] = Color::Black;
        let mut legal = vec![1; 25];
        legal[0] = 0;

        let candidates = noise_candidates(&board, &legal);
        assert!(candidates.contains(&1));
        assert!(candidates.contains(&(2 * 5 + 2)));
        assert!(!candidates.contains(&(3 * 5 + 3)));
    }

    #[test]
    fn adding_noise_preserves_probability_mass_and_never_uses_forbidden_points() {
        let board = vec![vec![Color::Blank; 3]; 3];
        let mut legal = vec![1; 9];
        legal[0] = 0;
        let raw_probs = vec![
            0.0,
            1.0 / 8.0,
            1.0 / 8.0,
            1.0 / 8.0,
            1.0 / 8.0,
            1.0 / 8.0,
            1.0 / 8.0,
            1.0 / 8.0,
            1.0 / 8.0,
        ];
        let mut action_probs = raw_probs.clone();
        let mut rng = rand::rng();
        super::add_self_play_noise(&mut action_probs, &raw_probs, &legal, &board, 0, &mut rng);

        assert!((action_probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(action_probs[0], 0.0);
        assert!(
            action_probs
                .iter()
                .all(|prob| prob.is_finite() && *prob >= 0.0)
        );
    }

    #[test]
    fn test_temp_at_explores_until_warmup_step() {
        for step in 0..=cfg::EXPLORE_STEP {
            assert_eq!(play::temp_at(step), cfg::EXPLORE_TEMP);
        }
    }

    #[test]
    fn test_temp_at_is_spent_at_greedy_from_step() {
        // mcts::policy_from_children turns anything at (or below) GREEDY_TEMP into a one-hot
        // greedy selection, so reaching the floor at the anchor is the point of the schedule
        assert_eq!(play::temp_at(cfg::GREEDY_FROM_STEP), cfg::GREEDY_TEMP);

        for step in cfg::GREEDY_FROM_STEP..cfg::GREEDY_FROM_STEP + 200 {
            assert_eq!(play::temp_at(step), cfg::GREEDY_TEMP);
        }
    }

    #[test]
    fn test_temp_at_is_monotonically_decreasing() {
        let mut previous = play::temp_at(0);
        for step in 1..u16::from(cfg::GREEDY_FROM_STEP) * 4 {
            let current = play::temp_at(step);
            assert!(current <= previous, "temp rose at step {step}");
            assert!(current >= cfg::GREEDY_TEMP);
            previous = current;
        }
    }

    #[test]
    fn test_temp_at_interpolates_between_anchors() {
        let mid = (cfg::EXPLORE_STEP + cfg::GREEDY_FROM_STEP) / 2;
        let temp = play::temp_at(mid);

        assert!(temp > cfg::GREEDY_TEMP && temp < cfg::EXPLORE_TEMP);

        // the decay is exponential, so equal spacing means equal ratios: the geometric mean
        // of the two anchors has to be the value halfway between them
        let expected = (cfg::EXPLORE_TEMP * cfg::GREEDY_TEMP).sqrt();
        assert!((temp - expected).abs() < 1e-12);
    }
}
