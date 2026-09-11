// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use rand;
use rand_distr::{Distribution, multi::Dirichlet};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, env, fs, io::Write, rc::Rc, sync::atomic::AtomicUsize};

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

pub struct SelfPlay {
    neural_network: Rc<RefCell<NeuralNetwork>>,
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
                // AlphaZero exploration noise: add Dirichlet noise on legal moves
                // η ~ Dir(α),π = (1 - ε)·p + ε·η(ε = cfg::DIRI,α = cfg::DIRICHLET_ALPHA)
                let legal_count = lm
                    .iter()
                    .take(action_probs.len())
                    .filter(|legal| **legal == 1u8)
                    .count();
                if legal_count >= 2 {
                    let dirichlet =
                        Dirichlet::new(&vec![cfg::DIRICHLET_ALPHA; legal_count]).unwrap();
                    let noise = dirichlet.sample(&mut rng);
                    let mut noise_idx = 0usize;
                    for (i, legal) in lm.iter().enumerate().take(action_probs.len()) {
                        if *legal == 1u8 {
                            action_probs[i] =
                                (1.0 - cfg::DIRI) * action_probs[i] + cfg::DIRI * noise[noise_idx];
                            noise_idx += 1;
                        }
                    }
                }

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
            println!(
                "Self play: total step num = {} winner = {}",
                step, win_col_2_i
            );
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

    use super::cfg;
    use crate::play;

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
