// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use rand::{self, RngExt};
use rand_distr::{Distribution, Gamma};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, env, fs, io::Write, ops::Div, rc::Rc};

use crate::{
    configuration::cfg,
    gomoku::{GameStage, Gomoku},
    mcts::MCTS,
    ortopt::NeuralNetwork,
    rule::Color,
};

pub struct SelfPlay {
    neural_network: Rc<RefCell<NeuralNetwork>>,
}

impl SelfPlay {
    pub fn new(nn: Rc<RefCell<NeuralNetwork>>) -> Self {
        Self { neural_network: nn }
    }

    pub async fn play(&self, save_id: u16) {
        const BUFFER_LEN: u16 = cfg::BOARD_SIZE as u16 * cfg::BOARD_SIZE as u16 + 1;
        const GAMMA_SHAPE: f64 = 0.3;
        const GAMMA_SCALE: f64 = 1.0;

        let game = Gomoku::new(cfg::BOARD_SIZE, cfg::N_IN_ROW);
        if let Some(gg) = game {
            let game_ref = Rc::new(RefCell::new(gg));
            let action_size = game_ref.borrow().get_action_size();
            let nn = Some(Rc::clone(&self.neural_network));
            let mut mcts = MCTS::new(
                nn,
                cfg::C_PUCT as f64,
                cfg::C_VIRTUAL_LOSS as f64,
                cfg::DEFAULT_SIMULATION_NUM,
                action_size,
            );

            let mut game_status = {
                let mut game = game_ref.borrow_mut();
                *game.get_game_status()
            };
            println!("Game rule: {}", game_ref.borrow().get_rule().bits());

            let mut step = 0;
            let mut board_buffer =
                vec![
                    vec![vec![0; cfg::BOARD_SIZE as usize]; cfg::BOARD_SIZE as usize];
                    BUFFER_LEN as usize
                ];
            let mut v_buffer = vec![0; BUFFER_LEN as usize];
            let mut p_buffer = vec![
                vec![0.0; cfg::BOARD_SIZE as usize * cfg::BOARD_SIZE as usize];
                BUFFER_LEN as usize
            ];
            let mut color_buffer = vec![0i8; BUFFER_LEN as usize];
            let mut last_move_buffer = vec![0; BUFFER_LEN as usize];

            let gamma = Gamma::new(GAMMA_SHAPE, GAMMA_SCALE).unwrap();
            let mut rng = rand::rng();

            let mut hasher = Sha256::new();
            while game_status.0 == GameStage::running {
                let temp = if step < cfg::EXPLORE_STEP { 1.0 } else { 1e-3 };
                let mut action_probs = mcts.get_action_probs(&game_ref.borrow(), temp).await;
                println!("Step: {}", step);
                let board = game_ref.borrow().get_board().clone();
                for (i, p) in action_probs.iter().enumerate() {
                    p_buffer[step as usize][i] = *p;
                }
                for i in 0..board.len() {
                    for j in 0..board[i as usize].len() {
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
                hasher.update(&lm);
                let mut sum = 0.0;
                for (i, legal) in lm.iter().enumerate().take(action_probs.len()) {
                    if *legal == 1u8 {
                        let noise = cfg::DIRI * gamma.sample(&mut rng);
                        action_probs[i] = action_probs[i] + noise;
                        sum = sum + action_probs[i]
                    }
                }

                if sum > f64::EPSILON {
                    action_probs.iter_mut().for_each(|x| *x = x.div(sum));
                }

                let rst = mcts.get_best_action_from_probs(&action_probs);
                mcts.update_root_with_action(rst);
                if !game_ref.borrow_mut().execute_move(rst) {
                    break;
                }
                game_status = {
                    let mut game = game_ref.borrow_mut();
                    *game.get_game_status()
                };
                game_ref.borrow().render();
                println!("");
                step = step.saturating_add(1);
            }

            let win_col_2_i8 = if game_status.1 == Color::Black {
                1
            } else if game_status.1 == Color::White {
                -1
            } else {
                0
            };
            println!(
                "Self play: total step num = {} winner = {}",
                step, win_col_2_i8
            );
            hasher.update(game_ref.borrow().get_last_move().to_ne_bytes());
            hasher.update(game_ref.borrow().get_legal_moves());
            hasher.update(rng.random_range(0..u16::MAX).to_ne_bytes());
            let hash_rst = hasher.finalize();
            let hex_string = hex::encode(hash_rst);

            let path = env::current_dir().unwrap();
            let new_path = path
                .join("data")
                .join("data_".to_string() + &save_id.to_string() + "_" + &hex_string);
            println!("Save path: {:?}", new_path);
            new_path.parent().map(fs::create_dir_all).transpose().expect("Unable to create dirs");
            let mut file = fs::File::create(new_path).expect("Unable to create file");
            _ = file.write_all(&(step as i32).to_ne_bytes());

            for i in 0..step {
                for j in 0..cfg::BOARD_SIZE {
                    for k in 0..cfg::BOARD_SIZE {
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
                let new_v = color_buffer[i as usize] as i32 * win_col_2_i8 as i32;
                v_buffer[i as usize] = new_v;
                _ = file.write_all(&v_buffer[i as usize].to_ne_bytes());
            }
            for i in 0..step {
                _ = file.write_all(&(color_buffer[i as usize] as i32).to_ne_bytes());
            }
            for i in 0..step {
                _ = file.write_all(&(last_move_buffer[i as usize] as i32).to_ne_bytes());
            }
        }
    }

    pub async fn self_play_for_train(&self, game_num: u16, start_batch_id: u16) {
        if self.neural_network.borrow().get_batch_size() < cfg::DEFAULT_BATCH_SIZE as usize {
            self.neural_network
                .borrow_mut()
                .set_batch_size(game_num as usize * cfg::DEFAULT_BATCH_SIZE as usize);
        }
        for i in 0..game_num {
            self.play(start_batch_id + i).await;
        }
    }
}
