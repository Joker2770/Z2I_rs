// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

mod caro;
mod configuration;
mod free_style;
mod gomoku;
mod mcts;
mod ortcommon;
mod ortopt;
mod play;
mod renju;
mod rule;
mod standard;

use gomoku::Gomoku;
use mcts::MCTS;
use std::{cell::RefCell, env, fs, io::Write, rc::Rc, sync::atomic::AtomicUsize};

use crate::{
    configuration::cfg, gomoku::GameStage, ortopt::NeuralNetwork, play::SelfPlay, rule::Color,
};

pub async fn generate_data_for_train(cur_weight_id: u16, start_batch_id: u16) {
    if let Ok(cur_path) = env::current_dir() {
        println!("Current folder: {:?}", cur_path);
        let model_path = cur_path
            .join("weights")
            .join(cur_weight_id.to_string() + ".onnx");

        println!("Current training model path: {:?}", model_path);

        let model = NeuralNetwork::new(&model_path, cfg::DEFAULT_BATCH_SIZE as usize);
        if let Ok(m) = model {
            let model_ref = Rc::new(RefCell::new(m));
            let sp = SelfPlay::new(model_ref);
            sp.self_play_for_train(cfg::NUM_2_SELF_PLAY, start_batch_id)
                .await;
        } else {
            eprintln!("Load model error!!!");
        }
    } else {
        eprintln!("Can not find current folder!!!");
    }
}

pub async fn play_for_eval(
    nn_a: Option<Rc<RefCell<NeuralNetwork>>>,
    nn_b: Option<Rc<RefCell<NeuralNetwork>>>,
    a_first: bool,
    do_render: bool,
    num_mcts_sim_a: u16,
    num_mcts_sim_b: u16,
) -> (u16, u16, u16) {
    let mut a_win = 0;
    let mut b_win = 0;
    let mut draw = 0;
    let mut step = 0;
    let gomoku = Gomoku::new(cfg::BOARD_SIZE, cfg::N_IN_ROW);
    if let Some(g) = gomoku {
        let g_ref = Rc::new(RefCell::new(g));
        let mut game_state = {
            let mut game = g_ref.borrow_mut();
            *game.get_game_status()
        };
        let mut ma = MCTS::new(
            nn_a,
            cfg::C_PUCT as f64,
            cfg::C_VIRTUAL_LOSS as f64,
            AtomicUsize::new(num_mcts_sim_a as usize),
            g_ref.borrow().get_action_size(),
        );
        let mut mb = MCTS::new(
            nn_b,
            cfg::C_PUCT as f64,
            cfg::C_VIRTUAL_LOSS as f64,
            AtomicUsize::new(num_mcts_sim_b as usize),
            g_ref.borrow().get_action_size(),
        );

        while game_state.0 == GameStage::Running {
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
                println!("");
            }
            game_state = {
                let mut game = g_ref.borrow_mut();
                *game.get_game_status()
            };

            step = step + 1;
        }
        println!("eval: total step num = {}", step);

        if (game_state.1 == Color::Black && a_first) || (game_state.1 == Color::White && !a_first) {
            println!("winner = a");
            a_win = a_win + 1;
        } else if (game_state.1 == Color::Black && !a_first)
            || (game_state.1 == Color::White && a_first)
        {
            println!("winner = b");
            b_win = b_win + 1;
        } else {
            draw = draw + 1
        }
    }

    (a_win, b_win, draw)
}

async fn run_eval_games(
    model_a: Option<Rc<RefCell<NeuralNetwork>>>,
    model_b: Option<Rc<RefCell<NeuralNetwork>>>,
    game_num: u16,
    num_mcts_sim_a: u16,
    num_mcts_sim_b: u16,
) -> (u16, u16, u16) {
    let mut result = (0, 0, 0);

    let a = model_a.clone();
    let b = model_b.clone();
    for game_index in 0..game_num {
        println!("Eval game {} start...", game_index + 1);
        let ma = a.clone();
        let mb = b.clone();

        let (a_win, b_win, draw) = play_for_eval(
            ma,
            mb,
            game_index % 2 == 0,
            cfg::RENDER_AT_EVAL,
            num_mcts_sim_a,
            num_mcts_sim_b,
        )
        .await;

        result.0 += a_win;
        result.1 += b_win;
        result.2 += draw;
        println!(
            "Eval game {} end. Current result: a_win={}, b_win={}, draw={}",
            game_index + 1,
            result.0,
            result.1,
            result.2
        );
    }

    result
}

pub async fn eval(
    weight_a_id: i32,
    weight_b_id: i32,
    game_num: u16,
    num_mcts_sim_a: u16,
    num_mcts_sim_b: u16,
) -> (u16, u16, u16) {
    if game_num == 0 {
        return (0, 0, 0);
    }

    let cur_path = env::current_dir().expect("Unable to get current folder");
    let load_model = |weight_id: i32| {
        if weight_id < 0 {
            None
        } else {
            let model_path = cur_path
                .join("weights")
                .join(weight_id.to_string() + ".onnx");
            match NeuralNetwork::new(&model_path, cfg::MAX_BATCH_SIZE as usize) {
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
        return (0, 0, 0);
    }

    run_eval_games(model_a, model_b, game_num, num_mcts_sim_a, num_mcts_sim_b).await
}

#[tokio::main]
async fn main() {
    const PASS_THRESHOLD: f64 = 0.6;
    let args: Vec<String> = env::args().collect();
    if args[1] == "prepare" {
        println!("Prepare for training.");
        _ = fs::create_dir("data");
        _ = fs::create_dir("weights");

        let mut f_1 =
            fs::File::create("current_and_best_weight.txt").expect("Unable to create file");
        _ = f_1.write_all(b"0 0");

        let mut f_2 = fs::File::create("random_mcts_number.txt").expect("Unable to create file");
        _ = f_2.write_all(&cfg::DEFAULT_SIMULATION_NUM.to_ne_bytes());
        println!("Next: Generate initial weight by python.");
    } else if args[1] == "generate" && args.len() == 3 {
        println!("Generate {} -th batch.", args[2]);
        let start_batch_id: u16 = args[2].parse().expect("Parameter Error!!!");

        if let Ok(content) = fs::read_to_string("current_and_best_weight.txt") {
            let mut iter = content.split_whitespace();
            if let Some(item) = iter.next() {
                let cur_weight: i32 = item.parse().unwrap();

                if cur_weight < 0 {
                    println!("LOAD error,check current_and_best_weight.txt");
                    return;
                } else {
                    println!(
                        "Generating... current_weight = {} start batch id: {}",
                        cur_weight, start_batch_id
                    );
                    generate_data_for_train(cur_weight as u16, start_batch_id).await;
                }
            } else {
                println!("current_and_best_weight.txt format error!!!");
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
            let num_mcts_sim_a: u16 = cfg::DEFAULT_SIMULATION_NUM as u16;
            let num_mcts_sim_b: u16 = cfg::DEFAULT_SIMULATION_NUM as u16;
            let result = eval(
                current_weight,
                best_weight,
                game_num,
                num_mcts_sim_a,
                num_mcts_sim_b,
            )
            .await;

            let mut result_log_info = current_weight.to_string()
                + "-th weight win: "
                + &result.0.to_string()
                + " "
                + &best_weight.to_string()
                + "-th weight win: "
                + &result.1.to_string()
                + " tie:"
                + &result.2.to_string()
                + "\n";
            let win_ratio = result.0 as f64 / (result.0 + result.1 + result.2) as f64;
            if win_ratio > PASS_THRESHOLD {
                result_log_info = result_log_info
                    + "new best weight: "
                    + &current_weight.to_string()
                    + " generated!!!\n";
                fs::write(
                    "current_and_best_weight.txt",
                    current_weight.to_string() + " " + &current_weight.to_string(),
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
        }
    } else if args[1] == "eval_with_random" && args.len() == 3 {
        let mut current_weight_id = 0;
        if let Ok(content) = fs::read_to_string("current_and_best_weight.txt") {
            let mut iter = content.split_whitespace();
            if let Some(item) = iter.next() {
                current_weight_id = item.parse().unwrap();
            }

            let mut num_random_mcts_sim = 0;
            if let Ok(content) = fs::read_to_string("random_mcts_number.txt") {
                num_random_mcts_sim = content.trim().parse().unwrap();
            }

            let game_num: u16 = args[2].parse().expect("Parameter Error!!!");
            let num_mcts_sim_a: u16 = cfg::DEFAULT_SIMULATION_NUM as u16;
            let num_mcts_sim_b: u16 = num_random_mcts_sim as u16;
            let result = eval(
                current_weight_id,
                -1,
                game_num,
                num_mcts_sim_a,
                num_mcts_sim_b,
            )
            .await;

            let mut result_log_info = current_weight_id.to_string()
                + "-th weight with mcts ["
                + &num_mcts_sim_a.to_string()
                + "] win: "
                + &result.0.to_string()
                + " Random with mcts ["
                + &num_mcts_sim_b.to_string()
                + "] win: "
                + &result.1.to_string()
                + " tie: "
                + &result.2.to_string()
                + "\n";
            let win_ratio = result.0 as f64 / (result.0 + result.1 + result.2) as f64;
            if win_ratio > PASS_THRESHOLD {
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
        }
    } else {
        println!("Hello, world!");
    }
}
