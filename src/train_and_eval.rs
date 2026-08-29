// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

#![deny(deprecated)]

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

use configuration::cfg;
use gomoku::{GameStage, Gomoku};
use mcts::MCTS;
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
        // 并行实例的 intra-op 线程数按实例数下调,避免总线程数成倍膨胀
        let intra_thread_num = ((cfg::DEFAULT_INTRA_THREAD_NUM as usize) / thread_num).max(2) as u8;

        // 模拟次数随权重代际增长
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
                    // Rc 仅在当前线程内使用;NeuralNetwork 仅含 UnboundedSender,可跨线程 move
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
        println!("eval: total step num = {}", step);

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
        return (0, 0, 0);
    }

    run_eval_games(model_a, model_b, game_num, num_mcts_sim_a, num_mcts_sim_b).await
}

/// 读取 elo.txt 中持久化的各权重 Elo 评级(weight_id -> elo)
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

/// 将 Elo 评级写回 elo.txt
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

/// 新候选权重继承父代(当前 best)权重的 Elo 评级作为起点。
/// 候选由 best 的棋谱训练而来,能力下限即 best;若每次都从 ELO_INITIAL
/// 重新起算,候选战胜 best 后评级反而可能低于 best 原有评级,导致 elo.txt
/// 长期停滞不增长。使用覆盖写入:候选 id 可能复用被拒轮次的旧 id,
/// 旧模型已被丢弃,其残留评级不应影响新模型。
fn inherit_elo(new_weight: i32, parent_weight: i32) {
    let mut ratings = load_elo();
    let parent_rating = ratings
        .get(&parent_weight)
        .copied()
        .unwrap_or(cfg::ELO_INITIAL);
    ratings.insert(new_weight, parent_rating);
    save_elo(&ratings);
}

/// 根据一场评估赛结果更新双方 Elo,返回日志描述
fn update_elo(weight_a: i32, weight_b: i32, result: (u16, u16, u16)) -> String {
    let total = result.0 + result.1 + result.2;
    if total == 0 {
        return String::new();
    }
    // a 的得分率:胜 1 分、和 DRAW_SCORE 分
    let score_a = (result.0 as f64 + cfg::DRAW_SCORE * result.2 as f64) / total as f64;

    let mut ratings = load_elo();
    let rating_a = ratings.get(&weight_a).copied().unwrap_or(cfg::ELO_INITIAL);
    let rating_b = ratings.get(&weight_b).copied().unwrap_or(cfg::ELO_INITIAL);

    // 标准 Elo:期望胜率 = 1 / (1 + 10^((Rb - Ra) / 400))
    let expected_a = 1.0 / (1.0 + 10f64.powf((rating_b - rating_a) / 400.0));
    let delta = cfg::ELO_K * (score_a - expected_a);
    let new_a = rating_a + delta;
    let new_b = rating_b - delta;
    ratings.insert(weight_a, new_a);
    ratings.insert(weight_b, new_b);
    save_elo(&ratings);

    // 由本场得分率反推的比赛表现 Elo 差(a 相对 b),截断避免 0/1 极端值
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
                // 自对弈数据应由已通过验收的最佳权重生成(AlphaZero 流程)
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
            // 候选权重继承 best 的评级起点,保证世系评级随迭代单调累积
            inherit_elo(current_weight, best_weight);
            let elo_info = update_elo(current_weight, best_weight, result);
            result_log_info.push_str(&elo_info);
            let win_ratio = (result.0 as f64 + cfg::DRAW_SCORE * result.2 as f64)
                / (result.0 + result.1 + result.2) as f64;
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
                // 候选被拒:回退到 best,下一轮从 best 重新训练(AlphaZero 流程)
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
            // 候选同样继承 best 评级,与 eval_with_winner 保持同一评级尺度;
            // 随机基线(-1)作为固定强度锚点自然漂移
            inherit_elo(current_weight_id, best_weight_id);
            let elo_info = update_elo(current_weight_id, -1, result);
            result_log_info.push_str(&elo_info);
            let win_ratio = (result.0 as f64 + cfg::DRAW_SCORE * result.2 as f64)
                / (result.0 + result.1 + result.2) as f64;
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
