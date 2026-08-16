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

use serde::Deserialize;
use std::{
    cell::RefCell,
    env,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    rc::Rc,
    sync::atomic::AtomicUsize,
    time::{Duration, Instant},
};

use crate::{
    configuration::cfg,
    gomoku::Gomoku,
    mcts::MCTS,
    ortopt::NeuralNetwork,
    rule::{Color, RuleFlag},
};

#[derive(Debug, Deserialize)]
struct ModelConfig {
    default_model: PathBuf,
    free_style_model: PathBuf,
    renju_model: PathBuf,
    standard_model: PathBuf,
    caro_model: PathBuf,
    standard_caro_model: PathBuf,
}

#[derive(Debug, Deserialize)]
struct MctsConfig {
    num_mct_sims: usize,
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    model: ModelConfig,
    #[serde(rename = "MCTS")]
    mcts: MctsConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                default_model: PathBuf::from("models/default.onnx"),
                free_style_model: PathBuf::from("models/free-style.onnx"),
                renju_model: PathBuf::from("models/renju.onnx"),
                standard_model: PathBuf::from("models/standard.onnx"),
                caro_model: PathBuf::from("models/caro.onnx"),
                standard_caro_model: PathBuf::from("models/standard-caro.onnx"),
            },
            mcts: MctsConfig {
                num_mct_sims: cfg::DEFAULT_SIMULATION_NUM,
            },
        }
    }
}

impl AppConfig {
    fn load() -> Self {
        let candidates = [
            env::current_dir().ok().map(|path| path.join("config.toml")),
            env::current_exe()
                .ok()
                .and_then(|path| path.parent().map(|parent| parent.join("config.toml"))),
        ];

        for path in candidates.into_iter().flatten() {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                match toml::from_str(&contents) {
                    Ok(config) => return config,
                    Err(error) => eprintln!("failed to parse {}: {error}", path.display()),
                }
            }
        }
        eprintln!("INFO config.toml not found; using built-in defaults");
        Self::default()
    }

    fn model_path(&self, rule: RuleFlag) -> &Path {
        if rule.contains(RuleFlag::Standard) && rule.contains(RuleFlag::Caro) {
            &self.model.standard_caro_model
        } else if rule.contains(RuleFlag::Renju) {
            &self.model.renju_model
        } else if rule.contains(RuleFlag::Caro) {
            &self.model.caro_model
        } else if rule.contains(RuleFlag::Standard) {
            &self.model.standard_model
        } else if rule == RuleFlag::FreeStyle {
            &self.model.free_style_model
        } else {
            &self.model.default_model
        }
    }
}

struct Brain {
    game: Option<Gomoku>,
    mcts: Option<MCTS>,
    ai_color: Color,
    rule: RuleFlag,
    simulation_num: usize,
    timeout_turn: Option<u64>,
    config: AppConfig,
    neural_network: Option<Rc<RefCell<NeuralNetwork>>>,
    loaded_model_path: Option<PathBuf>,
}

impl Brain {
    fn new() -> Self {
        let config = AppConfig::load();
        Self {
            game: None,
            mcts: None,
            ai_color: Color::Black,
            rule: RuleFlag::FreeStyle,
            timeout_turn: None,
            simulation_num: config.mcts.num_mct_sims,
            config,
            neural_network: None,
            loaded_model_path: None,
        }
    }

    fn resolve_model_path(&self) -> PathBuf {
        let model_path = self.config.model_path(self.rule);
        if model_path.is_absolute() {
            model_path.to_path_buf()
        } else {
            let candidates = [
                env::current_dir().ok().map(|dir| dir.join(model_path)),
                env::current_exe()
                    .ok()
                    .and_then(|exe| exe.parent().map(|dir| dir.join(model_path))),
            ];
            candidates
                .into_iter()
                .flatten()
                .find(|candidate| candidate.exists())
                .unwrap_or_else(|| model_path.to_path_buf())
        }
    }

    fn load_neural_network(&mut self) -> bool {
        let path = self.resolve_model_path();
        if !path.exists() {
            eprintln!("ERROR model not found: {}", path.display());
            self.neural_network = None;
            self.loaded_model_path = None;
            return false;
        }
        match NeuralNetwork::new(&path, cfg::DEFAULT_BATCH_SIZE as usize) {
            Ok(network) => {
                self.neural_network = Some(Rc::new(RefCell::new(network)));
                self.loaded_model_path = Some(path);
                true
            }
            Err(error) => {
                eprintln!("INFO failed to load model {}: {error}", path.display());
                self.neural_network = None;
                self.loaded_model_path = None;
                false
            }
        }
    }

    fn new_mcts(&self, action_size: u16) -> MCTS {
        MCTS::new(
            self.neural_network.clone(),
            cfg::C_PUCT as f64,
            cfg::C_VIRTUAL_LOSS,
            AtomicUsize::new(self.simulation_num),
            action_size,
        )
    }

    /// 依据 `INFO timeout_turn`（毫秒，0=尽快落子）计算思考截止时间；
    /// `None` 表示未收到该指令，不限制思考时间（跑满配置的仿真数）。
    fn think_deadline(&self) -> Option<Instant> {
        self.timeout_turn
            .map(|ms| Instant::now() + Duration::from_millis(ms))
    }

    fn start(&mut self, size: u8) -> bool {
        let Some(mut game) = Gomoku::new(size, cfg::N_IN_ROW) else {
            return false;
        };
        if !game.set_rule(self.rule) {
            return false;
        }
        self.load_neural_network();
        let action_size = game.get_action_size();
        self.game = Some(game);
        self.mcts = Some(self.new_mcts(action_size));
        true
    }

    /// 处理 `INFO rule <value>` 命令（该命令可能先于或晚于 START 到达）：
    /// - 规则未变化：直接返回，避免重复加载模型、浪费对局时间；
    /// - START 之前（尚无对局）：仅记录规则，模型将在 START 时按新规则加载；
    /// - START 之后、尚未落子：棋盘应用新规则；仅当新规则对应的模型与当前
    ///   已加载的模型不同（或尚未加载）时，才重新加载模型并重建 MCTS；
    /// - 已落子之后：不打断进行中的对局，新规则仅对下一局生效。
    fn apply_rule(&mut self, rule: RuleFlag) {
        if rule == self.rule {
            return;
        }
        self.rule = rule;
        let Some(game) = self.game.as_mut() else {
            return;
        };
        if !game.set_rule(rule) {
            return;
        }
        let action_size = game.get_action_size();
        let new_path = self.resolve_model_path();
        let model_unchanged =
            self.loaded_model_path.as_ref() == Some(&new_path) && self.neural_network.is_some();
        if !model_unchanged {
            self.load_neural_network();
            self.mcts = Some(self.new_mcts(action_size));
        }
    }

    async fn play_move(&mut self) -> Option<u16> {
        let game = self.game.as_ref()?;
        let mcts = self.mcts.as_ref()?;
        mcts.simulation_within(game, self.think_deadline()).await;
        let action = mcts.get_best_action_after_simulation(game);
        let game = self.game.as_mut()?;
        if !game.execute_move(action) {
            return None;
        }
        self.mcts.as_mut()?.update_root_with_action(&game, action);
        Some(action)
    }

    fn play_opponent_move(&mut self, action: u16) -> bool {
        let is_succeed = self
            .game
            .as_mut()
            .is_some_and(|game| game.execute_move(action));
        if is_succeed {
            if let Some(g) = self.game.as_ref()
                && let Some(m) = self.mcts.as_mut()
            {
                m.update_root_with_action(&g, action);
            }
        };
        is_succeed
    }

    async fn begin(&mut self) -> Option<u16> {
        if self.game.as_ref()?.get_cur_color() != &self.ai_color {
            return None;
        }
        self.play_move().await
    }

    async fn turn(&mut self, action: u16) -> Option<u16> {
        if self
            .game
            .as_ref()
            .is_some_and(|game| game.get_last_move() < 0)
        {
            self.ai_color = Color::White;
        }
        if self.game.as_ref()?.get_cur_color() == &self.ai_color || !self.play_opponent_move(action)
        {
            return None;
        }
        self.play_move().await
    }

    fn load_board(&mut self, stones: &[(u16, u8)]) -> bool {
        let Some(game) = self.game.as_mut() else {
            return false;
        };
        let mapped: Vec<_> = stones
            .iter()
            .map(|&(action, color)| {
                (
                    action,
                    if color == 1 {
                        self.ai_color
                    } else {
                        opposite(self.ai_color)
                    },
                )
            })
            .collect();
        let own_count = mapped
            .iter()
            .filter(|(_, color)| *color == self.ai_color)
            .count();
        let opponent_count = mapped.len() - own_count;
        let next_color = if own_count <= opponent_count {
            self.ai_color
        } else {
            opposite(self.ai_color)
        };
        if !game.load_position(&mapped, next_color) {
            return false;
        }
        let action_size = game.get_action_size();
        self.mcts = Some(self.new_mcts(action_size));
        true
    }
}

fn opposite(color: Color) -> Color {
    match color {
        Color::Black => Color::White,
        Color::White => Color::Black,
        Color::Blank => Color::Blank,
    }
}

fn parse_coordinates(value: &str) -> Option<(u16, u16)> {
    let mut parts = value.split(',');
    let x = parts.next()?.trim().parse().ok()?;
    let y = parts.next()?.trim().parse().ok()?;
    if parts.next().is_some() {
        None
    } else {
        Some((x, y))
    }
}

fn action_from_coordinates(size: u8, x: u16, y: u16) -> Option<u16> {
    if x >= size as u16 || y >= size as u16 {
        None
    } else {
        Some(y * size as u16 + x)
    }
}

fn board_size(game: &Gomoku) -> u8 {
    (game.get_action_size() as f64).sqrt() as u8
}

fn output_move(action: u16, size: u8) {
    println!("{},{}", action % size as u16, action / size as u16);
}

async fn run_protocol() {
    let stdin = io::stdin();
    let mut brain = Brain::new();
    let mut board_lines: Option<Vec<(u16, u8)>> = None;

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let command = line.trim();
        if command.is_empty() {
            continue;
        }

        if let Some(stones) = board_lines.as_mut() {
            if command.eq_ignore_ascii_case("DONE") {
                let pending = std::mem::take(stones);
                board_lines = None;
                if brain.load_board(&pending) {
                    if let Some(action) = brain.play_move().await {
                        let size = board_size(brain.game.as_ref().unwrap());
                        output_move(action, size);
                    } else {
                        println!("ERROR cannot play board position");
                    }
                } else {
                    println!("ERROR invalid board");
                }
            } else if let Some((coords, color)) = command.rsplit_once(',') {
                if let (Some((x, y)), Ok(color)) =
                    (parse_coordinates(coords), color.trim().parse::<u8>())
                {
                    if let Some(game) = brain.game.as_ref() {
                        if let Some(action) = action_from_coordinates(board_size(game), x, y) {
                            if (1..=3).contains(&color) {
                                stones.push((action, color));
                            }
                        }
                    }
                }
            }
            continue;
        }

        let mut fields = command.split_whitespace();
        let keyword = fields.next().unwrap_or_default().to_ascii_uppercase();
        match keyword.as_str() {
            "START" => {
                let size = fields.next().and_then(|value| value.parse().ok());
                if size.is_some_and(|size| brain.start(size)) {
                    println!("OK");
                } else {
                    println!("ERROR unsupported board size");
                }
            }
            "BEGIN" => {
                if let Some(action) = brain.begin().await {
                    output_move(action, board_size(brain.game.as_ref().unwrap()));
                } else {
                    println!("ERROR cannot begin");
                }
            }
            "TURN" => {
                let action = fields
                    .next()
                    .and_then(parse_coordinates)
                    .and_then(|(x, y)| {
                        brain
                            .game
                            .as_ref()
                            .and_then(|game| action_from_coordinates(board_size(game), x, y))
                    });
                if let Some(action) = action {
                    if let Some(response) = brain.turn(action).await {
                        output_move(response, board_size(brain.game.as_ref().unwrap()));
                    } else {
                        println!("ERROR no response");
                    }
                } else {
                    println!("ERROR invalid turn");
                }
            }
            "BOARD" => board_lines = Some(Vec::new()),
            "INFO" => {
                let key = fields.next().map(|key| key.to_ascii_lowercase());
                match key.as_deref() {
                    Some("rule") => {
                        if let Some(value) =
                            fields.next().and_then(|value| value.parse::<u8>().ok())
                        {
                            brain.apply_rule(RuleFlag::from_bits_truncate(value));
                        }
                    }
                    Some("timeout_turn") => {
                        if let Some(value) =
                            fields.next().and_then(|value| value.parse::<u64>().ok())
                        {
                            brain.timeout_turn = Some(value);
                        }
                    }
                    _ => {}
                }
            }
            "ABOUT" => println!("name=\"Z2I_rs\", version=\"0.1.0\", author=\"Joker2770\""),
            "END" => break,
            _ => println!("UNKNOWN"),
        }
        let _ = io::stdout().flush();
    }
}

#[tokio::main]
async fn main() {
    run_protocol().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_brain() -> Brain {
        Brain {
            simulation_num: 1,
            ..Brain::new()
        }
    }

    #[test]
    fn mandatory_coordinate_parsing() {
        assert_eq!(parse_coordinates("10, 11"), Some((10, 11)));
        assert_eq!(parse_coordinates("10,11,1"), None);
        assert_eq!(action_from_coordinates(15, 10, 11), Some(175));
        assert_eq!(action_from_coordinates(15, 15, 11), None);
    }

    #[test]
    fn timeout_turn_is_unlimited_by_default() {
        let brain = test_brain();
        assert!(brain.think_deadline().is_none());
    }

    #[test]
    fn zero_timeout_turn_sets_immediate_deadline() {
        let mut brain = test_brain();
        brain.timeout_turn = Some(0);
        let deadline = brain.think_deadline().expect("deadline should be set");
        assert!(deadline <= Instant::now() + Duration::from_millis(100));
    }

    #[test]
    fn positive_timeout_turn_sets_future_deadline() {
        let mut brain = test_brain();
        brain.timeout_turn = Some(60_000);
        assert!(brain.think_deadline().expect("deadline should be set") > Instant::now());
    }

    #[test]
    fn start_initializes_board_and_rule() {
        let mut brain = test_brain();
        brain.rule = RuleFlag::Standard;

        assert!(brain.start(15));
        assert_eq!(brain.game.as_ref().unwrap().get_action_size(), 225);
        assert_eq!(brain.game.as_ref().unwrap().get_rule(), &RuleFlag::Standard);
        assert_eq!(
            brain
                .game
                .as_ref()
                .unwrap()
                .get_legal_moves()
                .iter()
                .sum::<u8>(),
            225
        );
    }

    #[test]
    fn info_rule_before_start_is_applied_on_start() {
        let mut brain = test_brain();

        brain.apply_rule(RuleFlag::Standard);
        assert!(brain.start(15));

        assert_eq!(brain.game.as_ref().unwrap().get_rule(), &RuleFlag::Standard);
        assert!(brain.mcts.is_some());
    }

    #[test]
    fn repeated_same_rule_after_start_is_noop() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        brain.apply_rule(RuleFlag::FreeStyle);

        assert_eq!(
            brain.game.as_ref().unwrap().get_rule(),
            &RuleFlag::FreeStyle
        );
        assert_eq!(brain.game.as_ref().unwrap().get_last_move(), -1);
        assert!(brain.mcts.is_some());
    }

    #[test]
    fn apply_rule_after_start_reinitializes_game_and_mcts() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        brain.apply_rule(RuleFlag::Renju);

        assert_eq!(brain.rule, RuleFlag::Renju);
        assert_eq!(brain.game.as_ref().unwrap().get_rule(), &RuleFlag::Renju);
        assert!(brain.mcts.is_some());
    }

    #[tokio::test]
    async fn apply_rule_before_first_move_keeps_game_playable() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        brain.apply_rule(RuleFlag::Standard);
        let action = brain.begin().await.expect("BEGIN should return a move");

        assert!(action < 225);
        assert_eq!(brain.game.as_ref().unwrap().get_rule(), &RuleFlag::Standard);
        assert_eq!(brain.game.as_ref().unwrap().get_cur_color(), &Color::White);
    }

    #[tokio::test]
    async fn apply_rule_after_move_does_not_touch_game() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        let action = brain.begin().await.expect("BEGIN should return a move");
        brain.apply_rule(RuleFlag::Renju);

        assert_eq!(brain.rule, RuleFlag::Renju);
        assert_eq!(
            brain.game.as_ref().unwrap().get_rule(),
            &RuleFlag::FreeStyle
        );
        assert_eq!(brain.game.as_ref().unwrap().get_last_move(), action as i16);
    }

    #[tokio::test]
    async fn begin_returns_move_and_updates_turn() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        let action = brain.begin().await.expect("BEGIN should return a move");

        assert!(action < 225);
        assert_eq!(brain.game.as_ref().unwrap().get_cur_color(), &Color::White);
        assert_eq!(brain.game.as_ref().unwrap().get_last_move(), action as i16);
    }

    #[tokio::test]
    async fn turn_applies_opponent_move_then_returns_ai_move() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        let response = brain.turn(100).await.expect("TURN should return a move");

        assert_ne!(response, 100);
        assert_eq!(brain.game.as_ref().unwrap().get_cur_color(), &Color::Black);
        assert_eq!(
            brain.game.as_ref().unwrap().get_board()[6][10],
            Color::Black
        );
    }

    #[test]
    fn board_rebuild_maps_own_and_opponent_stones() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        assert!(brain.load_board(&[(112, 1), (113, 2)]));

        let game = brain.game.as_ref().unwrap();
        assert_eq!(game.get_board()[7][7], Color::Black);
        assert_eq!(game.get_board()[7][8], Color::White);
        assert_eq!(game.get_cur_color(), &Color::Black);
        assert_eq!(game.get_legal_moves()[112], 0);
        assert_eq!(game.get_legal_moves()[113], 0);
    }

    #[test]
    fn board_rebuild_rejects_duplicate_positions() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        assert!(!brain.load_board(&[(112, 1), (112, 2)]));
    }
}
