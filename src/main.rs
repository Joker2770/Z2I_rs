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

use configuration::cfg;
use gomoku::{GameStage, Gomoku};
use mcts::MCTS;
use ortopt::NeuralNetwork;
use rule::{Color, RuleFlag};

#[derive(Debug, Deserialize)]
struct ModelConfig {
    #[serde(default = "default_model_path")]
    default_model: PathBuf,
    #[serde(default = "default_free_style_model_path")]
    free_style_model: PathBuf,
    #[serde(default = "default_renju_model_path")]
    renju_model: PathBuf,
    #[serde(default = "default_standard_model_path")]
    standard_model: PathBuf,
    #[serde(default = "default_caro_model_path")]
    caro_model: PathBuf,
    #[serde(default = "default_standard_caro_model_path")]
    standard_caro_model: PathBuf,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default_model: default_model_path(),
            free_style_model: default_free_style_model_path(),
            renju_model: default_renju_model_path(),
            standard_model: default_standard_model_path(),
            caro_model: default_caro_model_path(),
            standard_caro_model: default_standard_caro_model_path(),
        }
    }
}

fn default_model_path() -> PathBuf {
    PathBuf::from("models/default.onnx")
}
fn default_free_style_model_path() -> PathBuf {
    PathBuf::from("models/free-style.onnx")
}
fn default_renju_model_path() -> PathBuf {
    PathBuf::from("models/renju.onnx")
}
fn default_standard_model_path() -> PathBuf {
    PathBuf::from("models/standard.onnx")
}
fn default_caro_model_path() -> PathBuf {
    PathBuf::from("models/caro.onnx")
}
fn default_standard_caro_model_path() -> PathBuf {
    PathBuf::from("models/standard-caro.onnx")
}

#[derive(Debug, Deserialize)]
struct MctsConfig {
    #[serde(default = "default_num_mct_sims")]
    num_mct_sims: usize,
    #[serde(default = "default_sim_per_batch_num")]
    num_sim_per_batch: u8,
    #[serde(default = "default_open_mind")]
    open_mind: bool,
    #[serde(default = "default_enable_ponder")]
    enable_ponder: bool,
    #[serde(default = "default_time_reserve_ms")]
    time_reserve_ms: u64,
    #[serde(default = "default_single_sim_reserve_ms")]
    single_sim_reserve_ms: u64,
    #[serde(default = "default_final_move_reserve_ms")]
    final_move_reserve_ms: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_mct_sims: default_num_mct_sims(),
            num_sim_per_batch: default_sim_per_batch_num(),
            open_mind: default_open_mind(),
            enable_ponder: default_enable_ponder(),
            time_reserve_ms: default_time_reserve_ms(),
            single_sim_reserve_ms: default_single_sim_reserve_ms(),
            final_move_reserve_ms: default_final_move_reserve_ms(),
        }
    }
}

fn default_num_mct_sims() -> usize {
    cfg::DEFAULT_SIMULATION_NUM
}

fn default_sim_per_batch_num() -> u8 {
    cfg::DEFAULT_SIM_PER_BATCH_NUM
}

fn default_open_mind() -> bool {
    false
}

fn default_enable_ponder() -> bool {
    true
}

fn default_time_reserve_ms() -> u64 {
    cfg::TIME_RESERVE_MS
}

fn default_single_sim_reserve_ms() -> u64 {
    cfg::SINGLE_SIM_RESERVE_MS
}

fn default_final_move_reserve_ms() -> u64 {
    cfg::FINAL_MOVE_RESERVE_MS
}

#[derive(Debug, Deserialize)]
struct OnnxruntimeConfig {
    #[serde(default = "default_num_intra_thread")]
    num_intra_thread: u8,
}

impl Default for OnnxruntimeConfig {
    fn default() -> Self {
        Self {
            num_intra_thread: default_num_intra_thread(),
        }
    }
}

fn default_num_intra_thread() -> u8 {
    cfg::DEFAULT_INTRA_THREAD_NUM
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    #[serde(default)]
    model: ModelConfig,
    #[serde(rename = "MCTS", default)]
    mcts: MctsConfig,
    #[serde(rename = "ONNXRUNTIME", default)]
    onnx: OnnxruntimeConfig,
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
                num_sim_per_batch: cfg::DEFAULT_SIM_PER_BATCH_NUM,
                open_mind: false,
                enable_ponder: true,
                time_reserve_ms: cfg::TIME_RESERVE_MS,
                single_sim_reserve_ms: cfg::SINGLE_SIM_RESERVE_MS,
                final_move_reserve_ms: cfg::FINAL_MOVE_RESERVE_MS,
            },
            onnx: OnnxruntimeConfig {
                num_intra_thread: cfg::DEFAULT_INTRA_THREAD_NUM,
            },
        }
    }
}

/// User-level config directory:
/// - Linux follows XDG: `$XDG_CONFIG_HOME/Z2I_rs`, falling back to `~/.config/Z2I_rs` when unset;
/// - macOS: `~/Library/Application Support/Z2I_rs`;
/// - Windows: `%APPDATA%\Z2I_rs`.
fn user_config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(dir) = env::var("XDG_CONFIG_HOME") {
            let dir = PathBuf::from(dir);
            // XDG spec: relative values should be ignored
            if dir.is_absolute() && !dir.as_os_str().is_empty() {
                return Some(dir.join("Z2I_rs"));
            }
        }
        env::var("HOME")
            .ok()
            .map(|home| PathBuf::from(home).join(".config").join("Z2I_rs"))
    }
    #[cfg(target_os = "macos")]
    {
        env::var("HOME").ok().map(|home| {
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("Z2I_rs")
        })
    }
    #[cfg(target_os = "windows")]
    {
        env::var("APPDATA")
            .ok()
            .map(|dir| PathBuf::from(dir).join("Z2I_rs"))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

impl AppConfig {
    fn load() -> Self {
        // search order: user config dir > current working dir > executable dir
        let candidates = [
            user_config_dir().map(|dir| dir.join("config.toml")),
            env::current_dir().ok().map(|path| path.join("config.toml")),
            env::current_exe()
                .ok()
                .and_then(|path| path.parent().map(|parent| parent.join("config.toml"))),
        ];

        for path in candidates.into_iter().flatten() {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                match toml::from_str(&contents) {
                    Ok(config) => {
                        eprintln!("MESSAGE loaded config from {}", path.display());
                        return config;
                    }
                    Err(error) => eprintln!("MESSAGE failed to parse {}: {error}", path.display()),
                }
            }
        }
        eprintln!("MESSAGE config.toml not found; using built-in defaults");
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
    sim_per_batch_num: u8,
    intra_thread_num: u8,
    timeout_turn: Option<u64>,
    time_left: Option<i64>,
    config: AppConfig,
    neural_network: Option<Rc<RefCell<NeuralNetwork>>>,
    loaded_model_path: Option<PathBuf>,
    open_mind: bool,
    enable_ponder: bool,
    random_ponder_batches_remaining: usize,
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
            time_left: None,
            simulation_num: config.mcts.num_mct_sims,
            sim_per_batch_num: config.mcts.num_sim_per_batch,
            intra_thread_num: config.onnx.num_intra_thread,
            open_mind: config.mcts.open_mind,
            enable_ponder: config.mcts.enable_ponder,
            config,
            neural_network: None,
            loaded_model_path: None,
            random_ponder_batches_remaining: 0,
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
            eprintln!("MESSAGE model not found: {}", path.display());
            self.neural_network = None;
            self.loaded_model_path = None;
            return false;
        }
        match NeuralNetwork::new(
            &path,
            cfg::DEFAULT_BATCH_SIZE as usize,
            self.intra_thread_num,
        ) {
            Ok(network) => {
                self.neural_network = Some(Rc::new(RefCell::new(network)));
                self.loaded_model_path = Some(path);
                true
            }
            Err(error) => {
                eprintln!("MESSAGE failed to load model {}: {error}", path.display());
                self.neural_network = None;
                self.loaded_model_path = None;
                false
            }
        }
    }

    fn new_mcts(&self, action_size: u16) -> MCTS {
        MCTS::new_with_timing(
            self.neural_network.clone(),
            cfg::C_PUCT as f64,
            cfg::C_VIRTUAL_LOSS,
            AtomicUsize::new(self.simulation_num),
            self.sim_per_batch_num,
            action_size,
            self.config.mcts.time_reserve_ms,
            self.config.mcts.single_sim_reserve_ms,
            self.config.mcts.final_move_reserve_ms,
        )
    }

    /// Compute the effective thinking deadline from the per-move and match-level clocks.
    /// `time_left` is signed because the protocol permits negative values after a timeout.
    /// `None` means the corresponding INFO value was not received.
    fn think_deadline(&self) -> Option<Instant> {
        let now = Instant::now();
        let turn_deadline = self
            .timeout_turn
            .map(|ms| now + Duration::from_millis(ms));
        let match_deadline = self.time_left.map(|ms| {
            now + Duration::from_millis(ms.max(0) as u64)
        });

        match (turn_deadline, match_deadline) {
            (Some(turn), Some(match_deadline)) => Some(turn.min(match_deadline)),
            (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
            (None, None) => None,
        }
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
        self.reset_random_ponder_budget();
        true
    }

    fn reset_random_ponder_budget(&mut self) {
        let batch_size = if self.sim_per_batch_num > 0 {
            self.sim_per_batch_num
        } else {
            cfg::DEFAULT_SIM_PER_BATCH_NUM
        } as usize;
        self.random_ponder_batches_remaining = self.simulation_num.div_ceil(batch_size);
    }

    /// Handle the `INFO rule <value>` command (it may arrive before or after START):
    /// - rule unchanged: return immediately to avoid reloading the model and wasting game time;
    /// - before START (no game yet): only record the rule; the model is loaded for the new rule at START;
    /// - after START but before any move: the board applies the new rule; the model is reloaded when
    ///   its path for the new rule differs from the one already loaded, and the search tree is always
    ///   rebuilt so it never carries over simulations made under the previous rule;
    /// - after moves have been played: do not interrupt the game in progress; the new rule takes
    ///   effect for the next game only.
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
        // The model path is resolved against the newly set rule, so the loaded model
        // always matches the rule that is about to be played.
        let new_path = self.resolve_model_path();
        let model_unchanged =
            self.loaded_model_path.as_ref() == Some(&new_path) && self.neural_network.is_some();
        if !model_unchanged {
            self.load_neural_network();
        }
        // Rebuild the search tree unconditionally after a rule switch: a tree produced
        // under the previous rule must not be reused.
        self.mcts = Some(self.new_mcts(action_size));
    }

    async fn play_move(&mut self) -> Option<u16> {
        let game = self.game.as_ref()?;
        let mcts = self.mcts.as_ref()?;
        let deadline = self.think_deadline();
        if self.open_mind {
            let size = board_size(game);
            mcts.simulation_within_reporting(
                game,
                deadline,
                Duration::from_millis(cfg::OPEN_MIND_REPORT_INTERVAL_MS),
                move |mcts: &MCTS| mcts.print_thinking(size),
            )
            .await;
        } else {
            mcts.simulation_within(game, deadline).await;
        }
        let action = mcts.get_best_action_after_simulation(game);
        let game = self.game.as_mut()?;
        if !game.execute_move(action) {
            return None;
        }
        self.mcts.as_mut()?.update_root_with_action(game, action);
        Some(action)
    }

    fn play_opponent_move(&mut self, action: u16) -> bool {
        let mut is_succeed = self
            .game
            .as_mut()
            .is_some_and(|game| game.execute_move(action));
        if is_succeed
            && let Some(g) = self.game.as_ref()
            && let Some(m) = self.mcts.as_mut()
        {
            m.update_root_with_action(g, action);
            if self.neural_network.is_none() {
                self.reset_random_ponder_budget();
            }
        } else {
            is_succeed = false;
        }
        is_succeed
    }

    /// Background pondering: run one time-bounded MCTS step while waiting for the opponent.
    /// - game not started or already over: return immediately, don't burn CPU;
    /// - run a full batch only when the timeout reserve can accommodate it;
    /// - otherwise run one simulation to reduce the delay before an opponent move is handled;
    /// - without a model, random MCTS is still allowed, but only for the configured simulation
    ///   budget so the tree cannot grow forever.
    async fn ponder_batch(&mut self) {
        let Some(game) = self.game.as_mut() else {
            return;
        };
        if game.get_game_status().0 != GameStage::Running || *game.get_cur_color() == self.ai_color
        {
            return;
        }
        if self.neural_network.is_none() {
            if self.random_ponder_batches_remaining == 0 {
                return;
            }
            self.random_ponder_batches_remaining -= 1;
        }
        let (Some(game), Some(mcts)) = (self.game.as_ref(), self.mcts.as_ref()) else {
            return;
        };
        mcts
            .simulation_step_within(game, self.think_deadline())
            .await;
    }

    /// Whether pondering is possible: it is only useful during the opponent's turn.
    fn should_ponder(&mut self) -> bool {
        self.enable_ponder
            && (self.neural_network.is_some() || self.random_ponder_batches_remaining > 0)
            && self.game.as_mut().is_some_and(|game| {
                game.get_game_status().0 == GameStage::Running
                    && game.get_cur_color() != &self.ai_color
            })
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
    let mut brain = Brain::new();
    let mut board_lines: Option<Vec<(u16, u8)>> = None;

    // A dedicated OS thread keeps reading manager commands and forwards them via a channel,
    // so the main loop can keep advancing background pondering while waiting for commands.
    // Note: do not use tokio::spawn to read tokio stdin — on exit the runtime would wait for
    // the read-line task blocked on the read, so the process couldn't exit immediately after
    // END; an std thread is terminated directly when the process exits.
    let (line_tx, mut line_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            match line {
                Ok(line) => {
                    if line_tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    loop {
        // Wait for the next command line; while a game is in progress (not in BOARD collection mode),
        // use the time waiting for the opponent's move to keep running MCTS simulations in the
        // background (batch by batch). When a command arrives, finish the current batch before
        // handling it so all virtual losses are resolved at a consistent batch boundary.
        let line = if board_lines.is_none() {
            loop {
                match line_rx.try_recv() {
                    Ok(line) => break Some(line),
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        if brain.should_ponder() {
                            brain.ponder_batch().await;
                        } else {
                            break line_rx.recv().await;
                        }
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break None,
                }
            }
        } else {
            line_rx.recv().await
        };
        let Some(line) = line else { break };
        let command = line.trim();
        if command.is_empty() {
            continue;
        }

        if let Some(stones) = board_lines.as_mut() {
            if command.eq_ignore_ascii_case("END") {
                break;
            }
            if command.eq_ignore_ascii_case("DONE") {
                let pending = std::mem::take(stones);
                board_lines = None;
                if brain.load_board(&pending) {
                    if let Some(action) = brain.play_move().await {
                        let size = board_size(brain.game.as_ref().unwrap());
                        output_move(action, size);
                    } else {
                        eprintln!("ERROR cannot play board position");
                    }
                } else {
                    eprintln!("ERROR invalid board");
                }
            } else if let Some((coords, color)) = command.rsplit_once(',')
                && let (Some((x, y)), Ok(color)) =
                    (parse_coordinates(coords), color.trim().parse::<u8>())
                && let Some(game) = brain.game.as_ref()
                && let Some(action) = action_from_coordinates(board_size(game), x, y)
                && (1..=3).contains(&color)
            {
                stones.push((action, color));
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
                    eprintln!("ERROR unsupported board size");
                }
            }
            "BEGIN" => {
                if let Some(action) = brain.begin().await {
                    output_move(action, board_size(brain.game.as_ref().unwrap()));
                } else {
                    eprintln!("ERROR cannot begin");
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
                        eprintln!("ERROR no response");
                    }
                } else {
                    eprintln!("ERROR invalid turn");
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
                    Some("time_left") => {
                        if let Some(value) =
                            fields.next().and_then(|value| value.parse::<i64>().ok())
                        {
                            brain.time_left = Some(value);
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
    fn time_left_caps_the_thinking_deadline() {
        let mut brain = test_brain();
        brain.timeout_turn = Some(60_000);
        brain.time_left = Some(1);
        let deadline = brain.think_deadline().expect("deadline should be set");
        assert!(deadline <= Instant::now() + Duration::from_millis(100));
    }

    #[test]
    fn negative_time_left_sets_an_immediate_deadline() {
        let mut brain = test_brain();
        brain.time_left = Some(-1);
        let deadline = brain.think_deadline().expect("deadline should be set");
        assert!(deadline <= Instant::now() + Duration::from_millis(100));
    }

    #[test]
    fn partial_config_keeps_defaults_for_missing_sections() {
        let config: AppConfig = toml::from_str(
            r#"
            [MCTS]
            num_mct_sims = 123
            "#,
        )
        .expect("partial config should deserialize with defaults");

        assert_eq!(
            config.model.default_model,
            PathBuf::from("models/default.onnx")
        );
        assert_eq!(config.mcts.num_mct_sims, 123);
        assert_eq!(
            config.mcts.num_sim_per_batch,
            cfg::DEFAULT_SIM_PER_BATCH_NUM
        );
        assert_eq!(config.onnx.num_intra_thread, cfg::DEFAULT_INTRA_THREAD_NUM);
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
    fn pondering_only_starts_on_the_opponents_turn() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        assert_eq!(
            brain.game.as_ref().unwrap().get_cur_color(),
            &brain.ai_color
        );
        assert!(!brain.should_ponder());

        assert!(brain.play_opponent_move(0));
        assert_ne!(
            brain.game.as_ref().unwrap().get_cur_color(),
            &brain.ai_color
        );
        assert!(brain.should_ponder());
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
    async fn pondering_starts_after_our_move() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        let _ = brain.begin().await.expect("BEGIN should return a move");

        assert_eq!(
            brain.game.as_ref().unwrap().get_cur_color(),
            &opposite(brain.ai_color)
        );
        assert!(brain.should_ponder());

        let opponent_action = brain
            .game
            .as_ref()
            .unwrap()
            .get_legal_moves()
            .iter()
            .position(|&legal| legal == 1)
            .unwrap() as u16;
        assert!(brain.play_opponent_move(opponent_action));
        assert_eq!(
            brain.game.as_ref().unwrap().get_cur_color(),
            &brain.ai_color
        );
        assert!(!brain.should_ponder());
    }

    #[tokio::test]
    async fn ponder_batch_runs_simulations_before_begin() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        // simulate background pondering while waiting for the opponent's move
        brain.ponder_batch().await;

        let action = brain.begin().await.expect("BEGIN should return a move");
        assert!(action < 225);
        assert_eq!(brain.game.as_ref().unwrap().get_last_move(), action as i16);
    }

    #[tokio::test]
    async fn play_move_with_open_mind_returns_a_move() {
        let mut brain = test_brain();
        brain.open_mind = true;
        assert!(brain.start(15));

        let action = brain.play_move().await.expect("move expected");

        assert!(action < 225);
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

    // --- INFO rule 0: free-style ---

    #[test]
    fn info_rule_0_parses_to_freestyle() {
        assert_eq!(RuleFlag::from_bits_truncate(0), RuleFlag::FreeStyle);
    }

    #[test]
    fn switching_to_rule_0_selects_free_style_model() {
        let mut brain = test_brain();

        brain.apply_rule(RuleFlag::Standard);
        assert_eq!(
            brain.config.model_path(brain.rule),
            Path::new("models/standard.onnx")
        );

        brain.apply_rule(RuleFlag::from_bits_truncate(0)); // INFO rule 0
        assert_eq!(brain.rule, RuleFlag::FreeStyle);
        assert_eq!(
            brain.config.model_path(brain.rule),
            Path::new("models/free-style.onnx")
        );
    }

    #[tokio::test]
    async fn apply_rule_rebuilds_search_tree_after_ponder() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        // Simulate the engine playing White: the initial empty board is then the
        // opponent's turn, so pondering is valid before the first move.
        brain.ai_color = Color::White;
        brain.ponder_batch().await;
        assert!(!brain.mcts.as_ref().unwrap().root_is_leaf());

        brain.apply_rule(RuleFlag::Standard);

        assert!(brain.mcts.as_ref().unwrap().root_is_leaf());
    }

    // --- INFO rule 9: standard caro (0b1001 = exactly-five(1) | caro(8)) ---

    #[test]
    fn info_rule_9_parses_to_standard_caro() {
        assert_eq!(
            RuleFlag::from_bits_truncate(9),
            RuleFlag::Standard | RuleFlag::Caro
        );
    }

    #[test]
    fn info_rule_9_selects_standard_caro_model() {
        let mut brain = test_brain();

        brain.apply_rule(RuleFlag::from_bits_truncate(9)); // INFO rule 9

        assert_eq!(brain.rule, RuleFlag::Standard | RuleFlag::Caro);
        assert_eq!(
            brain.config.model_path(brain.rule),
            Path::new("models/standard-caro.onnx")
        );
    }

    #[test]
    fn info_rule_9_before_start_is_applied_on_start() {
        let mut brain = test_brain();

        brain.apply_rule(RuleFlag::from_bits_truncate(9)); // arrived before START
        assert!(brain.start(15));

        assert_eq!(
            brain.game.as_ref().unwrap().get_rule(),
            &(RuleFlag::Standard | RuleFlag::Caro)
        );
    }

    #[test]
    fn info_rule_9_after_start_reinitializes_game() {
        let mut brain = test_brain();
        assert!(brain.start(15));

        brain.apply_rule(RuleFlag::from_bits_truncate(9)); // after START, before any move

        assert_eq!(brain.rule, RuleFlag::Standard | RuleFlag::Caro);
        assert_eq!(
            brain.game.as_ref().unwrap().get_rule(),
            &(RuleFlag::Standard | RuleFlag::Caro)
        );
        assert!(brain.mcts.is_some());
    }
}
