// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

pub mod cfg {
    pub const BOARD_SIZE: u8 = 15;
    pub const MAX_BOARD_SIZE: u8 = 25;
    pub const N_IN_ROW: u8 = 5;
    pub const C_PUCT: f32 = 5.0;
    pub const C_VIRTUAL_LOSS: f64 = 3.0;
    pub const CHANNEL_SIZE: u8 = 3;
    pub const DEFAULT_BATCH_SIZE: u16 = 128;
    pub const DEFAULT_SIMULATION_NUM: usize = 1024;
    pub const MAX_BATCH_SIZE: u16 = 512;
    pub const MIN_BATCH_SIZE: u16 = 1;
    pub const OUTPUT_0_NAME: &str = "V";
    pub const OUTPUT_1_NAME: &str = "P";
    pub const EXPLORE_STEP: u16 = BOARD_SIZE as u16 * BOARD_SIZE as u16;
    pub const DIRI: f64 = 0.02;
    pub const INTRA_THREAD_NUM: u8 = 4;
    pub const NUM_2_SELF_PLAY: u16 = 10;
    // 0 - free-style
    // 1 - standard
    // 4 - renju
    // 8 - caro
    // 1|8 - standard-caro
    pub const DEFAULT_RULE_FLAG: u8 = 0b_0000_0000;
    pub const EXPLORE_TEMP: f64 = 1.0;
    pub const GREEDY_TEMP: f64 = 1e-3;
    // 思考时间盈余阈值（毫秒）：剩余时间不足该值时停止仿真、立即落子
    pub const TIME_RESERVE_MS: u64 = 512;
    pub const RENDER_AT_EVAL: bool = true;
    pub const INFER_TASK_WAIT_US: u16 = 64;
    // suggest < 256, and >= 1
    pub const SIM_PER_BATCH: u16 = 16;
}
