// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

pub mod cfg {
    pub const BOARD_SIZE: u8 = 15;
    pub const MAX_BOARD_SIZE: u8 = 25;
    pub const MIN_BOARD_SIZE: u8 = 3;
    pub const N_IN_ROW: u8 = 5;
    pub const C_PUCT: f32 = 2.5;
    pub const C_VIRTUAL_LOSS: f64 = 3.0;
    pub const CHANNEL_SIZE: u8 = 3;
    // Colab T4: 16GB VRAM is ample, so inference batches can be larger
    // (few concurrent CPU instances, limited benefit but no harm)
    pub const DEFAULT_BATCH_SIZE: u16 = 256;
    // Colab T4: 2 vCPUs are the bottleneck, so the base simulation count is lowered;
    // it still grows with weight generation (SIMS_BOOST_*)
    pub const DEFAULT_SIMULATION_NUM: usize = 400;
    pub const MAX_BATCH_SIZE: u16 = 512;
    pub const MIN_BATCH_SIZE: u16 = 1;
    pub const EXPLORE_STEP: u16 = 15;
    // AlphaZero exploration noise: π = (1 - DIRI)·p + DIRI·η, η ~ Dir(DIRICHLET_ALPHA)
    // DIRI is the Dirichlet noise mixing factor ε (0.25 in the AlphaZero paper)
    pub const DIRI: f64 = 0.25;
    // Dirichlet concentration parameter α (0.3 in the AlphaZero paper)
    pub const DIRICHLET_ALPHA: f64 = 0.3;
    // Colab T4 has only 2 vCPUs, so 16 threads would be heavily oversubscribed;
    // after lowering, each instance effectively gets max(4/2,2)=2 during generation
    pub const DEFAULT_INTRA_THREAD_NUM: u8 = 4;
    // Colab T4: self-play games per batch lowered so one generate round finishes
    // within the session limit
    pub const NUM_2_SELF_PLAY: u16 = 16;
    // parallel self-play instances (each loads its own ONNX session and inference thread)
    pub const NUM_2_SELF_PLAY_THREADS: u8 = 2;
    // acceptance evaluation: candidate vs current best, draws scored as DRAW_SCORE,
    // the candidate becomes the new best if its win rate exceeds UPDATE_THRESHOLD
    // (55% in the AlphaZero paper)
    pub const UPDATE_THRESHOLD: f64 = 0.55;
    pub const DRAW_SCORE: f64 = 0.5;
    // Elo rating: updated with the standard Elo formula after each acceptance
    // evaluation (initial 1500, K factor 32)
    pub const ELO_INITIAL: f64 = 1500.0;
    pub const ELO_K: f64 = 32.0;
    // simulation count grows with weight generation (shared by self-play and
    // acceptance evaluation; equal sims on both sides for fairness):
    // sims = min(DEFAULT_SIMULATION_NUM + (weight_id / SIMS_BOOST_EVERY) * SIMS_BOOST_STEP, SIMS_CAP)
    pub const SIMS_BOOST_EVERY: u16 = 20;
    // Colab T4: growth step and cap lowered together so high-generation sim counts
    // stay feasible on 2 vCPUs
    pub const SIMS_BOOST_STEP: usize = 64;
    pub const SIMS_CAP: usize = 1200;
    // 0 - free-style
    // 1 - standard
    // 4 - renju
    // 8 - caro
    // 1|8 - standard-caro
    pub const DEFAULT_RULE_FLAG: u8 = 0b_0000_0000;
    pub const EXPLORE_TEMP: f64 = 1.0;
    pub const GREEDY_TEMP: f64 = 1e-3;
    pub const TEMP_DECAY: u8 = 12;
    // thinking-time reserve threshold (ms): stop simulations and move immediately
    // when the remaining time drops below this value
    pub const TIME_RESERVE_MS: u64 = 512;
    pub const RENDER_AT_EVAL: bool = true;
    // whether self-play (training data generation) renders the board step by step
    // and prints temp/Step; keep false for parallel generation to avoid interleaved
    // output and needless overhead
    pub const RENDER_AT_SELF_PLAY: bool = true;
    pub const INFER_TASK_WAIT_US: u16 = 2;
    // suggest < 256, and >= 1
    pub const DEFAULT_SIM_PER_BATCH_NUM: u8 = 16;
    // open_mind thinking output: interval between consecutive reports (ms)
    pub const OPEN_MIND_REPORT_INTERVAL_MS: u64 = 500;
    // max children kept in open_mind thinking output (filtered by visit count)
    pub const OPEN_MIND_THINKING_MAX_CHILDREN: usize = 10;

    pub const INFER_ASYNC: bool = true;
}
