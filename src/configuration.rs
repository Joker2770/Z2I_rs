// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

pub mod cfg {
    #[cfg(feature = "tic-tac-toe")]
    pub const BOARD_SIZE: u8 = 3;
    #[cfg(feature = "tic-tac-toe")]
    pub const N_IN_ROW: u8 = 3;
    #[cfg(feature = "tic-tac-toe")]
    pub const C_PUCT: f32 = 1.25;
    #[cfg(feature = "tic-tac-toe")]
    pub const C_VIRTUAL_LOSS: f64 = 1.0;
    #[cfg(feature = "tic-tac-toe")]
    pub const DEFAULT_SIMULATION_NUM: usize = 32;
    // temperature anchors (see cfg::temp_at): full exploration for the opening plies,
    // then an exponential decay that reaches GREEDY_TEMP by the GREEDY_FROM_STEP-th ply;
    // a 3x3 game is over after at most 9 plies
    #[cfg(feature = "tic-tac-toe")]
    pub const EXPLORE_STEP: u16 = 3;
    #[cfg(feature = "tic-tac-toe")]
    pub const GREEDY_FROM_STEP: u16 = 9;
    // Maximum noise mix; self-play tapers this by ply and normalized MCTS-policy entropy.
    #[cfg(feature = "tic-tac-toe")]
    pub const DIRI: f64 = 0.3;
    // Per-candidate Dirichlet concentration within the opening/midgame noise region.
    #[cfg(feature = "tic-tac-toe")]
    pub const DIRICHLET_ALPHA: f64 = 0.35;
    #[cfg(feature = "tic-tac-toe")]
    pub const DEFAULT_SIM_PER_BATCH_NUM: u8 = 2;
    #[cfg(feature = "tic-tac-toe")]
    pub const SIMS_BOOST_EVERY: u16 = 160;
    #[cfg(feature = "tic-tac-toe")]
    pub const SIMS_BOOST_STEP: usize = 4;
    #[cfg(feature = "tic-tac-toe")]
    pub const SIMS_CAP: usize = 32;

    #[cfg(not(feature = "tic-tac-toe"))]
    pub const BOARD_SIZE: u8 = 15;
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const N_IN_ROW: u8 = 5;
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const C_PUCT: f32 = 1.5;
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const C_VIRTUAL_LOSS: f64 = 1.0;
    // Colab T4: 2 vCPUs are the bottleneck, so the base simulation count is lowered;
    // it still grows with weight generation (SIMS_BOOST_*)
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const DEFAULT_SIMULATION_NUM: usize = 400;
    // temperature anchors (see cfg::temp_at): full exploration for the opening plies,
    // then an exponential decay that reaches GREEDY_TEMP by the GREEDY_FROM_STEP-th ply.
    // GREEDY_FROM_STEP is the measured average self-play game length of this profile, so
    // the schedule is spent by the time an average game ends and every later ply selects
    // greedily. Measured 2026-09-28 on the newest round of all five rule lineages (10 games
    // each): mean 17.7 (FreeStyle) / 20.3 (Standard) / 22.6 (Renju) / 18.4 (Caro) / 23.3
    // (Standard+Caro) plies, cross-rule mean ~20 and median 16-24. The previous 36 came
    // from the iter-1 renju log (46840 samples / 8 board symmetries / 164 games = 35.7
    // plies) and had stopped describing the games this profile actually plays: at 36 the
    // opening was still at temp 0.20 by ply 16, i.e. the deciding plies of a ~20-ply game
    // were never selected near-greedily. EXPLORE_STEP keeps the earlier ~28% ratio
    // (was 10 of 36); temp_at derives the decay length from both anchors.
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const EXPLORE_STEP: u16 = 6;
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const GREEDY_FROM_STEP: u16 = 20;
    // Maximum noise mix; self-play tapers this by ply and normalized MCTS-policy entropy.
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const DIRI: f64 = 0.25;
    // Per-candidate Dirichlet concentration within the opening/midgame noise region.
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const DIRICHLET_ALPHA: f64 = 0.3;
    // suggest < 256, and >= 1
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const DEFAULT_SIM_PER_BATCH_NUM: u8 = 16;
    // simulation count grows with weight generation (shared by self-play and
    // acceptance evaluation; equal sims on both sides for fairness):
    // sims = min(DEFAULT_SIMULATION_NUM + (weight_id / SIMS_BOOST_EVERY) * SIMS_BOOST_STEP, SIMS_CAP)
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const SIMS_BOOST_EVERY: u16 = 20;
    // Colab T4: growth step and cap lowered together so high-generation sim counts
    // stay feasible on 2 vCPUs
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const SIMS_BOOST_STEP: usize = 64;
    // 2026-09-28: `400 + (weight_id / 20) * 64` reaches the cap at weight 140, so every
    // generation of the current lineage (515+) has been paying it, and on 2 vCPUs it is the
    // dominant cost of a round: 16 games x ~20 plies x SIMS_CAP simulations. The newest
    // windows show rounds being cut short (10-13 games delivered instead of 16), so the cap
    // is traded for rounds that finish: 800 is still 2x DEFAULT_SIMULATION_NUM and keeps the
    // search targets sharp (the newest windows measure pi entropy 1.27-1.74 nats, top-1
    // 0.56-0.69), while 600 would halve the budget and risk the flat-target regime this
    // profile used to sit in (H(pi) ~ 2.6 nats with the older, softer search).
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const SIMS_CAP: usize = 800;

    pub const MAX_BOARD_SIZE: u8 = 25;
    pub const MIN_BOARD_SIZE: u8 = 3;
    pub const INPUT_CHANNEL_SIZE: u8 = 4;
    // Colab T4: 16GB VRAM is ample, so inference batches can be larger
    // (few concurrent CPU instances, limited benefit but no harm)
    pub const DEFAULT_BATCH_SIZE: u16 = 256;
    pub const MAX_BATCH_SIZE: u16 = 512;
    pub const MIN_BATCH_SIZE: u16 = 1;
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
    // 0 - free-style
    // 1 - standard
    // 4 - renju
    // 8 - caro
    // 1|8 - standard-caro
    pub const DEFAULT_RULE_FLAG: u8 = 0b_0000_0000;
    // move-selection temperature: EXPLORE_TEMP while step <= EXPLORE_STEP, then an
    // exponential decay down to GREEDY_TEMP, which mcts::policy_from_children treats as
    // one-hot greedy selection
    pub const EXPLORE_TEMP: f64 = 1.0;
    pub const GREEDY_TEMP: f64 = 1e-3;
    // minimum remaining time (ms) required before starting a full inference batch;
    // keep a margin above the measured ~1.3s batch time on the reference CPU
    pub const TIME_RESERVE_MS: u64 = 1800;
    // minimum remaining time (ms) required before starting one final simulation
    pub const SINGLE_SIM_RESERVE_MS: u64 = 800;
    // time (ms) kept for applying and reporting the selected move
    pub const FINAL_MOVE_RESERVE_MS: u64 = 500;
    pub const RENDER_AT_EVAL: bool = true;
    // whether self-play (training data generation) renders the board step by step
    // and prints temp/Step; keep false for parallel generation to avoid interleaved
    // output and needless overhead
    pub const RENDER_AT_SELF_PLAY: bool = true;
    pub const INFER_TASK_WAIT_US: u16 = 2;
    // open_mind thinking output: interval between consecutive reports (ms)
    pub const OPEN_MIND_REPORT_INTERVAL_MS: u64 = 500;
    // max children kept in open_mind thinking output (filtered by visit count)
    pub const OPEN_MIND_THINKING_MAX_CHILDREN: usize = 10;

    pub const INFER_ASYNC: bool = false;
}
