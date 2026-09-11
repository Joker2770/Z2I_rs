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
    #[cfg(feature = "tic-tac-toe")]
    pub const DIRI: f64 = 0.3;
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
    pub const C_PUCT: f32 = 2.5;
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const C_VIRTUAL_LOSS: f64 = 3.0;
    // Colab T4: 2 vCPUs are the bottleneck, so the base simulation count is lowered;
    // it still grows with weight generation (SIMS_BOOST_*)
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const DEFAULT_SIMULATION_NUM: usize = 400;
    // temperature anchors (see cfg::temp_at): full exploration for the opening plies,
    // then an exponential decay that reaches GREEDY_TEMP by the GREEDY_FROM_STEP-th ply.
    // 36 plies is the measured average self-play game length of this profile (iter-1 renju
    // log: 46840 samples / 8 board symmetries / 164 games = 35.7 plies), so the schedule is
    // spent by the time an average game ends; the previous fixed decay of 12 only reached
    // the floor at ply 98 (2.7x the average game), i.e. selection was never greedy.
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const EXPLORE_STEP: u16 = 10;
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const GREEDY_FROM_STEP: u16 = 36;
    // AlphaZero exploration noise: π = (1 - DIRI)·p + DIRI·η, η ~ Dir(DIRICHLET_ALPHA)
    // DIRI is the Dirichlet noise mixing factor ε (0.25 in the AlphaZero paper)
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const DIRI: f64 = 0.25;
    // Dirichlet concentration parameter α (0.3 in the AlphaZero paper)
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
    #[cfg(not(feature = "tic-tac-toe"))]
    pub const SIMS_CAP: usize = 1200;

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
        if step >= GREEDY_FROM_STEP {
            return GREEDY_TEMP;
        }

        let decay_len =
            f64::from(GREEDY_FROM_STEP - EXPLORE_STEP) / (EXPLORE_TEMP / GREEDY_TEMP).ln();

        GREEDY_TEMP
            .max(EXPLORE_TEMP * (-f64::from(step.saturating_sub(EXPLORE_STEP)) / decay_len).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::cfg;

    #[test]
    fn test_temp_at_explores_until_warmup_step() {
        for step in 0..=cfg::EXPLORE_STEP {
            assert_eq!(cfg::temp_at(step), cfg::EXPLORE_TEMP);
        }
    }

    #[test]
    fn test_temp_at_is_spent_at_greedy_from_step() {
        // mcts::policy_from_children turns anything at (or below) GREEDY_TEMP into a one-hot
        // greedy selection, so reaching the floor at the anchor is the point of the schedule
        assert_eq!(cfg::temp_at(cfg::GREEDY_FROM_STEP), cfg::GREEDY_TEMP);

        for step in cfg::GREEDY_FROM_STEP..cfg::GREEDY_FROM_STEP + 200 {
            assert_eq!(cfg::temp_at(step), cfg::GREEDY_TEMP);
        }
    }

    #[test]
    fn test_temp_at_is_monotonically_decreasing() {
        let mut previous = cfg::temp_at(0);
        for step in 1..u16::from(cfg::GREEDY_FROM_STEP) * 4 {
            let current = cfg::temp_at(step);
            assert!(current <= previous, "temp rose at step {step}");
            assert!(current >= cfg::GREEDY_TEMP);
            previous = current;
        }
    }

    #[test]
    fn test_temp_at_interpolates_between_anchors() {
        let mid = (cfg::EXPLORE_STEP + cfg::GREEDY_FROM_STEP) / 2;
        let temp = cfg::temp_at(mid);

        assert!(temp > cfg::GREEDY_TEMP && temp < cfg::EXPLORE_TEMP);

        // the decay is exponential, so equal spacing means equal ratios: the geometric mean
        // of the two anchors has to be the value halfway between them
        let expected = (cfg::EXPLORE_TEMP * cfg::GREEDY_TEMP).sqrt();
        assert!((temp - expected).abs() < 1e-12);
    }
}
