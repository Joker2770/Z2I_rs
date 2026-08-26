// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

pub mod cfg {
    pub const BOARD_SIZE: u8 = 15;
    pub const MAX_BOARD_SIZE: u8 = 25;
    pub const MIN_BOARD_SIZE: u8 = 3;
    pub const N_IN_ROW: u8 = 5;
    pub const C_PUCT: f32 = 2.5;
    pub const C_VIRTUAL_LOSS: f64 = 1.0;
    pub const CHANNEL_SIZE: u8 = 3;
    // Colab T4:16GB 显存充裕,推理批可加大(CPU 并发实例少,收益有限但无副作用)
    pub const DEFAULT_BATCH_SIZE: u16 = 256;
    // Colab T4:2 vCPU 是瓶颈,基础仿真数下调;仍随权重代际增长(SIMS_BOOST_*)
    pub const DEFAULT_SIMULATION_NUM: usize = 400;
    pub const MAX_BATCH_SIZE: u16 = 512;
    pub const MIN_BATCH_SIZE: u16 = 1;
    pub const EXPLORE_STEP: u16 = 15;
    // AlphaZero 探索噪声:π = (1 - DIRI)·p + DIRI·η,η ~ Dir(DIRICHLET_ALPHA)
    // DIRI 为 Dirichlet 噪声混合系数 ε(AlphaZero 论文取 0.25)
    pub const DIRI: f64 = 0.25;
    // Dirichlet 分布集中度参数 α(AlphaZero 论文取 0.3)
    pub const DIRICHLET_ALPHA: f64 = 0.3;
    // Colab T4 仅 2 vCPU,16 线程严重超额订阅;下调后生成阶段每实例实得 max(4/2,2)=2
    pub const DEFAULT_INTRA_THREAD_NUM: u8 = 4;
    // Colab T4:每批自对弈局数下调,保证单轮 generate 在会话限制内完成
    pub const NUM_2_SELF_PLAY: u16 = 16;
    // 并行自对弈实例数(每个实例独立加载 ONNX session 与推理线程)
    pub const NUM_2_SELF_PLAY_THREADS: u8 = 2;
    // 验收评估:候选 vs 当前最佳,和棋按 DRAW_SCORE 计分,
    // 胜率超过 UPDATE_THRESHOLD 则候选成为新 best(AlphaZero 论文为 55%)
    pub const UPDATE_THRESHOLD: f64 = 0.55;
    pub const DRAW_SCORE: f64 = 0.5;
    // Elo 评级:每场验收评估后按标准 Elo 公式更新(初始分 1500,K 因子 32)
    pub const ELO_INITIAL: f64 = 1500.0;
    pub const ELO_K: f64 = 32.0;
    // 模拟次数随权重代际增长(自对弈与验收评估共用,双方同 sims 保证公平):
    // sims = min(DEFAULT_SIMULATION_NUM + (weight_id / SIMS_BOOST_EVERY) * SIMS_BOOST_STEP, SIMS_CAP)
    pub const SIMS_BOOST_EVERY: u16 = 20;
    // Colab T4:代际增长步长与上限同步下调,避免高代际仿真量超出 2 vCPU 承受范围
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
    // 思考时间盈余阈值（毫秒）：剩余时间不足该值时停止仿真、立即落子
    pub const TIME_RESERVE_MS: u64 = 512;
    pub const RENDER_AT_EVAL: bool = true;
    // 自对弈(训练数据生成)是否逐步渲染棋盘与打印 temp/Step
    // 并行生成时建议保持 false,避免输出交错与无谓开销
    pub const RENDER_AT_SELF_PLAY: bool = true;
    pub const INFER_TASK_WAIT_US: u16 = 2;
    // suggest < 256, and >= 1
    pub const DEFAULT_SIM_PER_BATCH_NUM: u8 = 16;
    // open_mind 思考过程输出：相邻两次输出的间隔（毫秒）
    pub const OPEN_MIND_REPORT_INTERVAL_MS: u64 = 500;
    // open_mind 思考过程输出中保留的子节点数量上限（按访问次数过滤）
    pub const OPEN_MIND_THINKING_MAX_CHILDREN: usize = 10;
}
