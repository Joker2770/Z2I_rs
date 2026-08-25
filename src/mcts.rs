// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use futures::future;
use std::cell::RefCell;
use std::ops::Div;
use std::rc::{Rc, Weak};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::task;

use crate::configuration::cfg;
use crate::gomoku::{GameStage, Gomoku};
use crate::ortopt::NeuralNetwork;
use crate::rule::Color;

#[derive(Debug)]
pub struct MCTSNode {
    parent: RefCell<Weak<MCTSNode>>,
    children: RefCell<Vec<Rc<MCTSNode>>>,
    visits: RefCell<AtomicUsize>,
    prior_probs: f64,
    total_value: RefCell<f64>,
    virtual_loss: RefCell<AtomicU16>,
    action: u16,
}

impl MCTSNode {
    pub fn new() -> Self {
        MCTSNode {
            parent: RefCell::new(Weak::new()),
            children: RefCell::new(Vec::new()),
            visits: RefCell::new(AtomicUsize::new(0)),
            prior_probs: 0.0,
            total_value: RefCell::new(0.0),
            virtual_loss: RefCell::new(AtomicU16::new(0)),
            action: u16::MAX,
        }
    }

    pub fn parent(&self) -> Option<Rc<Self>> {
        self.parent.borrow().upgrade()
    }

    pub fn is_leaf(&self) -> bool {
        self.children.borrow().is_empty()
    }

    pub fn get_puct_value(
        &self,
        c_puct: f64,
        c_virtual_loss: f64,
        sum_visits_from_parents: usize,
    ) -> f64 {
        let virtual_loss =
            c_virtual_loss * self.virtual_loss.borrow().load(Ordering::SeqCst) as f64;
        let v = self.visits.borrow().load(Ordering::SeqCst);
        let get_q = |v: usize, v_l: f64| -> f64 {
            if v == 0 {
                0.0
            } else {
                let vf = v as f64;
                let w = *self.total_value.borrow();
                (w - v_l) / vf
            }
        };
        let get_u = |v: usize, c_puct: f64| -> f64 {
            let vf = v as f64;
            c_puct * self.prior_probs * (sum_visits_from_parents as f64).sqrt() / (1.0 + vf)
        };

        let q = get_q(v, virtual_loss);
        let u = get_u(v, c_puct);

        q + u
    }

    pub fn select(&self, c_puct: f64, c_virtual_loss: f64) -> Option<Rc<Self>> {
        let mut best_value = f64::MIN;
        let mut best_child = None;
        let sum_visits_from_parent = self.visits.borrow().load(Ordering::SeqCst);
        for c in self.children.borrow().iter() {
            let value = c.get_puct_value(c_puct, c_virtual_loss, sum_visits_from_parent);
            if value > best_value {
                // println!(
                //     "v: {}, W: {}, v: {}, v_l: {}",
                //     value,
                //     c.total_value.borrow(),
                //     c.visits.borrow().load(Ordering::SeqCst),
                //     c.virtual_loss.borrow().load(Ordering::SeqCst)
                // );
                best_value = value;
                best_child = Some(Rc::clone(c));
            }
        }
        // println!();
        if let Some(b_c) = best_child {
            let old_vl = b_c.virtual_loss.borrow().load(Ordering::SeqCst);
            let new_vl = old_vl.saturating_add(1);
            b_c.virtual_loss
                .borrow_mut()
                .store(new_vl, Ordering::SeqCst);
            best_child = Some(Rc::clone(&b_c));
        }

        best_child
    }

    pub fn expand(self: &Rc<Self>, action: u16, prior: f64) -> bool {
        for c in self.children.borrow().iter() {
            if action == c.action {
                return false;
            }
        }

        if prior > f64::EPSILON {
            let mut new_child = MCTSNode::new();
            new_child.action = action;
            new_child.prior_probs = prior;
            *new_child.parent.borrow_mut() = Rc::downgrade(self);

            self.children.borrow_mut().push(Rc::new(new_child));
            true
        } else {
            false
        }
    }

    pub fn backpropagate(&self, value: f64) {
        if let Some(p) = self.parent() {
            p.backpropagate(-value);
        }

        let old_vl = self.virtual_loss.borrow().load(Ordering::SeqCst);
        let new_vl = old_vl.saturating_sub(1);
        self.virtual_loss
            .borrow_mut()
            .store(new_vl, Ordering::SeqCst);

        let new_visits = self
            .visits
            .borrow()
            .load(Ordering::SeqCst)
            .saturating_add(1);
        self.visits.borrow_mut().store(new_visits, Ordering::SeqCst);
        *self.total_value.borrow_mut() += value;
    }
}

pub struct MCTS {
    root: RefCell<Rc<MCTSNode>>,
    neural_network: Option<Rc<RefCell<NeuralNetwork>>>,
    simulation_num: AtomicUsize,
    action_size: u16,
    c_puct: f64,
    c_virtual_loss: f64,
    sims_per_batch: u8,
    /// 串行化对搜索树的读写（select 下降 / expand / backpropagate），
    /// 避免多个并发仿真任务同时抢占 select。锁内不能包含 `.await`。
    tree_lock: Mutex<()>,
}

impl MCTS {
    pub fn new(
        neural_network: Option<Rc<RefCell<NeuralNetwork>>>,
        c_puct: f64,
        c_virtual_loss: f64,
        simulation_num: AtomicUsize,
        sim_per_batch_num: u8,
        action_size: u16,
    ) -> Self {
        let sims_per_batch = if sim_per_batch_num > 0 {
            sim_per_batch_num
        } else {
            cfg::DEFAULT_SIM_PER_BATCH_NUM
        };

        MCTS {
            root: RefCell::new(Rc::new(MCTSNode::new())),
            neural_network,
            simulation_num: if simulation_num.load(Ordering::Relaxed) > 0 {
                simulation_num
            } else {
                AtomicUsize::new(cfg::DEFAULT_SIMULATION_NUM)
            },
            action_size,
            c_puct,
            c_virtual_loss,
            sims_per_batch,
            tree_lock: Mutex::new(()),
        }
    }

    pub fn set_simulation_num(&mut self, sims_num: usize) -> bool {
        if sims_num > 0 {
            self.simulation_num.store(sims_num, Ordering::Release);
            true
        } else {
            false
        }
    }

    pub fn update_root_with_action(&mut self, gomoku: &Gomoku, select_action: u16) -> bool {
        let new_root = {
            let root = Rc::clone(&self.root.borrow());
            let mut select_child = None;
            for c in root.children.borrow().iter() {
                if c.action == select_action {
                    select_child = Some(Rc::clone(c));
                }
            }
            select_child
        };

        let mut is_succeed = false;

        if let Some(node) = new_root {
            *node.parent.borrow_mut() = Weak::new();
            *self.root.borrow_mut() = Rc::clone(&node);
            is_succeed = true
        } else {
            let legal_moves_hash_tab = gomoku.get_legal_moves();
            for (action, legal) in legal_moves_hash_tab.iter().enumerate() {
                if action == select_action as usize && *legal == 1 {
                    let mut new_child = MCTSNode::new();
                    new_child.action = action as u16;
                    *new_child.parent.borrow_mut() = Weak::new();

                    *self.root.borrow_mut() = Rc::new(new_child);
                    return true;
                }
            }
        }
        is_succeed
    }

    pub async fn get_action_probs(&self, gomoku: &Gomoku, temp: f64) -> Vec<f64> {
        self.get_action_probs_within(gomoku, temp, None).await
    }

    /// 在截止时间前尽可能多地执行仿真批次：
    /// - `deadline` 为 `None`：跑满配置的仿真次数；
    /// - 否则至少执行一批（保证根节点有子节点可选出着法），
    ///   之后每批开始前若时间盈余不足 `TIME_RESERVE_MS` 毫秒即停止仿真。
    pub async fn get_action_probs_within(
        &self,
        gomoku: &Gomoku,
        temp: f64,
        deadline: Option<Instant>,
    ) -> Vec<f64> {
        // for _ in 0..self.simulation_num {
        //     _ = self.simulation(gomoku).await;
        // }
        self.simulation_within(gomoku, deadline).await;

        let priors_size = gomoku.get_action_size() as usize;
        let mut action_probs = vec![0.0; priors_size];
        let root = self.root.borrow();
        let children = root.children.borrow();

        // greedy
        if (temp - cfg::GREEDY_TEMP).abs() < f64::EPSILON {
            let mut best_action = u16::MAX;
            let mut most_visits = 0;
            for c in children.iter() {
                let c_v = c.visits.borrow().load(Ordering::SeqCst);
                if c_v > most_visits
                    || (c_v == most_visits
                        && c.prior_probs
                            > children
                                .iter()
                                .find(|candidate| candidate.action == best_action)
                                .map_or(-1.0, |candidate| candidate.prior_probs))
                {
                    most_visits = c_v;
                    best_action = c.action;
                }
            }

            if !children.is_empty() {
                action_probs[best_action as usize] = 1.0;
            }
        }
        // explore
        else {
            // 将访问次数转化为策略概率：π(a) ∝ N(a)^(1/τ)。
            // 为避免数值溢出，先在对数域计算 log π(a) = (1/τ) * ln(N(a))，
            // 再减去最大对数值（max_log_prob）后取 exp，即 softmax 的数值稳定写法。
            let inv_temp = (1.0).div(temp);
            let mut log_probs = vec![f64::NEG_INFINITY; priors_size];
            let mut max_log_prob = f64::NEG_INFINITY;
            for c in children.iter() {
                let c_v = c.visits.borrow().load(Ordering::SeqCst);
                if c_v > 0 {
                    let log_prob = inv_temp * (c_v as f64).ln();
                    log_probs[c.action as usize] = log_prob;
                    max_log_prob = max_log_prob.max(log_prob);
                }
            }

            let mut sum = 0.0;
            for (idx, log_prob) in log_probs.iter().enumerate() {
                if log_prob.is_finite() {
                    action_probs[idx] = (log_prob - max_log_prob).exp();
                    sum += action_probs[idx];
                }
            }
            // println!("Sum of action_probs before normalization: {}", sum);
            if sum > f64::EPSILON {
                action_probs.iter_mut().for_each(|x| *x = x.div(sum));
            } else {
                for c in children.iter() {
                    action_probs[c.action as usize] = c.prior_probs;
                }
                let prior_sum: f64 = action_probs.iter().sum();
                if prior_sum > f64::EPSILON {
                    action_probs.iter_mut().for_each(|x| *x = x.div(prior_sum));
                }
            }
        }
        action_probs
    }

    pub fn get_best_action_from_probs(&self, probs: &[f64]) -> u16 {
        let mut best_action = u16::MAX;
        let mut best_probs = -1.0;

        for (i, item) in probs.iter().enumerate() {
            if *item > best_probs {
                best_probs = *item;
                best_action = i as u16;
            }
        }

        best_action
    }

    pub async fn get_best_action(&self, gomoku: &Gomoku) -> u16 {
        self.get_best_action_within(gomoku, None).await
    }

    pub async fn get_best_action_within(&self, gomoku: &Gomoku, deadline: Option<Instant>) -> u16 {
        // for _ in 0..self.simulation_num {
        //     self.simulation(gomoku).await;
        // }
        self.simulation_within(gomoku, deadline).await;

        let root = self.root.borrow();
        let children = root.children.borrow();

        if children.is_empty() {
            for (action, legal) in gomoku.get_legal_moves().iter().enumerate() {
                if *legal == 1 {
                    return action as u16;
                }
            }
            return u16::MAX;
        }

        let mut best_action = u16::MAX;
        let mut most_visits = 0usize;
        for c in children.iter() {
            let c_v = c.visits.borrow().load(Ordering::SeqCst);
            if c_v > most_visits
                || (c_v == most_visits
                    && c.prior_probs
                        > children
                            .iter()
                            .find(|candidate| candidate.action == best_action)
                            .map_or(-1.0, |candidate| candidate.prior_probs))
            {
                most_visits = c_v;
                best_action = c.action;
            }
        }
        best_action
    }

    /// 执行一批并发仿真（`sims_per_batch` 个）。
    /// 批次边界是搜索树状态的一致点：批内仿真引发的 virtual_loss 均已回退，
    /// 因此适合在批次边界让出控制权（例如把对手的思考时间用于后台思考）。
    pub async fn simulation_batch(&self, gomoku: &Gomoku) {
        let simulations = (0..self.sims_per_batch).map(|_| self.simulation(gomoku));
        future::join_all(simulations).await;
    }

    pub async fn simulation_within(&self, gomoku: &Gomoku, deadline: Option<Instant>) {
        let sim_batch = self
            .simulation_num
            .load(Ordering::Relaxed)
            .div(self.sims_per_batch as usize);
        for batches_done in 0..sim_batch {
            if batches_done > 0
                && let Some(deadline) = deadline
                && Instant::now() + Duration::from_millis(cfg::TIME_RESERVE_MS) >= deadline
            {
                break;
            }
            self.simulation_batch(gomoku).await;
        }
    }

    /// 在 `simulation_within` 的基础上，每隔 `report_interval` 调用一次 `report`，
    /// 用于把根节点的搜索进展定期输出给 manager（open_mind 调试输出）。
    pub async fn simulation_within_reporting(
        &self,
        gomoku: &Gomoku,
        deadline: Option<Instant>,
        report_interval: Duration,
        mut report: impl FnMut(&Self),
    ) {
        let sim_batch = self
            .simulation_num
            .load(Ordering::Relaxed)
            .div(self.sims_per_batch as usize);
        let mut next_report = Instant::now() + report_interval;
        for batches_done in 0..sim_batch {
            if batches_done > 0
                && let Some(deadline) = deadline
                && Instant::now() + Duration::from_millis(cfg::TIME_RESERVE_MS) >= deadline
            {
                break;
            }
            self.simulation_batch(gomoku).await;
            if Instant::now() >= next_report {
                report(self);
                next_report = Instant::now() + report_interval;
            }
        }
    }

    pub fn get_best_action_after_simulation(&self, gomoku: &Gomoku) -> u16 {
        let root = self.root.borrow();
        let children = root.children.borrow();

        if children.is_empty() {
            for (action, legal) in gomoku.get_legal_moves().iter().enumerate() {
                if *legal == 1 {
                    return action as u16;
                }
            }
            return u16::MAX;
        }

        let mut best_action = u16::MAX;
        let mut most_visits = 0usize;
        for c in children.iter() {
            let c_v = c.visits.borrow().load(Ordering::SeqCst);
            if c_v >= most_visits {
                most_visits = c_v;
                best_action = c.action;
            }
        }
        best_action
    }

    /// 输出根节点第一层子节点的动作坐标与访问次数：
    /// `DEBUG thinking x1,y1,visits1 x2,y2,visits2 ...`
    /// 子节点过多时，过滤访问次数较少的子节点。
    pub fn print_thinking(&self, board_size: u8) {
        let root = self.root.borrow();
        let children = root.children.borrow();
        if children.is_empty() {
            return;
        }

        let mut entries: Vec<(u16, usize)> = children
            .iter()
            .map(|child| (child.action, child.visits.borrow().load(Ordering::SeqCst)))
            .filter(|(_, visits)| *visits > 0)
            .collect();
        if entries.is_empty() {
            return;
        }

        // 子节点过多时，只保留访问次数最多的若干个
        if entries.len() > cfg::OPEN_MIND_THINKING_MAX_CHILDREN {
            entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            entries.truncate(cfg::OPEN_MIND_THINKING_MAX_CHILDREN);
        }

        let size = board_size as u16;
        let message = entries
            .iter()
            .map(|(action, visits)| format!("{},{},{}", action % size, action / size, visits))
            .collect::<Vec<_>>()
            .join(" ");
        println!("DEBUG thinking {message}");
    }

    pub fn get_action_by_sample(&self, probs: &[f64]) -> u16 {
        let total: f64 = probs.iter().take(self.action_size as usize).sum();
        if total <= f64::EPSILON {
            // 概率全为 0(不应发生):退化为取第一个非零概率的动作
            return probs
                .iter()
                .take(self.action_size as usize)
                .position(|p| *p > 0.0)
                .map(|i| i as u16)
                .unwrap_or(0);
        }
        // 按总和归一后采样,避免浮点误差导致累积和永不越过 r
        let r: f64 = rand::random();
        let mut accum = 0.0;
        for (i, p) in probs.iter().take(self.action_size as usize).enumerate() {
            accum += p / total;
            if accum > r {
                return i as u16;
            }
        }
        // 浮点误差边界:r 极接近 1 时返回最后一个非零概率的动作
        probs
            .iter()
            .take(self.action_size as usize)
            .enumerate()
            .rev()
            .find(|(_, p)| **p > 0.0)
            .map(|(i, _)| i as u16)
            .unwrap_or(0)
    }

    pub async fn simulation(&self, gomoku: &Gomoku) {
        // 阶段一：持锁执行 select 下降。锁把多个并发仿真任务的树内选择串行化，
        // 避免它们同时抢占 select、选到同一分支或并发修改 virtual_loss。
        let (node, mut g) = {
            let _select_guard = self.tree_lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut node = Rc::clone(&self.root.borrow());
            let mut g = gomoku.clone();

            loop {
                if node.is_leaf() {
                    break;
                }

                if let Some(c) = node.select(self.c_puct, self.c_virtual_loss) {
                    if g.execute_move(c.action) {
                        node = Rc::clone(&c);
                    } else {
                        eprintln!("Illegal move!!!");
                        drop(c);
                    }
                } else {
                    break;
                }
            }

            (node, g)
        }; // 推理前释放锁，锁内不包含 .await

        let (game_stage, color) = {
            let status = g.get_game_status();
            *status
        };
        let mut value = 0.0;

        if game_stage == GameStage::Running {
            let mut action_priors = vec![0.0; self.action_size as usize];
            let legal_hash_tab = g.get_legal_moves();

            let mut sum = 0.0;
            if let Some(nn) = &self.neural_network {
                let rx = nn
                    .borrow()
                    .commit(&g)
                    .expect("inference worker is unavailable");
                let result = task::spawn_blocking(move || rx.recv().unwrap())
                    .await
                    .unwrap()
                    .expect("inference failed");
                action_priors = result.0;
                value = result.1;
                for i in 0..self.action_size {
                    if 1 == legal_hash_tab[i as usize] {
                        sum += action_priors[i as usize];
                    } else {
                        action_priors[i as usize] = 0.0;
                    }
                }

                if sum > f64::EPSILON {
                    action_priors.iter_mut().for_each(|x| *x = x.div(sum));
                } else {
                    sum = legal_hash_tab.iter().map(|&x| x as f64).sum();
                    if sum > f64::EPSILON {
                        for i in 0..action_priors.len() {
                            action_priors[i] = legal_hash_tab[i] as f64 / sum;
                        }
                    }
                }
            } else {
                sum = legal_hash_tab.iter().map(|&x| x as f64).sum();
                if sum > f64::EPSILON {
                    for i in 0..action_priors.len() {
                        action_priors[i] = legal_hash_tab[i] as f64 / sum;
                    }
                }
            }

            // 阶段二：持锁完成 expand 与 backpropagate，保证树变更的原子性
            let _tree_guard = self.tree_lock.lock().unwrap_or_else(|e| e.into_inner());
            for (i, v) in legal_hash_tab.iter().enumerate() {
                if *v == 1 {
                    node.expand(i as u16, action_priors[i]);
                }
            }
            node.backpropagate(value);
        } else {
            value = if color == Color::Blank {
                0.0
            } else if color == *g.get_cur_color() {
                1.0
            } else {
                -1.0
            };
            let _tree_guard = self.tree_lock.lock().unwrap_or_else(|e| e.into_inner());
            node.backpropagate(value);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn get_best_action_on_first_move_returns_a_legal_action() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let mcts = MCTS::new(
            None,
            1.0,
            3.0,
            AtomicUsize::new(1),
            1,
            game.get_action_size(),
        );

        let action = mcts.get_best_action(&game).await;

        assert_ne!(action, u16::MAX, "first move should not be invalid");
        assert!(action < game.get_action_size());
        assert_eq!(
            game.get_legal_moves()[action as usize],
            1,
            "first move must come from the legal move set"
        );
    }

    #[tokio::test]
    async fn simulation_expands_and_backpropagates() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let mcts = MCTS::new(
            None,
            1.0,
            3.0,
            AtomicUsize::new(1),
            1,
            game.get_action_size(),
        );

        mcts.simulation(&game).await;

        let root = mcts.root.borrow();
        assert_eq!(root.visits.borrow().load(Ordering::SeqCst), 1);
        assert_eq!(
            root.children.borrow().len(),
            game.get_action_size() as usize
        );
        assert!(
            root.children
                .borrow()
                .iter()
                .all(|child| child.prior_probs > 0.0)
        );

        for _ in 0..3 {
            mcts.simulation(&game).await;
        }

        assert_eq!(root.visits.borrow().load(Ordering::SeqCst), 4);
        let child_0 = root.children.borrow()[0].clone();
        let selected_child_visits = child_0.visits.borrow().load(Ordering::SeqCst);
        assert_eq!(selected_child_visits, 1);
    }

    #[tokio::test]
    async fn get_action_probs_returns_normalized_legal_probs() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let mcts = MCTS::new(
            None,
            1.0,
            3.0,
            AtomicUsize::new(2),
            1,
            game.get_action_size(),
        );

        let probs = mcts.get_action_probs(&game, 1.0).await;

        assert_eq!(probs.len(), game.get_action_size() as usize);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            probs
                .iter()
                .enumerate()
                .all(|(action, probability)| game.get_legal_moves()[action] == 1
                    || *probability == 0.0)
        );
    }

    #[tokio::test]
    async fn get_action_probs_falls_back_to_priors_after_one_simulation() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let mcts = MCTS::new(
            None,
            1.0,
            3.0,
            AtomicUsize::new(1),
            1,
            game.get_action_size(),
        );

        let probs = mcts.get_action_probs(&game, 1.0).await;

        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(probs.iter().all(|probability| *probability >= 0.0));
    }

    #[test]
    fn get_action_by_sample_respects_probs_and_handles_zero_total() {
        let mcts = MCTS::new(None, 1.0, 3.0, AtomicUsize::new(1), 1, 4);

        // 单点分布:必然返回唯一非零项
        assert_eq!(mcts.get_action_by_sample(&[0.0, 1.0, 0.0, 0.0]), 1);
        assert_eq!(mcts.get_action_by_sample(&[0.0, 0.0, 0.0, 1.0]), 3);

        // 全零分布:退化为返回 0
        assert_eq!(mcts.get_action_by_sample(&[0.0, 0.0, 0.0, 0.0]), 0);

        // 概率和不恰好为 1(浮点误差):仍只返回非零概率的动作
        let probs = [0.3, 0.3, 0.3999999, 0.0];
        for _ in 0..1000 {
            let action = mcts.get_action_by_sample(&probs);
            assert!(action < 3 && probs[action as usize] > 0.0);
        }

        // 均匀分布:多次采样应覆盖全部非零项
        let mut seen = [false; 4];
        for _ in 0..1000 {
            let action = mcts.get_action_by_sample(&[0.25, 0.25, 0.25, 0.25]);
            seen[action as usize] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[tokio::test]
    async fn past_deadline_stops_after_first_batch() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let mcts = MCTS::new(
            None,
            1.0,
            3.0,
            AtomicUsize::new(2048),
            cfg::DEFAULT_SIM_PER_BATCH_NUM,
            game.get_action_size(),
        );

        let deadline = Instant::now();
        let probs = mcts
            .get_action_probs_within(&game, 1.0, Some(deadline))
            .await;

        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let root_visits = mcts.root.borrow().visits.borrow().load(Ordering::SeqCst);
        assert_eq!(root_visits, cfg::DEFAULT_SIM_PER_BATCH_NUM as usize);
    }

    #[tokio::test]
    async fn future_deadline_runs_all_configured_simulations() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let sims = 32;
        let mcts = MCTS::new(
            None,
            1.0,
            3.0,
            AtomicUsize::new(sims),
            cfg::DEFAULT_SIM_PER_BATCH_NUM,
            game.get_action_size(),
        );

        let deadline = Instant::now() + Duration::from_secs(60);
        let probs = mcts
            .get_action_probs_within(&game, 1.0, Some(deadline))
            .await;

        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let expected = (sims / cfg::DEFAULT_SIM_PER_BATCH_NUM as usize)
            * cfg::DEFAULT_SIM_PER_BATCH_NUM as usize;
        let root_visits = mcts.root.borrow().visits.borrow().load(Ordering::SeqCst);
        assert_eq!(root_visits, expected);
    }

    #[tokio::test]
    async fn simulation_within_reporting_calls_reporter_at_intervals() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let sims_per_batch = 1u8;
        let sims = 16usize;
        let mcts = MCTS::new(
            None,
            1.0,
            3.0,
            AtomicUsize::new(sims),
            sims_per_batch,
            game.get_action_size(),
        );

        let reports = std::cell::Cell::new(0usize);
        mcts.simulation_within_reporting(&game, None, Duration::ZERO, |_| {
            reports.set(reports.get() + 1);
        })
        .await;

        assert_eq!(reports.get(), sims);
    }

    #[test]
    fn build_mcts_tree_simply() {
        use std::rc::Rc;

        let root = Rc::new(MCTSNode::new());
        // initially leaf
        assert!(root.is_leaf());

        // expand two children
        assert!(root.expand(1, 0.5));
        assert!(root.expand(2, 0.3));
        // cannot expand same action twice
        assert!(!root.expand(1, 0.5));

        assert!(!root.is_leaf());

        // select should return first child when visits are zero (tie-breaker)
        let sel = root.select(1.0, 3.0).unwrap();
        assert_eq!(sel.action, 1);

        // backpropagate from child and check visits and total_value propagation
        // record parent before
        let child = sel;
        let parent = child.parent().unwrap();

        child.backpropagate(1.0);

        assert_eq!(
            child
                .visits
                .borrow()
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert!((*child.total_value.borrow() - 1.0).abs() < 1e-12);

        assert_eq!(
            parent
                .visits
                .borrow()
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert!((*parent.total_value.borrow() + 1.0).abs() < 1e-12);
    }
}
