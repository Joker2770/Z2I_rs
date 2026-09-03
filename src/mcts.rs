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
        let get_u = |v: usize, c_puct: f64, v_l: f64| -> f64 {
            let vf = v as f64;
            // A node with 0 visits ignores virtual loss in Q (Q == 0 when v == 0), so the
            // exploration term must also account for virtual loss: otherwise every simulation
            // in a batch scores a 0-visit branch identically, picks the same one, and floods
            // the inference queue with duplicate positions.
            c_puct * self.prior_probs * (sum_visits_from_parents as f64).sqrt() / (1.0 + vf + v_l)
        };

        let q = get_q(v, virtual_loss);
        let u = get_u(v, c_puct, virtual_loss);

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

struct TimingStats {
    batch_average: Duration,
    single_average: Duration,
    batch_minimum: Duration,
    single_minimum: Duration,
    final_move_reserve: Duration,
}

pub struct MCTS {
    root: RefCell<Rc<MCTSNode>>,
    neural_network: Option<Rc<RefCell<NeuralNetwork>>>,
    simulation_num: AtomicUsize,
    action_size: u16,
    c_puct: f64,
    c_virtual_loss: f64,
    sims_per_batch: u8,
    /// Serializes reads/writes of the search tree (select descent / expand / backpropagate),
    /// so concurrent simulation tasks cannot race on select. The lock must not be held across `.await`.
    tree_lock: Mutex<()>,
    timing: Mutex<TimingStats>,
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
        Self::new_with_timing(
            neural_network,
            c_puct,
            c_virtual_loss,
            simulation_num,
            sim_per_batch_num,
            action_size,
            cfg::TIME_RESERVE_MS,
            cfg::SINGLE_SIM_RESERVE_MS,
            cfg::FINAL_MOVE_RESERVE_MS,
        )
    }

    pub fn new_with_timing(
        neural_network: Option<Rc<RefCell<NeuralNetwork>>>,
        c_puct: f64,
        c_virtual_loss: f64,
        simulation_num: AtomicUsize,
        sim_per_batch_num: u8,
        action_size: u16,
        time_reserve_ms: u64,
        single_sim_reserve_ms: u64,
        final_move_reserve_ms: u64,
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
            timing: Mutex::new(TimingStats {
                batch_average: Duration::from_millis(time_reserve_ms),
                single_average: Duration::from_millis(single_sim_reserve_ms),
                batch_minimum: Duration::from_millis(time_reserve_ms),
                single_minimum: Duration::from_millis(single_sim_reserve_ms),
                final_move_reserve: Duration::from_millis(final_move_reserve_ms),
            }),
        }
    }

    fn timing_reserve(
        average: Duration,
        minimum: Duration,
        final_move_reserve: Duration,
    ) -> Duration {
        let estimate_ms = average.as_millis().min(u64::MAX as u128) as u64;
        let safe_ms = estimate_ms
            .saturating_mul(5)
            .saturating_div(4)
            .saturating_add(final_move_reserve.as_millis().min(u64::MAX as u128) as u64);
        Duration::from_millis(safe_ms).max(minimum)
    }

    fn batch_reserve(&self) -> Duration {
        let timing = self.timing.lock().unwrap_or_else(|e| e.into_inner());
        Self::timing_reserve(
            timing.batch_average,
            timing.batch_minimum,
            timing.final_move_reserve,
        )
    }

    fn single_reserve(&self) -> Duration {
        let timing = self.timing.lock().unwrap_or_else(|e| e.into_inner());
        Self::timing_reserve(
            timing.single_average,
            timing.single_minimum,
            timing.final_move_reserve,
        )
    }

    fn record_duration(average: &mut Duration, sample: Duration) {
        let old_ms = average.as_millis();
        let sample_ms = sample.as_millis();
        let next_ms = old_ms
            .saturating_mul(3)
            .saturating_add(sample_ms)
            .saturating_div(4)
            .max(1);
        *average = Duration::from_millis(next_ms.min(u64::MAX as u128) as u64);
    }

    fn record_batch_duration(&self, elapsed: Duration) {
        let mut timing = self.timing.lock().unwrap_or_else(|e| e.into_inner());
        Self::record_duration(&mut timing.batch_average, elapsed);
    }

    fn record_single_duration(&self, elapsed: Duration) {
        let mut timing = self.timing.lock().unwrap_or_else(|e| e.into_inner());
        Self::record_duration(&mut timing.single_average, elapsed);
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

    /// Run as many simulation batches as possible before the deadline:
    /// - `deadline` is `None`: run the configured number of simulations;
    /// - start a full batch only when enough time remains for the batch reserve;
    /// - when a full batch no longer fits, run at most one final simulation;
    /// - the first simulation is always started for a finite deadline, including
    ///   `timeout_turn 0`, because a move must still be based on a searched tree.
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
            // Convert visit counts into policy probabilities: π(a) ∝ N(a)^(1/τ).
            // To avoid numerical overflow, compute log π(a) = (1/τ) * ln(N(a)) in the log domain
            // and subtract the max (max_log_prob) before exponentiating — the numerically stable softmax form.
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

    /// Run one batch of concurrent simulations (`sims_per_batch` of them).
    /// A batch boundary is a consistent point of the search tree: the virtual_loss
    /// from simulations in the batch has been rolled back, so it is a good place to
    /// yield control (e.g. to use the opponent's thinking time for background pondering).
    pub async fn simulation_batch(&self, gomoku: &Gomoku) {
        let simulations = (0..self.sims_per_batch).map(|_| self.simulation(gomoku));
        future::join_all(simulations).await;
    }

    pub async fn simulation_within(&self, gomoku: &Gomoku, deadline: Option<Instant>) {
        let sim_batch = self
            .simulation_num
            .load(Ordering::Relaxed)
            .div(self.sims_per_batch as usize)
            .max(1);
        let target_simulations = sim_batch * self.sims_per_batch as usize;
        let mut completed_simulations = 0;
        while completed_simulations < target_simulations {
            if let Some(deadline) = deadline {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if completed_simulations > 0 && remaining <= self.single_reserve() {
                    break;
                }
                if remaining <= self.batch_reserve() {
                    // A single simulation is not safely cancellable, but the first one is
                    // mandatory even when the deadline has already reached its reserve. When a
                    // full batch no longer fits, keep using single simulations while safe time
                    // remains instead of stopping after the first one.
                    let started = Instant::now();
                    self.simulation(gomoku).await;
                    self.record_single_duration(started.elapsed());
                    completed_simulations += 1;
                    continue;
                }
            }
            let batch_size = self.sims_per_batch;
            let started = Instant::now();
            let simulations = (0..batch_size).map(|_| self.simulation(gomoku));
            future::join_all(simulations).await;
            self.record_batch_duration(started.elapsed());
            completed_simulations += batch_size as usize;
        }
    }

    /// Like `simulation_within`, but calls `report` every `report_interval` to send
    /// the root search progress to the manager periodically (open_mind debug output).
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
            .div(self.sims_per_batch as usize)
            .max(1);
        let target_simulations = sim_batch * self.sims_per_batch as usize;
        let mut completed_simulations = 0;
        let mut next_report = Instant::now() + report_interval;
        while completed_simulations < target_simulations {
            if let Some(deadline) = deadline {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if completed_simulations > 0 && remaining <= self.single_reserve() {
                    break;
                }
                if remaining <= self.batch_reserve() {
                    let started = Instant::now();
                    self.simulation(gomoku).await;
                    self.record_single_duration(started.elapsed());
                    completed_simulations += 1;
                    if Instant::now() >= next_report {
                        report(self);
                        next_report = Instant::now() + report_interval;
                    }
                    continue;
                }
            }
            let batch_size = self.sims_per_batch;
            let started = Instant::now();
            let simulations = (0..batch_size).map(|_| self.simulation(gomoku));
            future::join_all(simulations).await;
            self.record_batch_duration(started.elapsed());
            completed_simulations += batch_size as usize;
            if Instant::now() >= next_report {
                report(self);
                next_report = Instant::now() + report_interval;
            }
        }
    }

    /// Whether the search tree root has no children yet (nothing expanded).
    /// A freshly (re)built tree returns `true`, which is useful to assert that
    /// the search tree was actually reset (e.g. after a rule switch).
    pub fn root_is_leaf(&self) -> bool {
        self.root.borrow().is_leaf()
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

    /// Print the action coordinates and visit counts of the root's first-level children:
    /// `DEBUG thinking x1,y1,visits1 x2,y2,visits2 ...`
    /// When there are too many children, filter out those with fewer visits.
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

        // when there are too many children, keep only those with the most visits
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
            // all-zero probabilities (should not happen): fall back to the first
            // action with a non-zero probability
            return probs
                .iter()
                .take(self.action_size as usize)
                .position(|p| *p > 0.0)
                .map(|i| i as u16)
                .unwrap_or(0);
        }
        // normalize by the total before sampling, so floating-point error can't keep
        // the cumulative sum from ever crossing r
        let r: f64 = rand::random();
        let mut accum = 0.0;
        for (i, p) in probs.iter().take(self.action_size as usize).enumerate() {
            accum += p / total;
            if accum > r {
                return i as u16;
            }
        }
        // floating-point edge case: when r is extremely close to 1, return the last
        // action with a non-zero probability
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
        // phase 1: select descent while holding the lock. The lock serializes in-tree
        // selection across concurrent simulation tasks, so they cannot race on select,
        // pick the same branch, or modify virtual_loss concurrently.
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
                        // defensive: `select` has already incremented this child's virtual loss;
                        // roll it back and stop descending, otherwise the loop would re-select the
                        // same illegal child, leaking virtual loss and potentially spinning forever
                        // on a permanently over-penalized node
                        let old_vl = c.virtual_loss.borrow().load(Ordering::SeqCst);
                        c.virtual_loss
                            .borrow_mut()
                            .store(old_vl.saturating_sub(1), Ordering::SeqCst);
                        break;
                    }
                } else {
                    break;
                }
            }

            (node, g)
        }; // release the lock before inference; the lock is not held across .await

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

            // phase 2: expand and backpropagate while holding the lock, keeping tree mutations atomic
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

        // one-hot distribution: must return the single non-zero entry
        assert_eq!(mcts.get_action_by_sample(&[0.0, 1.0, 0.0, 0.0]), 1);
        assert_eq!(mcts.get_action_by_sample(&[0.0, 0.0, 0.0, 1.0]), 3);

        // all-zero distribution: falls back to 0
        assert_eq!(mcts.get_action_by_sample(&[0.0, 0.0, 0.0, 0.0]), 0);

        // sum not exactly 1 (floating-point error): still only returns actions with non-zero probability
        let probs = [0.3, 0.3, 0.3999999, 0.0];
        for _ in 0..1000 {
            let action = mcts.get_action_by_sample(&probs);
            assert!(action < 3 && probs[action as usize] > 0.0);
        }

        // uniform distribution: repeated sampling should cover all non-zero entries
        let mut seen = [false; 4];
        for _ in 0..1000 {
            let action = mcts.get_action_by_sample(&[0.25, 0.25, 0.25, 0.25]);
            seen[action as usize] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[tokio::test]
    async fn past_deadline_still_runs_one_simulation() {
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
        let _ = mcts
            .get_action_probs_within(&game, 1.0, Some(deadline))
            .await;

        // Even an expired deadline must perform the mandatory first simulation.
        let root_visits = mcts.root.borrow().visits.borrow().load(Ordering::SeqCst);
        assert_eq!(root_visits, 1);
    }

    #[tokio::test]
    async fn short_deadline_still_runs_one_simulation() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let mcts = MCTS::new(
            None,
            1.0,
            3.0,
            AtomicUsize::new(2048),
            cfg::DEFAULT_SIM_PER_BATCH_NUM,
            game.get_action_size(),
        );

        for remaining in [0, 100] {
            let deadline = Instant::now() + Duration::from_millis(remaining);
            mcts.simulation_within(&game, Some(deadline)).await;
            let root_visits = mcts.root.borrow().visits.borrow().load(Ordering::SeqCst);
            assert!(root_visits >= 1);
            *mcts.root.borrow_mut() = Rc::new(MCTSNode::new());
        }
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
