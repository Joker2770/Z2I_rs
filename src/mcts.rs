// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use futures::future;
use std::cell::RefCell;
use std::ops::Div;
use std::rc::{Rc, Weak};
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
        if self.children.borrow().is_empty() {
            true
        } else {
            false
        }
    }

    pub fn get_q(&self) -> f64 {
        let v = self.visits.borrow().load(Ordering::SeqCst);
        let q = if v == 0 {
            0.0
        } else {
            let vf = v as f64;
            let w = self.total_value.borrow().clone() as f64;
            w / vf
        };
        q
    }

    pub fn get_u(&self, c_puct: f64) -> f64 {
        let vf = self.visits.borrow().load(Ordering::SeqCst) as f64;
        let u = c_puct * self.prior_probs * vf.sqrt() / (1.0 + vf);
        u
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
        let get_q = |v: usize, v_l: f64| {
            let q = if v == 0 {
                0.0
            } else {
                let vf = v as f64;
                let w = *self.total_value.borrow() as f64;
                (w - v_l) / vf
            };
            q
        };
        let get_u = |v: usize, c_puct: f64| {
            let vf = v as f64;
            let u =
                c_puct * self.prior_probs * (sum_visits_from_parents as f64).sqrt() / (1.0 + vf);
            u
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
                best_value = value;
                best_child = Some(Rc::clone(&c));
            }
        }
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
}

impl MCTS {
    pub fn new(
        neural_network: Option<Rc<RefCell<NeuralNetwork>>>,
        c_puct: f64,
        c_virtual_loss: f64,
        simulation_num: AtomicUsize,
        action_size: u16,
    ) -> Self {
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
        let sim_per_batch = if cfg::SIM_PER_BATCH > 0 && cfg::SIM_PER_BATCH < 256 {
            cfg::SIM_PER_BATCH as usize
        } else {
            1
        };
        let sim_batch = self
            .simulation_num
            .load(Ordering::Relaxed)
            .div(sim_per_batch)
            + 1;
        let mut batches_done = 0usize;
        for _ in 0..sim_batch {
            if batches_done > 0 {
                if let Some(deadline) = deadline {
                    if Instant::now() + Duration::from_millis(cfg::TIME_RESERVE_MS) >= deadline {
                        break;
                    }
                }
            }
            let simulations = (0..sim_per_batch).map(|_| self.simulation(gomoku));
            future::join_all(simulations).await;
            batches_done += 1;
        }
        let priors_size = gomoku.get_action_size() as usize;
        let mut action_probs = vec![0.0; priors_size];
        let root = self.root.borrow();
        let children = root.children.borrow();

        // greedy
        if (temp - cfg::GREEDY_TEMP).abs() < f64::EPSILON {
            let mut best_action = u16::MAX;
            let mut most_visits = 0;
            for c in children.iter() {
                let c_v = c.visits.borrow().load(Ordering::SeqCst) as usize;
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
            let mut sum = 0.0;
            for c in children.iter() {
                let c_v = c.visits.borrow().load(Ordering::SeqCst) as usize;
                if c_v > 0 {
                    let idx = c.action as usize;
                    action_probs[idx] = (c_v as f64).powf((1.0).div(temp));
                    sum = sum + action_probs[idx];
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

    pub fn get_best_action_from_probs(&self, probs: &Vec<f64>) -> u16 {
        let mut best_action = u16::MAX;
        let mut best_probs = -1.0;

        for i in 0..probs.len() {
            if probs[i] > best_probs {
                best_probs = probs[i];
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
        let sim_per_batch = if cfg::SIM_PER_BATCH > 0 && cfg::SIM_PER_BATCH < 256 {
            cfg::SIM_PER_BATCH as usize
        } else {
            1
        };
        let sim_batch = self
            .simulation_num
            .load(Ordering::Relaxed)
            .div(sim_per_batch)
            + 1;
        let mut batches_done = 0usize;
        for _ in 0..sim_batch {
            if batches_done > 0 {
                if let Some(deadline) = deadline {
                    if Instant::now() + Duration::from_millis(cfg::TIME_RESERVE_MS) >= deadline {
                        break;
                    }
                }
            }
            let simulations = (0..sim_per_batch).map(|_| self.simulation(gomoku));
            future::join_all(simulations).await;
            batches_done += 1;
        }

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
            let c_v = c.visits.borrow().load(Ordering::SeqCst) as usize;
            if c_v >= most_visits {
                most_visits = c_v;
                best_action = c.action;
            }
        }
        best_action
    }

    pub async fn simulation_within(&self, gomoku: &Gomoku, deadline: Option<Instant>) {
        let sim_per_batch = if cfg::SIM_PER_BATCH > 0 && cfg::SIM_PER_BATCH < 256 {
            cfg::SIM_PER_BATCH as usize
        } else {
            1
        };
        let sim_batch = self
            .simulation_num
            .load(Ordering::Relaxed)
            .div(sim_per_batch)
            + 1;
        let mut batches_done = 0usize;
        for _ in 0..sim_batch {
            if batches_done > 0 {
                if let Some(deadline) = deadline {
                    if Instant::now() + Duration::from_millis(cfg::TIME_RESERVE_MS) >= deadline {
                        break;
                    }
                }
            }
            let simulations = (0..sim_per_batch).map(|_| self.simulation(gomoku));
            future::join_all(simulations).await;
            batches_done += 1;
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
            let c_v = c.visits.borrow().load(Ordering::SeqCst) as usize;
            if c_v >= most_visits {
                most_visits = c_v;
                best_action = c.action;
            }
        }
        best_action
    }

    pub fn get_action_by_sample(&self, probs: &Vec<f64>) -> u16 {
        let r = rand::random();
        let mut idx = 0;
        let mut accum = 0.0;
        for i in 0..self.action_size {
            accum += probs[i as usize];
            if accum > r {
                idx = i as u16;
                break;
            }
        }
        idx
    }

    pub async fn simulation(&self, gomoku: &Gomoku) {
        let (node, mut g) = {
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
        };

        let (game_stage, color) = {
            let status = g.get_game_status();
            *status
        };
        let mut value = 0.0;

        if game_stage == GameStage::Running {
            let mut action_priors = vec![0.0; self.action_size as usize];
            let legal_hash_tab = g.get_legal_moves();

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
                let mut sum = 0.0;
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
                    let sum: f64 = legal_hash_tab.iter().map(|&x| x as f64).sum();
                    for i in 0..action_priors.len() {
                        action_priors[i] = legal_hash_tab[i] as f64 / sum;
                    }
                }
            } else {
                let sum: f64 = legal_hash_tab.iter().map(|&x| x as f64).sum();
                for i in 0..action_priors.len() {
                    action_priors[i] = legal_hash_tab[i] as f64 / sum;
                }
            }

            for (i, v) in legal_hash_tab.iter().enumerate() {
                if *v == 1 {
                    node.expand(i as u16, action_priors[i]);
                }
            }
        } else {
            value = if color == Color::Blank { 0.0 } else { 1.0 };
        }
        node.backpropagate(value);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn get_best_action_on_first_move_returns_a_legal_action() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let mcts = MCTS::new(None, 1.0, 3.0, AtomicUsize::new(1), game.get_action_size());

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
        let mcts = MCTS::new(None, 1.0, 3.0, AtomicUsize::new(1), game.get_action_size());

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
        let mcts = MCTS::new(None, 1.0, 3.0, AtomicUsize::new(2), game.get_action_size());

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
        let mcts = MCTS::new(None, 1.0, 3.0, AtomicUsize::new(1), game.get_action_size());

        let probs = mcts.get_action_probs(&game, 1.0).await;

        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(probs.iter().all(|probability| *probability >= 0.0));
    }

    #[tokio::test]
    async fn past_deadline_stops_after_first_batch() {
        let game = Gomoku::new(15, 5).expect("valid test board");
        let mcts = MCTS::new(
            None,
            1.0,
            3.0,
            AtomicUsize::new(2048),
            game.get_action_size(),
        );

        let deadline = Instant::now();
        let probs = mcts
            .get_action_probs_within(&game, 1.0, Some(deadline))
            .await;

        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let root_visits = mcts.root.borrow().visits.borrow().load(Ordering::SeqCst);
        assert_eq!(root_visits, cfg::SIM_PER_BATCH as usize);
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
            game.get_action_size(),
        );

        let deadline = Instant::now() + Duration::from_secs(60);
        let probs = mcts
            .get_action_probs_within(&game, 1.0, Some(deadline))
            .await;

        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let expected = (sims / cfg::SIM_PER_BATCH as usize + 1) * cfg::SIM_PER_BATCH as usize;
        let root_visits = mcts.root.borrow().visits.borrow().load(Ordering::SeqCst);
        assert_eq!(root_visits, expected);
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
