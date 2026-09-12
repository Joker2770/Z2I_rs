// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

//! Weight health probe: a handful of fixed positions that a sane network must get right.
//!
//! A destroyed weight is not a rare accident. A training round on corrupt targets, a
//! structure conversion whose distillation went wrong, or a stale ONNX companion file all
//! produce a model that *loads and plays* while being no better than a random policy. Such
//! a model burns a whole evaluation round (or worse, gets promoted), so it is worth one
//! second to ask it three questions first:
//!
//! * **Can you still win in one?** The side to move has exactly one immediate five, and the
//!   raw policy (no search) must put its top-1 on that point.
//! * **Can you still see the threat?** The opponent threatens five; the block is the only
//!   move that does not lose on the spot.
//! * **Does the value head still agree with reality?** A won position must evaluate
//!   positive, a lost one negative, and both decisively so.
//!
//! Every probe is built as a **parity-correct** position (`b == w` on Black's turn,
//! `b == w + 1` on White's turn) whose last move is a harmless Black stone, because that is
//! the only shape the learner ever trains on: probing an unreachable position would test
//! the network off-distribution and produce false failures.
//!
//! All probes are "White to move" on purpose. White has no forbidden moves under any rule
//! this engine supports, so the expected move stays valid for FreeStyle, Standard, Caro and
//! Renju alike, and the probing positions stay free of forbidden-move ambiguity.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use crate::{
    configuration::cfg,
    gomoku::{GameStage, Gomoku},
    ortopt::NeuralNetwork,
    rule::Color,
};

/// What a probe checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProbeKind {
    /// The side to move can win immediately at exactly one point.
    WinInOne,
    /// The opponent threatens five; this move is the only one that does not lose at once.
    MustBlock,
    /// The side to move is winning (`true`) or already lost (`false`).
    ValueSign { winning: bool },
}

/// One fixed probe position.
#[derive(Clone, Debug)]
pub struct ProbePosition {
    pub name: &'static str,
    /// Stones in play order; the last entry is the move just played, always a harmless
    /// Black stone (every probe is White to move, so Black moved last).
    pub stones: Vec<(u16, Color)>,
    /// The point the policy must prefer; `None` for value-only probes.
    pub expected: Option<u16>,
    /// Whether a failure should reject the weight or only warn.
    pub critical: bool,
    pub kind: ProbeKind,
}

// All offsets below are relative to the board centre, so a single 15x15 layout describes the
// whole set. Keep every offset inside [-7, 7]: the centre of a 15x15 board is 7, and an
// offset outside that range would leave the board (the tests assert this).

/// White four with a single completion point at `(0, -1)` (row 7, col 6).
const WHITE_FOUR_H: &[(i32, i32)] = &[(0, -3), (0, -2), (0, 0), (0, 1)];
/// White four with a single completion point at `(-1, 0)` (row 6, col 7).
const WHITE_FOUR_V: &[(i32, i32)] = &[(-4, 0), (-3, 0), (-2, 0), (0, 0)];
/// White four with a single completion point at `(1, 1)` (row 8, col 8).
const WHITE_FOUR_D: &[(i32, i32)] = &[(-2, -2), (-1, -1), (0, 0), (2, 2)];
/// White open four, winning at `(0, -4)` or `(0, 1)`: the mover is simply winning.
const WHITE_OPEN_FOUR: &[(i32, i32)] = &[(0, -3), (0, -2), (0, -1), (0, 0)];

/// Five quiet Black stones in the far corner: two pairs and one lone stone, so no 5-window
/// holds more than two of them. Black has no threat and no forbidden shape of its own, and
/// the lone last stone keeps the "move just played" semantics harmless.
const QUIET_BLACK: &[(i32, i32)] = &[(-7, -7), (-7, -6), (-5, -7), (-5, -6), (-3, -4)];
/// Four quiet White stones: the same corner shape without the lone stone. Used where Black
/// holds the threat, so White must have no threat of its own.
const QUIET_WHITE: &[(i32, i32)] = &[(-7, -7), (-7, -6), (-5, -7), (-5, -6)];

/// Black four at row 10 cols 3,4,5,7 with a single completion point at col 6, plus one
/// harmless corner stone as the move just played.
const BLOCK_BLACK: &[(i32, i32)] = &[(3, -4), (3, -3), (3, -2), (3, 0), (-3, -4)];
/// Black open four at row 10 cols 3,4,5,6 (winning at col 2 or col 7, so unstoppable) plus
/// one harmless corner stone as the move just played.
const OPEN_BLACK: &[(i32, i32)] = &[(3, -4), (3, -3), (3, -2), (3, -1), (-3, -4)];

/// The completion point of each four above, as `(dr, dc)` from the centre.
const WHITE_FOUR_H_WIN: (i32, i32) = (0, -1);
const WHITE_FOUR_V_WIN: (i32, i32) = (-1, 0);
const WHITE_FOUR_D_WIN: (i32, i32) = (1, 1);
const BLOCK_BLACK_WIN: (i32, i32) = (3, -1);

/// Map a centre-relative offset to a board index.
fn at(centre: i32, offset: (i32, i32), board_size: i32) -> u16 {
    let row = centre + offset.0;
    let col = centre + offset.1;
    debug_assert!(
        (0..board_size).contains(&row) && (0..board_size).contains(&col),
        "probe offset {offset:?} leaves the board"
    );
    (row * board_size + col) as u16
}

/// Place a group of offsets as Black stones.
fn place_black(centre: i32, offsets: &[(i32, i32)], board_size: i32) -> Vec<(u16, Color)> {
    offsets
        .iter()
        .map(|&offset| (at(centre, offset, board_size), Color::Black))
        .collect()
}

/// Build the probe set for this board configuration.
///
/// The layouts assume a 15x15 board with five in a row. Other configurations get an empty
/// set, which [`score_probes`] reads as "nothing to check" rather than a failure.
pub fn probe_positions(board_size: u8, n_in_row: u8) -> Vec<ProbePosition> {
    if board_size != 15 || n_in_row != 5 {
        return Vec::new();
    }
    let n = i32::from(board_size);
    let centre = (n - 1) / 2;

    // White moves in every probe, so its stones come first and Black -- one stone ahead,
    // last to move -- closes the list.
    let build = |name: &'static str,
                 white: &[(i32, i32)],
                 black: &[(i32, i32)],
                 expected: Option<u16>,
                 critical: bool,
                 kind: ProbeKind| {
        let mut stones: Vec<(u16, Color)> = white
            .iter()
            .map(|&offset| (at(centre, offset, n), Color::White))
            .collect();
        stones.extend(place_black(centre, black, n));
        debug_assert_eq!(
            stones.iter().filter(|(_, c)| *c == Color::Black).count(),
            stones.iter().filter(|(_, c)| *c == Color::White).count() + 1,
            "{name} must be parity correct for White to move"
        );
        ProbePosition {
            name,
            stones,
            expected,
            critical,
            kind,
        }
    };

    vec![
        build(
            "win in one (horizontal gap)",
            WHITE_FOUR_H,
            QUIET_BLACK,
            Some(at(centre, WHITE_FOUR_H_WIN, n)),
            true,
            ProbeKind::WinInOne,
        ),
        build(
            "win in one (vertical gap)",
            WHITE_FOUR_V,
            QUIET_BLACK,
            Some(at(centre, WHITE_FOUR_V_WIN, n)),
            true,
            ProbeKind::WinInOne,
        ),
        build(
            "win in one (diagonal gap)",
            WHITE_FOUR_D,
            QUIET_BLACK,
            Some(at(centre, WHITE_FOUR_D_WIN, n)),
            true,
            ProbeKind::WinInOne,
        ),
        // Warn only: a policy that prefers a counter-threat over the block is weak rather
        // than broken, and this must not reject an otherwise healthy candidate.
        build(
            "must block the four",
            QUIET_WHITE,
            BLOCK_BLACK,
            Some(at(centre, BLOCK_BLACK_WIN, n)),
            false,
            ProbeKind::MustBlock,
        ),
        build(
            "value: mover wins",
            WHITE_OPEN_FOUR,
            QUIET_BLACK,
            None,
            true,
            ProbeKind::ValueSign { winning: true },
        ),
        build(
            "value: mover loses",
            QUIET_WHITE,
            OPEN_BLACK,
            None,
            true,
            ProbeKind::ValueSign { winning: false },
        ),
    ]
}

/// One probe's raw measurement from a single forward pass.
#[derive(Clone, Debug)]
pub struct ProbeMeasurement {
    pub name: &'static str,
    pub kind: ProbeKind,
    pub critical: bool,
    pub expected: Option<u16>,
    /// Policy argmax and its probability.
    pub top1: u16,
    pub top1_prob: f64,
    /// Probability mass the network returned (1.0 for a healthy softmax).
    pub prob_sum: f64,
    pub value: f64,
}

/// How much a network reacts to the constant colour plane (channel 3) being flipped.
///
/// Channel 3 carries the absolute side-to-move colour, which on every reachable position is
/// already a function of the two stone planes: this engine never passes (a forbidden Black
/// move ends the game instead), so Black to move implies equal stone counts and White to move
/// implies Black is one stone ahead. The plane is therefore informationally redundant -- for
/// Renju as well -- and this measurement says whether a trained network *uses* it anyway:
/// a value near zero means the plane is ignored, a large value means the network leans on it
/// (which is the learnability the plane was added for, at the cost of a real input).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColourPlaneSensitivity {
    pub max_value_shift: f64,
    pub max_prob_shift: f64,
}

impl ColourPlaneSensitivity {
    /// Whether the network is indifferent to the plane (numeric noise only).
    pub fn is_ignored(&self) -> bool {
        self.max_value_shift < 0.01 && self.max_prob_shift < 0.01
    }
}

/// Measure the flip of channel 3 over the probe set.
pub fn colour_plane_sensitivity(
    measurements: &[ProbeMeasurement],
    flipped: &[ProbeMeasurement],
) -> ColourPlaneSensitivity {
    let mut max_value_shift: f64 = 0.0;
    let mut max_prob_shift: f64 = 0.0;
    for (normal, other) in measurements.iter().zip(flipped.iter()) {
        max_value_shift = max_value_shift.max((normal.value - other.value).abs());
        // policy agreement: 1 - sum(min(p_a, p_b)) would need the full vectors, so compare
        // the top-1 probability and whether the argmax moved
        max_prob_shift = max_prob_shift.max((normal.top1_prob - other.top1_prob).abs());
        if normal.top1 != other.top1 {
            max_prob_shift = max_prob_shift.max(normal.top1_prob);
        }
    }
    ColourPlaneSensitivity {
        max_value_shift,
        max_prob_shift,
    }
}

/// One pass/fail line of the report.
#[derive(Clone, Debug, PartialEq)]
pub struct Criterion {
    pub name: String,
    pub critical: bool,
    pub passed: bool,
    pub detail: String,
}

/// The full probe result.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeReport {
    pub criteria: Vec<Criterion>,
    pub elapsed: Duration,
    /// Mean policy top-1 probability over the probes.
    ///
    /// This is the one number that separates "weak" from "destroyed": a destroyed weight
    /// returns the uniform 1/225 here, while a merely weaker weight keeps the tactics but
    /// with visibly lower confidence (a healthy model answers the block probe around 0.8,
    /// a diffuse one around 0.2). Comparing the candidate's sharpness against best's in the
    /// evaluation log tells you whether a lost round is a regression or a real gap.
    pub sharpness: f64,
}

impl ProbeReport {
    /// Whether the weight is usable: every critical criterion passed.
    pub fn passed(&self) -> bool {
        self.criteria.iter().all(|c| c.passed || !c.critical)
    }

    pub fn failures(&self) -> Vec<&Criterion> {
        self.criteria.iter().filter(|c| !c.passed).collect()
    }

    /// One-line verdict, for comparing two weights side by side in a log.
    pub fn headline(&self) -> String {
        format!(
            "weight probe: {} ({:.1}s, policy sharpness {:.3})",
            if self.passed() { "PASS" } else { "FAIL" },
            self.elapsed.as_secs_f64(),
            self.sharpness
        )
    }

    /// Compact multi-line summary for stdout and the evaluation log.
    pub fn summary(&self) -> String {
        let mut text = format!("{}\n", self.headline());
        for criterion in &self.criteria {
            text.push_str(&format!(
                "  {:<18} {:<5} {}{}\n",
                criterion.name,
                if criterion.passed { "ok" } else { "FAIL" },
                criterion.detail,
                if criterion.critical { "" } else { " (advisory)" }
            ));
        }
        text
    }
}

/// Turn raw measurements into a verdict.
///
/// Geometry and the value sign are hard requirements: a network that returns no usable
/// policy mass or a value with the wrong sign will not play well at any simulation count.
/// Tactics are required in bulk rather than individually -- a weak but sane weight may miss
/// one of the three win-in-one shapes (which are only seen through a raw policy, with no
/// search), while a destroyed one misses all of them. The block probe only warns.
pub fn score_probes(measurements: &[ProbeMeasurement], elapsed: Duration) -> ProbeReport {
    let mut criteria = Vec::new();

    let bad_geometry: Vec<&ProbeMeasurement> = measurements
        .iter()
        .filter(|m| {
            !m.value.is_finite()
                || !m.top1_prob.is_finite()
                || !(0.9..=1.1).contains(&m.prob_sum)
                || m.value.abs() > 1.0 + 1e-6
        })
        .collect();
    criteria.push(Criterion {
        name: "outputs".to_string(),
        critical: true,
        passed: bad_geometry.is_empty(),
        detail: if bad_geometry.is_empty() {
            format!(
                "{} position(s), |v| max {:.3}",
                measurements.len(),
                measurements
                    .iter()
                    .map(|m| m.value.abs())
                    .fold(0.0, f64::max)
            )
        } else {
            format!(
                "{} position(s) returned non-finite, unnormalized or out-of-range output",
                bad_geometry.len()
            )
        },
    });

    let win_in_one: Vec<&ProbeMeasurement> = measurements
        .iter()
        .filter(|m| m.kind == ProbeKind::WinInOne)
        .collect();
    let win_hits = win_in_one
        .iter()
        .filter(|m| m.expected == Some(m.top1))
        .count();
    // half the shapes is the line: a uniform policy (which is what a zeroed network falls
    // back to) answers index 0 everywhere and solves none of them
    let win_ok = win_hits >= win_in_one.len().div_ceil(2);
    criteria.push(Criterion {
        name: "win in one".to_string(),
        critical: !win_in_one.is_empty(),
        passed: win_ok,
        detail: if win_in_one.is_empty() {
            "no shapes on this board".to_string()
        } else {
            // always name every shape with its confidence: a miss tells you which tactic is
            // gone, and the probabilities are what make a weak policy comparable to best's
            let shapes: Vec<String> = win_in_one
                .iter()
                .map(|m| {
                    let label = m.name.trim_start_matches("win in one (").trim_end_matches(')');
                    if m.expected == Some(m.top1) {
                        format!("{label} p={:.3}", m.top1_prob)
                    } else {
                        format!("{label} MISS p={:.3}", m.top1_prob)
                    }
                })
                .collect();
            format!(
                "{}/{} solved: {}",
                win_hits,
                win_in_one.len(),
                shapes.join(", ")
            )
        },
    });

    for measurement in measurements
        .iter()
        .filter(|m| m.kind == ProbeKind::MustBlock)
    {
        let ok = measurement.expected == Some(measurement.top1);
        criteria.push(Criterion {
            name: "block the four".to_string(),
            critical: measurement.critical,
            passed: ok,
            detail: if ok {
                format!("top1 {} p={:.3}", measurement.top1, measurement.top1_prob)
            } else {
                format!(
                    "top1 {} p={:.3}, expected {}",
                    measurement.top1,
                    measurement.top1_prob,
                    measurement.expected.unwrap_or_default()
                )
            },
        });
    }

    for measurement in measurements
        .iter()
        .filter(|m| matches!(m.kind, ProbeKind::ValueSign { .. }))
    {
        let ProbeKind::ValueSign { winning } = measurement.kind else {
            continue;
        };
        let sign_ok = if winning {
            measurement.value > 0.0
        } else {
            measurement.value < 0.0
        };
        // a decided position evaluating near zero is a broken value head even when the sign
        // happens to come out right
        let decisive = measurement.value.abs() >= 0.5;
        criteria.push(Criterion {
            name: if winning {
                "value (winning)".to_string()
            } else {
                "value (losing)".to_string()
            },
            critical: true,
            passed: sign_ok && decisive,
            detail: format!(
                "v={:+.3}{}",
                measurement.value,
                if sign_ok && !decisive {
                    " (sign ok but too flat)"
                } else {
                    ""
                }
            ),
        });
    }

    let sharpness = if measurements.is_empty() {
        0.0
    } else {
        measurements.iter().map(|m| m.top1_prob).sum::<f64>() / measurements.len() as f64
    };

    ProbeReport {
        criteria,
        elapsed,
        sharpness,
    }
}

/// Run every probe with one forward pass each.
pub async fn probe_weight(
    weights_dir: &Path,
    weight_id: i32,
    intra_thread_num: u8,
) -> Result<ProbeReport, String> {
    if weight_id < 0 {
        return Err("no weight to probe".to_string());
    }
    let model_path = weights_dir.join(format!("{weight_id}.onnx"));
    if !model_path.exists() {
        return Err(format!("{} does not exist", model_path.display()));
    }
    let network = NeuralNetwork::new(
        &model_path,
        cfg::MAX_BATCH_SIZE as usize,
        intra_thread_num,
    )
    .map_err(|error| format!("load {} error: {error}", model_path.display()))?;

    let started = Instant::now();
    let measurements = probe_positions(cfg::BOARD_SIZE, cfg::N_IN_ROW);
    let mut results = Vec::with_capacity(measurements.len());
    let mut flipped_results = Vec::with_capacity(measurements.len());
    for probe in measurements {
        let mut game = Gomoku::new(cfg::BOARD_SIZE, cfg::N_IN_ROW)
            .ok_or_else(|| "board configuration rejected".to_string())?;
        if !game.load_position(&probe.stones, Color::White) {
            return Err(format!("probe {} is not a legal position", probe.name));
        }
        if game.get_game_status().0 != GameStage::Running {
            return Err(format!("probe {} is already decided", probe.name));
        }
        let (probs, value) = infer(&network, &game, probe.name).await?;
        results.push(measurement_for(&probe, &probs, value));

        // same position with the constant colour plane negated: on every reachable position
        // that plane is already implied by the stone planes, so a network that has really
        // learned the game has no reason to move its answer
        if cfg::INPUT_CHANNEL_SIZE >= 4 {
            let mut state = network.transform_gomoku_2_tensor(&game);
            let plane = game.get_board_size() as usize * game.get_board_size() as usize;
            for value in state[3 * plane..4 * plane].iter_mut() {
                *value = -*value;
            }
            let receiver = network
                .commit_state(state)
                .map_err(|error| format!("probe {} could not be queued: {error}", probe.name))?;
            let (probs, value) = receiver
                .await
                .map_err(|error| format!("probe {} inference dropped: {error}", probe.name))?
                .map_err(|error| format!("probe {} inference failed: {error}", probe.name))?;
            flipped_results.push(measurement_for(&probe, &probs, value));
        }
    }

    let mut report = score_probes(&results, started.elapsed());
    if !flipped_results.is_empty() {
        let sensitivity = colour_plane_sensitivity(&results, &flipped_results);
        report.criteria.push(Criterion {
            name: "colour plane".to_string(),
            // informational: it describes the network, it does not judge it
            critical: false,
            passed: true,
            detail: format!(
                "flipping ch3 shifts value by {:.3} and policy top-1 by {:.3} -- {}",
                sensitivity.max_value_shift,
                sensitivity.max_prob_shift,
                if sensitivity.is_ignored() {
                    "ignored, so the plane is redundant here"
                } else {
                    "used by the network"
                }
            ),
        });
    }
    Ok(report)
}

/// Run one forward pass for a probe position.
async fn infer(
    network: &NeuralNetwork,
    game: &Gomoku,
    name: &str,
) -> Result<(Vec<f64>, f64), String> {
    let receiver = network
        .commit(game)
        .map_err(|error| format!("probe {name} could not be queued: {error}"))?;
    receiver
        .await
        .map_err(|error| format!("probe {name} inference dropped: {error}"))?
        .map_err(|error| format!("probe {name} inference failed: {error}"))
}

/// Fold one pass into a measurement.
fn measurement_for(probe: &ProbePosition, probs: &[f64], value: f64) -> ProbeMeasurement {
    let (top1, top1_prob) = probs
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |best, (index, prob)| {
            if *prob > best.1 { (index, *prob) } else { best }
        });
    ProbeMeasurement {
        name: probe.name,
        kind: probe.kind,
        critical: probe.critical,
        expected: probe.expected,
        top1: top1 as u16,
        top1_prob,
        prob_sum: probs.iter().sum(),
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOARD: u8 = 15;
    const ROW: u8 = 5;

    fn board_index(index: u16) -> (usize, usize) {
        (
            index as usize / BOARD as usize,
            index as usize % BOARD as usize,
        )
    }

    /// A fresh game with the probe's stones and `next` to move.
    fn game_from(probe: &ProbePosition, next: Color) -> Gomoku {
        let mut game = Gomoku::new(BOARD, ROW).expect("valid board");
        assert!(
            game.load_position(&probe.stones, next),
            "{} must load",
            probe.name
        );
        game
    }

    /// Whether `side` wins immediately by playing `candidate`.
    fn wins_immediately(probe: &ProbePosition, side: Color, candidate: u16) -> bool {
        let mut game = game_from(probe, side);
        if !game.execute_move(candidate) {
            return false;
        }
        let status = *game.get_game_status();
        status.0 == GameStage::End && status.1 == side
    }

    fn probes_of(kind: ProbeKind) -> Vec<ProbePosition> {
        probe_positions(BOARD, ROW)
            .into_iter()
            .filter(|probe| probe.kind == kind)
            .collect()
    }

    #[test]
    fn probes_are_parity_correct_running_positions() {
        let probes = probe_positions(BOARD, ROW);
        assert_eq!(probes.len(), 6, "the probe set is expected to be stable");
        for probe in &probes {
            let count = |color: Color| probe.stones.iter().filter(|(_, c)| *c == color).count();
            // White to move in a Black-first game: Black is exactly one stone ahead
            assert_eq!(
                count(Color::Black),
                count(Color::White) + 1,
                "{} is off-parity",
                probe.name
            );
            // and the last move is Black's, which is what "White to move" means
            assert_eq!(
                probe.stones.last().expect("stones").1,
                Color::Black,
                "{} must end with a Black move",
                probe.name
            );
            let mut game = game_from(probe, Color::White);
            assert_eq!(*game.get_cur_color(), Color::White);
            assert_eq!(
                game.get_game_status().0,
                GameStage::Running,
                "{} must not already be decided",
                probe.name
            );
        }
    }

    #[test]
    fn every_win_in_one_probe_has_exactly_one_winning_move() {
        let wins = probes_of(ProbeKind::WinInOne);
        assert_eq!(wins.len(), 3);
        for probe in wins {
            let expected = probe.expected.expect("a win-in-one probe has a target");
            let (row, col) = board_index(expected);
            assert_eq!(
                game_from(&probe, Color::White).get_board()[row][col],
                Color::Blank,
                "{}: the target point must be empty",
                probe.name
            );
            assert!(
                wins_immediately(&probe, Color::White, expected),
                "{}: playing the target must win for White",
                probe.name
            );
            // uniqueness: no other legal move wins, otherwise the probe is ambiguous
            for candidate in 0..(BOARD as u16 * BOARD as u16) {
                if candidate == expected {
                    continue;
                }
                assert!(
                    !wins_immediately(&probe, Color::White, candidate),
                    "{}: move {candidate} also wins, so the probe does not pin the answer",
                    probe.name
                );
            }
        }
    }

    #[test]
    fn the_block_probe_threat_is_real_and_white_cannot_win_first() {
        let blocks = probes_of(ProbeKind::MustBlock);
        assert_eq!(blocks.len(), 1);
        let probe = &blocks[0];
        let block = probe.expected.expect("a block probe has a target");

        // Black, to move on the same stones, wins immediately at the target point
        assert!(
            wins_immediately(probe, Color::Black, block),
            "Black must threaten five at the block point"
        );
        // so the block is the only move that does not lose at once
        for candidate in 0..(BOARD as u16 * BOARD as u16) {
            assert!(
                !wins_immediately(probe, Color::White, candidate),
                "White must not have a win in one at {candidate}"
            );
        }
        let (row, col) = board_index(block);
        assert_eq!(
            game_from(probe, Color::White).get_board()[row][col],
            Color::Blank,
            "the block point must be empty"
        );
    }

    #[test]
    fn value_probes_are_decided_by_an_open_four() {
        for probe in probe_positions(BOARD, ROW) {
            let ProbeKind::ValueSign { winning } = probe.kind else {
                continue;
            };
            // the side holding the open four wins no matter who moves first
            let four_holder = if winning { Color::White } else { Color::Black };
            let winning_points: Vec<u16> = (0..(BOARD as u16 * BOARD as u16))
                .filter(|candidate| wins_immediately(&probe, four_holder, *candidate))
                .collect();
            assert!(
                winning_points.len() >= 2,
                "{}: an open four needs two completion points, found {}",
                probe.name,
                winning_points.len()
            );
            assert_eq!(
                game_from(&probe, Color::White).get_game_status().0,
                GameStage::Running,
                "{} must not already be over",
                probe.name
            );
        }
    }

    fn measurement(
        name: &'static str,
        kind: ProbeKind,
        expected: Option<u16>,
        top1: u16,
        value: f64,
    ) -> ProbeMeasurement {
        ProbeMeasurement {
            name,
            kind,
            critical: true,
            expected,
            top1,
            top1_prob: 0.7,
            prob_sum: 1.0,
            value,
        }
    }

    fn healthy() -> Vec<ProbeMeasurement> {
        let mut probes = vec![
            measurement("h", ProbeKind::WinInOne, Some(11), 11, 0.4),
            measurement("v", ProbeKind::WinInOne, Some(22), 22, 0.5),
            measurement("d", ProbeKind::WinInOne, Some(33), 33, 0.6),
            measurement("b", ProbeKind::MustBlock, Some(44), 44, -0.3),
            measurement("w", ProbeKind::ValueSign { winning: true }, None, 5, 0.95),
            measurement("l", ProbeKind::ValueSign { winning: false }, None, 6, -0.92),
        ];
        // the block probe is advisory only, exactly as `probe_positions` marks it
        probes[3].critical = false;
        probes
    }

    #[test]
    fn a_healthy_weight_passes() {
        let report = score_probes(&healthy(), Duration::ZERO);
        assert!(report.passed(), "{}", report.summary());
        assert!(report.failures().is_empty());
    }

    #[test]
    fn a_destroyed_weight_fails() {
        // a zeroed network falls back to a uniform policy (top-1 is index 0 everywhere) and
        // a flat value of 0
        let dead: Vec<ProbeMeasurement> = healthy()
            .into_iter()
            .map(|mut m| {
                m.top1 = 0;
                m.top1_prob = 1.0 / 225.0;
                m.value = 0.0;
                m
            })
            .collect();
        let report = score_probes(&dead, Duration::ZERO);
        assert!(!report.passed());
        let failed: Vec<&str> = report
            .failures()
            .iter()
            .map(|criterion| criterion.name.as_str())
            .collect();
        assert!(failed.contains(&"win in one"), "{failed:?}");
        assert!(
            failed.iter().any(|name| name.starts_with("value")),
            "{failed:?}"
        );
    }

    #[test]
    fn an_inverted_value_head_fails_even_with_good_tactics() {
        let mut inverted = healthy();
        for m in &mut inverted {
            if matches!(m.kind, ProbeKind::ValueSign { .. }) {
                m.value = -m.value;
            }
        }
        let report = score_probes(&inverted, Duration::ZERO);
        assert!(!report.passed());
        assert_eq!(report.failures().len(), 2, "{}", report.summary());
    }

    #[test]
    fn a_flat_value_on_decided_positions_fails() {
        let mut flat = healthy();
        for m in &mut flat {
            if matches!(m.kind, ProbeKind::ValueSign { .. }) {
                m.value = 0.01 * m.value.signum();
            }
        }
        assert!(!score_probes(&flat, Duration::ZERO).passed());
    }

    #[test]
    fn one_missed_tactic_is_tolerated_and_the_block_only_warns() {
        let mut weak = healthy();
        weak[0].top1 = 99;
        weak[3].top1 = 98; // the block probe
        let report = score_probes(&weak, Duration::ZERO);
        assert!(report.passed(), "{}", report.summary());
        assert_eq!(report.failures().len(), 1);
        assert!(!report.failures()[0].critical, "the block probe only warns");
        // two misses out of three shapes is the line
        weak[1].top1 = 97;
        assert!(!score_probes(&weak, Duration::ZERO).passed());
    }

    #[test]
    fn unusable_outputs_fail_the_geometry_check() {
        let mut broken = healthy();
        broken[0].prob_sum = 0.0;
        broken[1].value = f64::NAN;
        let report = score_probes(&broken, Duration::ZERO);
        assert!(!report.passed());
        assert_eq!(report.failures()[0].name, "outputs");
    }

    #[test]
    fn the_colour_plane_flip_only_touches_channel_three() {
        let probe = probe_positions(BOARD, ROW)
            .into_iter()
            .next()
            .expect("a probe exists");
        let game = game_from(&probe, Color::White);
        let state = NeuralNetwork::transform_board_2_tensor(
            game.get_board(),
            game.get_board_size(),
            game.get_last_move(),
            game.get_cur_color(),
        );
        let plane = BOARD as usize * BOARD as usize;
        assert_eq!(state.len(), 4 * plane, "this build feeds four channels");
        // White to move: the constant plane carries -1 everywhere
        assert!(state[3 * plane..].iter().all(|value| *value == -1.0));
        assert!(
            state[..3 * plane]
                .iter()
                .any(|value| *value != 0.0),
            "the stone planes must be populated for the flip to be measurable"
        );

        let mut flipped = state.clone();
        for value in flipped[3 * plane..].iter_mut() {
            *value = -*value;
        }
        assert!(flipped[3 * plane..].iter().all(|value| *value == 1.0));
        assert_eq!(state[..3 * plane], flipped[..3 * plane], "only ch3 may change");
        assert_ne!(state, flipped);
    }

    #[test]
    fn colour_plane_sensitivity_reports_the_largest_shift() {
        let normal = healthy();
        let mut flipped = normal.clone();
        // the network leans on the plane: value moves and the argmax changes
        flipped[0].value += 0.4;
        flipped[1].top1 = 123;
        let sensitivity = colour_plane_sensitivity(&normal, &flipped);
        assert!((sensitivity.max_value_shift - 0.4).abs() < 1e-12);
        assert!(sensitivity.max_prob_shift >= normal[1].top1_prob);
        assert!(!sensitivity.is_ignored());

        // and indifference is the other extreme
        let same = colour_plane_sensitivity(&normal, &normal);
        assert!(same.is_ignored());
        assert_eq!(same.max_value_shift, 0.0);
    }

    #[test]
    fn sharpness_reports_the_mean_top1_confidence() {
        let report = score_probes(&healthy(), Duration::ZERO);
        // every synthetic measurement uses p = 0.7
        assert!((report.sharpness - 0.7).abs() < 1e-12);
        assert!(report.headline().contains("sharpness 0.700"), "{}", report.headline());
        assert_eq!(score_probes(&[], Duration::ZERO).sharpness, 0.0);
    }

    #[test]
    fn the_shape_detail_always_names_and_scores_every_shape() {
        let mut measurements = healthy();
        let names: Vec<&'static str> = probe_positions(BOARD, ROW)
            .iter()
            .filter(|probe| probe.kind == ProbeKind::WinInOne)
            .map(|probe| probe.name)
            .collect();
        for (measurement, name) in measurements
            .iter_mut()
            .filter(|m| m.kind == ProbeKind::WinInOne)
            .zip(names)
        {
            measurement.name = name;
        }
        // the middle shape misses, and the detail must say so without hiding the others
        measurements[1].top1 = 7;
        let report = score_probes(&measurements, Duration::ZERO);
        let detail = &report
            .criteria
            .iter()
            .find(|criterion| criterion.name == "win in one")
            .expect("the criterion exists")
            .detail;
        assert!(detail.starts_with("2/3 solved:"), "{detail}");
        assert!(detail.contains("horizontal gap p=0.700"), "{detail}");
        assert!(detail.contains("vertical gap MISS p=0.700"), "{detail}");
        assert!(detail.contains("diagonal gap p=0.700"), "{detail}");
    }

    #[test]
    fn other_board_configurations_have_nothing_to_check() {
        assert!(probe_positions(3, 3).is_empty());
        // no probes means no criteria, which must not read as a failure
        let report = score_probes(&[], Duration::ZERO);
        assert!(report.passed());
        assert!(report.summary().contains("PASS"));
    }
}
