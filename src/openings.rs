// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

//! Opening book for colour-paired acceptance evaluation.
//!
//! A rule can be colour symmetric and still not be colour symmetric *as a data
//! distribution*: Black always moves first, so the reachable positions satisfy
//! `b == w` on Black's turn and `b == w + 1` on White's turn, and they are only
//! closed under a colour swap inside the mirror universe where White moves first.
//! Evaluating two networks from the empty board therefore measures the first-move
//! advantage at least as much as the networks, and the deterministic search replays
//! near-identical games every round, so the game outcomes are strongly correlated.
//!
//! This module fixes both problems:
//!
//! * every opening in the book has an **even** ply count, so its colour swap is
//!   again a legal "Black to move" position (equal stone counts);
//! * [`schedule`] plays each opening twice with the colours exchanged, so the
//!   first-move advantage cancels *inside* a pair and the paired comparison has far
//!   more statistical power than the same number of independent games.

use std::{fs, path::Path};

use crate::{
    gomoku::{GameStage, Gomoku},
    rule::Color,
};

/// File name of the optional opening book, relative to the training work dir.
pub const OPENING_FILE: &str = "openings.txt";

/// Swap the two playing colours, leaving [`Color::Blank`] untouched.
fn opposite(color: Color) -> Color {
    match color {
        Color::Black => Color::White,
        Color::White => Color::Black,
        Color::Blank => Color::Blank,
    }
}

/// One book opening: stones in play order, colours alternating from Black.
///
/// Only even-length openings are accepted, which is what makes the colour-swapped
/// twin a legal position for Black to move as well.
#[derive(Clone, Debug, PartialEq)]
pub struct Opening {
    stones: Vec<(u16, Color)>,
}

impl Opening {
    /// Stones in play order, ready for [`Gomoku::load_position`].
    pub fn stones(&self) -> &[(u16, Color)] {
        &self.stones
    }

    /// Ply count of the opening (always even).
    pub fn plies(&self) -> usize {
        self.stones.len()
    }

    /// The opening with both colours exchanged on the same points.
    ///
    /// This is the position of the second game of a colour-swapped pair: net A takes
    /// White while the opponent gets exactly the stones net A had as Black.
    pub fn swapped(&self) -> Self {
        Self {
            stones: self
                .stones
                .iter()
                .map(|&(index, color)| (index, opposite(color)))
                .collect(),
        }
    }

    /// Apply one of the eight dihedral transforms of the square board
    /// (`0..=7`: four rotations, each optionally mirrored).
    ///
    /// All supported rules are invariant under these transforms, so a transformed
    /// opening is the same game type in a different orientation. Spreading a small
    /// book over the eight transforms yields many distinct-looking pairs without
    /// pretending they are independent samples.
    pub fn transformed(&self, transform: usize, board_size: u8) -> Self {
        let n = i32::from(board_size);
        let t = transform % 8;
        let (rotation, mirror) = (t % 4, t / 4 == 1);
        Self {
            stones: self
                .stones
                .iter()
                .map(|&(index, color)| {
                    let (row, col) = (i32::from(index) / n, i32::from(index) % n);
                    let (row, col) = match rotation {
                        0 => (row, col),
                        1 => (col, n - 1 - row),
                        2 => (n - 1 - row, n - 1 - col),
                        _ => (n - 1 - col, row),
                    };
                    let col = if mirror { n - 1 - col } else { col };
                    ((row * n + col) as u16, color)
                })
                .collect(),
        }
    }
}

/// Built-in opening book as centre-relative offsets, so it also works on small
/// boards (offsets falling off the board are dropped, e.g. in a 3x3 build).
/// Every entry has an even ply count, as required by the paired schedule.
const DEFAULT_OPENINGS: &[&[(i32, i32)]] = &[
    &[(0, 0), (0, -1)],
    &[(0, 0), (-2, 2)],
    &[(0, 0), (0, -1), (-1, 0), (1, 0)],
    &[(0, 0), (-1, -1), (1, 1), (-1, 1)],
    &[(-1, -1), (1, 1), (1, -1), (-1, 1)],
    &[(0, 0), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1)],
];

/// Check that an opening is a legal, still-running position for Black to move.
fn is_usable(stones: &[(u16, Color)], board_size: u8, n_in_row: u8) -> bool {
    // an odd ply count would make the colour swap an unreachable position, and a
    // single stone cannot form a pair at all
    if stones.len() < 2 || !stones.len().is_multiple_of(2) {
        return false;
    }
    let Some(mut game) = Gomoku::new(board_size, n_in_row) else {
        return false;
    };
    // load_position rejects out-of-range and duplicated points
    if !game.load_position(stones, Color::Black) {
        return false;
    }
    // an opening that already ends the game is not a position to search from
    matches!(game.get_game_status().0, GameStage::Running)
}

/// The built-in book, filtered down to the openings that are usable on this board.
pub fn default_openings(board_size: u8, n_in_row: u8) -> Vec<Opening> {
    let n = i32::from(board_size);
    let centre = (n - 1) / 2;
    let mut openings = Vec::new();
    for offsets in DEFAULT_OPENINGS {
        let stones: Option<Vec<(u16, Color)>> = offsets
            .iter()
            .enumerate()
            .map(|(index, &(d_row, d_col))| {
                let (row, col) = (centre + d_row, centre + d_col);
                if row < 0 || col < 0 || row >= n || col >= n {
                    return None;
                }
                let color = if index % 2 == 0 {
                    Color::Black
                } else {
                    Color::White
                };
                Some(((row * n + col) as u16, color))
            })
            .collect();
        if let Some(stones) = stones
            && is_usable(&stones, board_size, n_in_row)
        {
            openings.push(Opening { stones });
        }
    }
    openings
}

/// Parse an opening book.
///
/// One opening per line: comma or whitespace separated board indices
/// (`row * board_size + col`), colours alternating from Black, so the count must be
/// even. `#` starts a comment; blank lines are ignored. Unusable lines are reported
/// and skipped, so a partially broken file still yields a usable book.
pub fn parse_openings(text: &str, board_size: u8, n_in_row: u8) -> Vec<Opening> {
    let action_size = u32::from(board_size) * u32::from(board_size);
    let mut openings = Vec::new();
    for (line_index, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        let mut stones = Vec::new();
        let mut malformed = false;
        for (position, token) in line
            .split([',', ' ', '\t'])
            .filter(|token| !token.is_empty())
            .enumerate()
        {
            match token.parse::<u32>() {
                Ok(index) if index < action_size => {
                    let color = if position % 2 == 0 {
                        Color::Black
                    } else {
                        Color::White
                    };
                    stones.push((index as u16, color));
                }
                _ => {
                    malformed = true;
                    break;
                }
            }
        }

        if malformed || !is_usable(&stones, board_size, n_in_row) {
            eprintln!(
                "openings: skipping line {} ({}): need an even count of distinct, \
                 in-range indices that does not already end the game",
                line_index + 1,
                raw.trim()
            );
            continue;
        }
        openings.push(Opening { stones });
    }
    openings
}

/// Render a book in the [`parse_openings`] format (used to seed a new work dir).
pub fn format_book(openings: &[Opening]) -> String {
    let mut text = String::from(
        "# Colour-paired evaluation openings, one per line.\n\
         # Indices are row * board_size + col, colours alternate Black, White, ...\n\
         # so the count must be even (that is what makes the colour-swapped twin of\n\
         # each opening a legal position for Black to move).\n\
         # Each opening is played twice per evaluation, once with each colour, and\n\
         # the first-move advantage cancels inside the resulting pair.\n",
    );
    for opening in openings {
        let line = opening
            .stones()
            .iter()
            .map(|(index, _)| index.to_string())
            .collect::<Vec<_>>()
            .join(",");
        text.push_str(&line);
        text.push('\n');
    }
    text
}

/// Load the book from `file`, falling back to the built-in book when the file is
/// missing, unreadable, or contains no usable opening.
pub fn load_openings(file: &Path, board_size: u8, n_in_row: u8) -> Vec<Opening> {
    match fs::read_to_string(file) {
        Ok(text) => {
            let parsed = parse_openings(&text, board_size, n_in_row);
            if parsed.is_empty() {
                eprintln!(
                    "openings: {} has no usable opening, falling back to the built-in book",
                    file.display()
                );
                default_openings(board_size, n_in_row)
            } else {
                parsed
            }
        }
        Err(_) => {
            let defaults = default_openings(board_size, n_in_row);
            println!(
                "openings: no {}, using the built-in book ({} openings)",
                file.display(),
                defaults.len()
            );
            defaults
        }
    }
}

/// One scheduled evaluation game.
///
/// `a_first` and `opening` must stay consistent: an odd game index plays the
/// colour-swapped opening with net A as White, so the pair shares one opening.
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledGame {
    pub opening: Option<Opening>,
    pub a_first: bool,
}

/// Build the per-game opening schedule.
///
/// With an empty book this reproduces the legacy behaviour exactly (empty board,
/// colours alternating, so only the first-move advantage alternates). With a book,
/// games come in colour-swapped pairs: pair `p` uses opening `p % len` under
/// transform `(p / len) % 8`, and the second game of the pair swaps the colours.
pub fn schedule(openings: &[Opening], game_num: u16, board_size: u8) -> Vec<ScheduledGame> {
    let mut games = Vec::with_capacity(game_num as usize);
    for game_index in 0..game_num as usize {
        let a_first = game_index % 2 == 0;
        let opening = if openings.is_empty() {
            None
        } else {
            let pair = game_index / 2;
            let book_index = pair % openings.len();
            let transform = (pair / openings.len()) % 8;
            let base = openings[book_index].transformed(transform, board_size);
            Some(if a_first { base } else { base.swapped() })
        };
        games.push(ScheduledGame { opening, a_first });
    }
    games
}

/// Number of complete colour-swapped pairs covered by `game_num` games.
pub fn pair_count(game_num: usize) -> usize {
    game_num / 2
}

/// `ln(exp(a) + exp(b))` without overflowing.
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let (high, low) = if a > b { (a, b) } else { (b, a) };
    if high == f64::NEG_INFINITY {
        return high;
    }
    high + (low - high).exp().ln_1p()
}

/// Two-sided exact sign-test p-value: `2 * P(X <= min(k, n - k))` for `X ~ B(n, 1/2)`.
///
/// The probability of a fair coin producing an outcome at least as lopsided as
/// `successes` in `trials` attempts, computed in log space (the naive `2^-n` factor
/// underflows for a few hundred pairs).
pub fn sign_test_p_value(successes: u32, trials: u32) -> f64 {
    if trials == 0 {
        return 1.0;
    }
    let tail = successes.min(trials - successes);
    // term_i = ln(C(trials, i)) - trials * ln 2, built up by the ratio of neighbours
    let mut term = -(f64::from(trials)) * std::f64::consts::LN_2;
    let mut sum = term;
    for i in 1..=tail {
        term += (f64::from(trials - i + 1) / f64::from(i)).ln();
        sum = log_sum_exp(sum, term);
    }
    (2.0 * sum.exp()).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOARD: u8 = 15;
    const ROW: u8 = 5;

    #[test]
    fn default_book_is_pairable_and_loadable() {
        let book = default_openings(BOARD, ROW);
        assert!(!book.is_empty(), "the built-in book must not be empty");
        for opening in &book {
            assert_eq!(opening.plies() % 2, 0, "colour swap needs an even ply count");
            assert!(
                is_usable(opening.stones(), BOARD, ROW),
                "built-in openings must be legal, running positions"
            );
            // the swapped twin must be usable as well: that is the whole point of the
            // even ply count
            assert!(
                is_usable(opening.swapped().stones(), BOARD, ROW),
                "the colour-swapped twin must be legal for Black to move"
            );
            // and it must really have the colours exchanged
            for ((_, color), (_, twin)) in opening.stones().iter().zip(opening.swapped().stones()) {
                assert_eq!(*twin, opposite(*color));
            }
        }
    }

    #[test]
    fn swapped_twice_is_the_original() {
        let opening = default_openings(BOARD, ROW)[0].clone();
        assert_eq!(opening.swapped().swapped(), opening);
    }

    #[test]
    fn transformed_is_a_dihedral_action() {
        // an asymmetric opening: its orbit under the eight board symmetries has size 8
        let n = u16::from(BOARD);
        let centre = (n - 1) / 2 * n + (n - 1) / 2;
        let opening = Opening {
            stones: vec![
                (centre, Color::Black),
                (centre - 1, Color::White),
                (centre - n - 1, Color::Black),
                (centre + n + 2, Color::White),
            ],
        };

        assert_eq!(opening.transformed(0, BOARD), opening, "transform 0 is the identity");

        for transform in 0..8 {
            let moved = opening.transformed(transform, BOARD);
            assert_eq!(moved.plies(), opening.plies());
            // colours stay attached to their stones in play order
            for (index, (_, color)) in moved.stones().iter().enumerate() {
                assert_eq!(*color, opening.stones()[index].1);
            }
            // the board map is a bijection, so distinct points stay distinct
            let mut points: Vec<u16> = moved.stones().iter().map(|&(index, _)| index).collect();
            points.sort_unstable();
            points.dedup();
            assert_eq!(points.len(), opening.plies());
        }

        // four quarter turns are the identity, and the mirrors are involutions
        let four_turns = (0..4).fold(opening.clone(), |acc, _| acc.transformed(1, BOARD));
        assert_eq!(four_turns, opening);
        assert_eq!(opening.transformed(4, BOARD).transformed(4, BOARD), opening);
        assert_eq!(opening.transformed(7, BOARD).transformed(7, BOARD), opening);

        // and the eight symmetries place this opening in eight distinct ways
        let mut placements = std::collections::HashSet::new();
        for transform in 0..8 {
            placements.insert(
                opening
                    .transformed(transform, BOARD)
                    .stones()
                    .iter()
                    .map(|&(index, _)| index)
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(placements.len(), 8);
    }

    #[test]
    fn transformed_is_a_board_symmetry() {
        // rotating the centre point of an odd board gives the centre back
        let centre = u16::from((BOARD - 1) / 2 * BOARD + (BOARD - 1) / 2);
        let opening = Opening {
            stones: vec![(centre, Color::Black), (centre + 1, Color::White)],
        };
        for transform in 0..8 {
            let moved = opening.transformed(transform, BOARD);
            assert_eq!(moved.stones()[0].0, centre);
            assert_eq!(moved.stones()[0].1, Color::Black);
            assert_eq!(moved.stones()[1].1, Color::White);
        }
    }

    #[test]
    fn parse_rejects_malformed_lines_and_keeps_good_ones() {
        let text = "\n\
            # comment line\n\
            112,98\n\
            7\n\
            112,112\n\
            9999,0\n\
            112,98,113,114 # trailing comment\n";
        let parsed = parse_openings(text, BOARD, ROW);
        assert_eq!(parsed.len(), 2, "only the two well-formed openings survive");
        assert_eq!(parsed[0].stones(), &[(112, Color::Black), (98, Color::White)]);
        assert_eq!(parsed[1].plies(), 4);
    }

    #[test]
    fn schedule_pairs_each_opening_with_its_colour_swap() {
        let book = default_openings(BOARD, ROW);
        let games = schedule(&book, 6, BOARD);
        assert_eq!(games.len(), 6);
        for pair in 0..3 {
            let first = &games[pair * 2];
            let second = &games[pair * 2 + 1];
            assert!(first.a_first, "the first game of a pair gives A the Black stones");
            assert!(!second.a_first, "the second game of a pair gives A the White stones");
            let first_opening = first.opening.as_ref().expect("book game has an opening");
            let second_opening = second.opening.as_ref().expect("book game has an opening");
            assert_eq!(
                &first_opening.swapped(),
                second_opening,
                "pair {pair} must be the colour swap of the same opening"
            );
        }
    }

    #[test]
    fn schedule_without_a_book_is_the_legacy_alternation() {
        let games = schedule(&[], 5, BOARD);
        assert_eq!(games.len(), 5);
        for (index, game) in games.iter().enumerate() {
            assert!(game.opening.is_none());
            assert_eq!(game.a_first, index % 2 == 0);
        }
        assert_eq!(pair_count(5), 2);
        assert_eq!(pair_count(6), 3);
    }

    #[test]
    fn format_book_round_trips_through_the_parser() {
        let book = default_openings(BOARD, ROW);
        let rendered = format_book(&book);
        assert_eq!(parse_openings(&rendered, BOARD, ROW), book);
    }

    #[test]
    fn small_boards_keep_whatever_fits() {
        let book = default_openings(3, 3);
        assert!(!book.is_empty());
        for opening in &book {
            assert_eq!(opening.plies() % 2, 0);
            assert!(is_usable(opening.stones(), 3, 3));
        }
    }

    #[test]
    fn decided_positions_are_not_usable() {
        let n = u16::from(BOARD);
        let row = 7u16;
        // filler stones so the move counter passes the engine's minimum before it
        // bothers to look at the last move at all
        let mut stones: Vec<(u16, Color)> = vec![
            (0, Color::Black),
            (1, Color::White),
            (2, Color::Black),
            (3, Color::White),
            (n, Color::Black),
            (n + 1, Color::White),
            (2 * n, Color::Black),
            (2 * n + 1, Color::White),
        ];
        // the final stone completes six in a row: a finished free-style game
        for col in 3..=8u16 {
            stones.push((row * n + col, Color::Black));
        }
        assert_eq!(stones.len() % 2, 0);
        assert!(
            !is_usable(&stones, BOARD, ROW),
            "an opening that already ends the game must be rejected"
        );
    }

    #[test]
    fn odd_ply_counts_are_not_usable() {
        // Black - White - Black leaves the colour-swapped twin unreachable for Black
        let stones = vec![
            (112, Color::Black),
            (111, Color::White),
            (97, Color::Black),
        ];
        assert!(!is_usable(&stones, BOARD, ROW));
    }

    #[test]
    fn sign_test_matches_hand_computed_values() {
        assert_eq!(sign_test_p_value(0, 0), 1.0);
        // 2 * (1 + 2) / 4 is capped at 1
        assert_eq!(sign_test_p_value(1, 2), 1.0);
        // 2 * C(10, 0) / 2^10
        assert!((sign_test_p_value(0, 10) - 2.0 / 1024.0).abs() < 1e-12);
        // 2 * (1 + 4) / 16
        assert!((sign_test_p_value(1, 4) - 0.625).abs() < 1e-12);
        // lopsided in either direction gives the same p-value
        assert_eq!(sign_test_p_value(1, 11), sign_test_p_value(10, 11));
        assert_eq!(sign_test_p_value(5, 10), 1.0);
        // large n must not underflow to a zero-probability shortcut
        assert!(sign_test_p_value(500, 1000) > 0.9);
        assert!(sign_test_p_value(400, 1000) < 1e-9);
    }
}
