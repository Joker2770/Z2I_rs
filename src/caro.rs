// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use crate::rule::{Board, Color, RuleOpt};

const WIN_SHAPES: &[[i8; 7]] = &[
    [1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 2],
    [1, 1, 1, 1, 1, 0, 3],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1],
    [2, 0, 1, 1, 1, 1, 1],
    [3, 0, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 0],
    [2, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0],
    [3, 1, 1, 1, 1, 1, 0],
    [3, 1, 1, 1, 1, 1, 1],
    [3, 1, 1, 1, 1, 1, 2],
    [3, 1, 1, 1, 1, 1, 3],
    [0, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 3],
    [2, 1, 1, 1, 1, 1, 3],
    [2, 2, 2, 2, 2, 0, 0],
    [2, 2, 2, 2, 2, 0, 1],
    [2, 2, 2, 2, 2, 0, 2],
    [2, 2, 2, 2, 2, 0, 3],
    [0, 2, 2, 2, 2, 2, 0],
    [0, 0, 2, 2, 2, 2, 2],
    [1, 0, 2, 2, 2, 2, 2],
    [2, 0, 2, 2, 2, 2, 2],
    [3, 0, 2, 2, 2, 2, 2],
    [1, 2, 2, 2, 2, 2, 0],
    [1, 2, 2, 2, 2, 2, 2],
    [0, 2, 2, 2, 2, 2, 1],
    [2, 2, 2, 2, 2, 2, 1],
    [2, 2, 2, 2, 2, 2, 2],
    [0, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 0],
    [3, 2, 2, 2, 2, 2, 0],
    [3, 2, 2, 2, 2, 2, 1],
    [3, 2, 2, 2, 2, 2, 2],
    [3, 2, 2, 2, 2, 2, 3],
    [0, 2, 2, 2, 2, 2, 3],
    [1, 2, 2, 2, 2, 2, 3],
    [2, 2, 2, 2, 2, 2, 3],
];

#[derive(Clone, Copy)]
pub struct CaroJudge;

impl CaroJudge {
    pub fn new() -> Self {
        Self
    }

    fn is_pos_out_of_board(n: usize, x: isize, y: isize) -> bool {
        x < 0 || y < 0 || (x as usize) > n - 1 || (y as usize) > n - 1
    }

    fn find_shape(board: &Board, last_move: i16, p_drt: (isize, isize)) -> bool {
        let n = board.len();
        let idx = last_move as usize;
        let row = idx / n;
        let col = idx % n;
        let mut v_color: Vec<i8> = Vec::new();

        let cur = board[row][col];
        if cur == Color::Black {
            v_color.push(1);
        } else if cur == Color::White {
            v_color.push(2);
        } else {
            return false;
        }

        // forward
        let mut p_x = row as isize + p_drt.0;
        let mut p_y = col as isize + p_drt.1;
        loop {
            if Self::is_pos_out_of_board(n, p_x, p_y) {
                v_color.push(3);
            } else {
                let v = board[p_x as usize][p_y as usize];
                if v == Color::Black {
                    v_color.push(1);
                } else if v == Color::White {
                    v_color.push(2);
                } else {
                    v_color.push(0);
                }
            }
            p_x += p_drt.0;
            p_y += p_drt.1;
            if (row as isize - p_x).abs() > 5 || (col as isize - p_y).abs() > 5 {
                break;
            }
        }

        v_color.reverse();
        p_x = row as isize - p_drt.0;
        p_y = col as isize - p_drt.1;
        loop {
            if Self::is_pos_out_of_board(n, p_x, p_y) {
                v_color.push(3);
            } else {
                let v = board[p_x as usize][p_y as usize];
                if v == Color::Black {
                    v_color.push(1);
                } else if v == Color::White {
                    v_color.push(2);
                } else {
                    v_color.push(0);
                }
            }
            p_x -= p_drt.0;
            p_y -= p_drt.1;
            if (row as isize - p_x).abs() > 5 || (col as isize - p_y).abs() > 5 {
                break;
            }
        }

        if v_color.len() >= 7 {
            for j in 0..=v_color.len() - 7 {
                'shapes: for shape in WIN_SHAPES.iter() {
                    for k in 0..7 {
                        if shape[k] != v_color[j + k] {
                            continue 'shapes;
                        }
                    }
                    return true;
                }
            }
        }

        false
    }

    pub fn check_win(&self, board: &Board, last_move: i16) -> bool {
        if last_move < 0 {
            return false;
        }
        // let n = board.len();
        let dirs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)];
        for d in dirs.iter() {
            if Self::find_shape(board, last_move, *d) {
                return true;
            }
        }
        false
    }
}

impl RuleOpt for CaroJudge {
    fn check_win(&mut self, board: &Board, last_move: i16) -> bool {
        CaroJudge::check_win(self, board, last_move)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 15;

    fn idx(row: usize, col: usize) -> usize {
        row * N + col
    }

    fn board_with(stones: &[(usize, Color)]) -> Board {
        let mut board = vec![vec![Color::Blank; N]; N];
        for &(i, c) in stones {
            board[i / N][i % N] = c;
        }
        board
    }

    fn black_row(row: usize, cols: std::ops::RangeInclusive<usize>) -> Vec<(usize, Color)> {
        cols.map(|c| (idx(row, c), Color::Black)).collect()
    }

    #[test]
    fn win_horizontal_five_black_last_at_end() {
        // x x x x x, last move is the rightmost stone
        let row = 7;
        let stones = black_row(row, 3..=7);
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(row, 7) as i16));
    }

    #[test]
    fn win_horizontal_five_black_last_at_other_end() {
        // x x x x x, last move is the leftmost stone
        let row = 7;
        let stones = black_row(row, 3..=7);
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(row, 3) as i16));
    }

    #[test]
    fn win_horizontal_five_last_in_middle() {
        // x x x x x, last move is the middle stone
        let row = 7;
        let stones = black_row(row, 3..=7);
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(row, 5) as i16));
    }

    #[test]
    fn win_vertical_five_white() {
        // o o o o o in a column
        let col = 7;
        let stones: Vec<(usize, Color)> =
            (3..=7).map(|r| (idx(r, col), Color::White)).collect();
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(7, col) as i16));
    }

    #[test]
    fn win_main_diagonal_five_black() {
        let stones: Vec<(usize, Color)> =
            (0..5).map(|d| (idx(3 + d, 3 + d), Color::Black)).collect();
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(7, 7) as i16));
    }

    #[test]
    fn win_anti_diagonal_five_black() {
        let stones: Vec<(usize, Color)> =
            (0..5).map(|d| (idx(3 + d, 11 - d), Color::Black)).collect();
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(7, 7) as i16));
    }

    #[test]
    fn win_six_in_a_row() {
        // overline still wins under caro rule
        let row = 7;
        let stones = black_row(row, 3..=8);
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(row, 8) as i16));
    }

    #[test]
    fn win_blocked_at_one_end() {
        // o x x x x x, white blocks one end, five still wins
        let row = 7;
        let mut stones = black_row(row, 4..=8);
        stones.push((idx(row, 3), Color::White));
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(row, 6) as i16));
    }

    #[test]
    fn win_at_top_edge() {
        // five along the top edge, border acts as '3'
        let stones = black_row(0, 0..=4);
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(0, 4) as i16));
    }

    #[test]
    fn win_at_corner_diagonal() {
        let stones: Vec<(usize, Color)> = (0..5).map(|d| (idx(d, d), Color::Black)).collect();
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(judge.check_win(&board, idx(4, 4) as i16));
    }

    #[test]
    fn four_in_a_row_is_not_win() {
        let row = 7;
        let stones = black_row(row, 3..=6);
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(!judge.check_win(&board, idx(row, 6) as i16));
    }

    #[test]
    fn broken_four_is_not_win() {
        // x x x _ x, gap breaks the line
        let row = 7;
        let stones = vec![idx(row, 3), idx(row, 4), idx(row, 5), idx(row, 7)]
            .into_iter()
            .map(|i| (i, Color::Black))
            .collect::<Vec<_>>();
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(!judge.check_win(&board, idx(row, 7) as i16));
    }

    #[test]
    fn interrupted_line_is_not_win() {
        // x x o x x, opponent stone splits the line
        let row = 7;
        let mut stones = vec![idx(row, 3), idx(row, 4), idx(row, 6), idx(row, 7)]
            .into_iter()
            .map(|i| (i, Color::Black))
            .collect::<Vec<_>>();
        stones.push((idx(row, 5), Color::White));
        let board = board_with(&stones);
        let judge = CaroJudge::new();
        assert!(!judge.check_win(&board, idx(row, 7) as i16));
    }

    #[test]
    fn negative_last_move_is_not_win() {
        let board = board_with(&[]);
        let judge = CaroJudge::new();
        assert!(!judge.check_win(&board, -1));
        assert!(!judge.check_win(&board, -2));
    }

    #[test]
    fn blank_last_move_is_not_win() {
        let board = board_with(&[]);
        let judge = CaroJudge::new();
        assert!(!judge.check_win(&board, 0));
    }

    #[test]
    fn rule_opt_trait_check_win() {
        // exercise the RuleOpt implementation explicitly
        let row = 7;
        let stones = black_row(row, 3..=7);
        let board = board_with(&stones);
        let mut judge = CaroJudge::new();
        assert!(RuleOpt::check_win(&mut judge, &board, idx(row, 7) as i16));
    }
}
