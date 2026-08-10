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
        let n = board.len();
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
