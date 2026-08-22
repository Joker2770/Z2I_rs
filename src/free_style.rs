// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use crate::rule::{Board, Color, RuleOpt};

#[derive(Clone, Copy)]
pub struct FreeStyleJudge;

impl FreeStyleJudge {
    pub fn new() -> Self {
        FreeStyleJudge
    }

    fn is_pos_out_of_board(n: usize, x: isize, y: isize) -> bool {
        x < 0 || y < 0 || (x as usize) > n - 1 || (y as usize) > n - 1
    }

    fn count_near_stone(board: &Board, last_move: i16, p_drt: (isize, isize)) -> i32 {
        if last_move == -1 {
            return 0;
        }
        let n = board.len();
        let idx = last_move as usize;
        let row = (idx / n) as isize;
        let col = (idx % n) as isize;
        let mut count = 0i32;
        let mut p_x = row + p_drt.0;
        let mut p_y = col + p_drt.1;
        while !Self::is_pos_out_of_board(n, p_x, p_y) {
            let v = board[p_x as usize][p_y as usize];
            if v == Color::Blank {
                break;
            }
            if v == board[row as usize][col as usize] {
                count += 1;
            } else {
                return count;
            }
            p_x += p_drt.0;
            p_y += p_drt.1;
            if (row - p_x).abs() > 5 || (col - p_y).abs() > 5 {
                break;
            }
        }
        count
    }

    pub fn check_win(&self, board: &Board, last_move: i16) -> bool {
        if last_move < 0 {
            return last_move == -1;
        }
        let n = board.len();
        if n == 0 {
            return false;
        }
        let dirs = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
            (-1, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
        ];
        let i_up = Self::count_near_stone(board, last_move, dirs[0]);
        let i_down = Self::count_near_stone(board, last_move, dirs[1]);
        let i_left = Self::count_near_stone(board, last_move, dirs[2]);
        let i_right = Self::count_near_stone(board, last_move, dirs[3]);
        let i_leftup = Self::count_near_stone(board, last_move, dirs[4]);
        let i_rightdown = Self::count_near_stone(board, last_move, dirs[5]);
        let i_leftdown = Self::count_near_stone(board, last_move, dirs[7]);
        let i_rightup = Self::count_near_stone(board, last_move, dirs[6]);

        if i_up + i_down >= 4
            || i_left + i_right >= 4
            || i_leftup + i_rightdown >= 4
            || i_leftdown + i_rightup >= 4
        {
            return true;
        }
        false
    }
}

impl RuleOpt for FreeStyleJudge {
    fn check_win(&mut self, board: &Board, last_move: i16) -> bool {
        FreeStyleJudge::check_win(self, board, last_move)
    }
}
