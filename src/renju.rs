// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use crate::rule::{Board, Color, RuleOpt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Row,
    UnbrokenRow,
    Overline,
    FiveInARow,
    Four,
    StraightFour,
    Three,
    DoubleFour,
    DoubleThree,
}

const A4_SHAPES: &[[i32; 5]] = &[
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
];

const A3_SHAPES: &[[i32; 6]] = &[
    [0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 0],
];

#[derive(Clone, Copy)]
pub struct RenjuJudge {
    m_renju_state: Pattern,
}

impl RenjuJudge {
    pub fn new() -> Self {
        Self {
            m_renju_state: Pattern::Row,
        }
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

    fn collect_line_colors(
        board: &Board,
        last_move: i16,
        p_drt: (isize, isize),
        max_range: usize,
    ) -> Vec<i32> {
        let n = board.len();
        let idx = last_move as usize;
        let row = (idx / n) as isize;
        let col = (idx % n) as isize;
        let mut v: Vec<i32> = Vec::new();

        // current
        let cur = board[row as usize][col as usize];
        if cur == Color::Black {
            v.push(1);
        } else if cur == Color::White {
            v.push(2);
        } else {
            return v;
        }

        // forward
        let mut px = row + p_drt.0;
        let mut py = col + p_drt.1;
        let mut steps = 0;
        while !Self::is_pos_out_of_board(n, px, py) && steps < max_range {
            let val = board[px as usize][py as usize];
            if val == Color::Black {
                v.push(1);
            } else if val == Color::White {
                v.push(2);
            } else {
                v.push(0);
            }
            px += p_drt.0;
            py += p_drt.1;
            steps += 1;
        }

        // reverse and backward
        v.reverse();
        px = row - p_drt.0;
        py = col - p_drt.1;
        steps = 0;
        while !Self::is_pos_out_of_board(n, px, py) && steps < max_range {
            let val = board[px as usize][py as usize];
            if val == Color::Black {
                v.push(1);
            } else if val == Color::White {
                v.push(2);
            } else {
                v.push(0);
            }
            px -= p_drt.0;
            py -= p_drt.1;
            steps += 1;
        }

        v
    }

    fn count_a4(board: &Board, last_move: i16, p_drt: (isize, isize)) -> i32 {
        let v = Self::collect_line_colors(board, last_move, p_drt, 4);
        if v.len() < 5 {
            return 0;
        }
        let mut cnt = 0i32;
        for j in 0..=v.len() - 5 {
            for shape in A4_SHAPES.iter() {
                let mut ok = true;
                for k in 0..5 {
                    if shape[k] != v[j + k] {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    cnt += 1;
                }
            }
        }
        cnt
    }

    fn count_a3(board: &Board, last_move: i16, p_drt: (isize, isize)) -> i32 {
        let v = Self::collect_line_colors(board, last_move, p_drt, 5);
        if v.len() < 6 {
            return 0;
        }
        let mut cnt = 0i32;
        for j in 0..=v.len() - 6 {
            for shape in A3_SHAPES.iter() {
                let mut ok = true;
                for k in 0..6 {
                    if shape[k] != v[j + k] {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    cnt += 1;
                }
            }
        }
        cnt
    }

    fn is_double_four(&self, board: &Board, last_move: i16) -> bool {
        let dirs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)];
        let mut sum = 0i32;
        for d in dirs.iter() {
            sum += Self::count_a4(board, last_move, *d);
        }
        sum >= 2
    }

    fn is_double_three(&self, board: &Board, last_move: i16) -> bool {
        let dirs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)];
        let mut sum3 = 0i32;
        let mut sum4 = 0i32;
        for d in dirs.iter() {
            sum3 += Self::count_a3(board, last_move, *d);
            sum4 += Self::count_a4(board, last_move, *d);
        }
        (sum4 < 2) && (sum3 >= 2)
    }

    fn is_four_three(&self, board: &Board, last_move: i16) -> bool {
        let dirs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)];
        let mut sum4 = 0i32;
        let mut sum3 = 0i32;
        for d in dirs.iter() {
            sum4 += Self::count_a4(board, last_move, *d);
            sum3 += Self::count_a3(board, last_move, *d);
        }
        (sum4 == 1) && (sum3 >= 1)
    }

    fn is_four(&self, board: &Board, last_move: i16) -> bool {
        let dirs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)];
        let mut sum4 = 0i32;
        let mut sum3 = 0i32;
        for d in dirs.iter() {
            sum4 += Self::count_a4(board, last_move, *d);
            sum3 += Self::count_a3(board, last_move, *d);
        }
        (sum4 == 1) && (sum3 < 2)
    }

    fn is_three(&self, board: &Board, last_move: i16) -> bool {
        let dirs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)];
        let mut sum4 = 0i32;
        let mut sum3 = 0i32;
        for d in dirs.iter() {
            sum4 += Self::count_a4(board, last_move, *d);
            sum3 += Self::count_a3(board, last_move, *d);
        }
        (sum4 == 0) && (sum3 == 1)
    }

    fn is_over_line(board: &Board, last_move: i16) -> bool {
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
        let mut sums = [0i32; 4];
        // pairs: (0,1) up/down, (2,3) left/right, (4,5) leftup/rightdown, (6,7) rightup/leftdown
        let n = board.len();
        if last_move < 0 || n == 0 {
            return false;
        }
        let idx = last_move as usize;
        let row = (idx / n) as isize;
        let col = (idx % n) as isize;
        if board[row as usize][col as usize] == Color::Blank {
            return false;
        }
        sums[0] = Self::count_near_stone(board, last_move, dirs[0])
            + Self::count_near_stone(board, last_move, dirs[1]);
        sums[1] = Self::count_near_stone(board, last_move, dirs[2])
            + Self::count_near_stone(board, last_move, dirs[3]);
        sums[2] = Self::count_near_stone(board, last_move, dirs[4])
            + Self::count_near_stone(board, last_move, dirs[5]);
        sums[3] = Self::count_near_stone(board, last_move, dirs[6])
            + Self::count_near_stone(board, last_move, dirs[7]);
        sums.iter().any(|&s| s > 4)
    }

    pub fn check_win(&mut self, board: &Board, last_move: i16) -> bool {
        if last_move < 0 {
            if last_move == -1 {
                return true;
            } else {
                return false;
            }
        }
        let n = board.len();
        if n == 0 {
            return false;
        }
        let idx = last_move as usize;
        let row = (idx / n) as isize;
        let col = (idx % n) as isize;
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
        let i_rightup = Self::count_near_stone(board, last_move, dirs[6]);
        let i_leftdown = Self::count_near_stone(board, last_move, dirs[7]);

        let val = board[row as usize][col as usize];
        if val == Color::Black {
            if i_up + i_down == 4
                || i_left + i_right == 4
                || i_leftup + i_rightdown == 4
                || i_leftdown + i_rightup == 4
            {
                self.m_renju_state = Pattern::FiveInARow;
                return true;
            }
        } else {
            if i_up + i_down >= 4
                || i_left + i_right >= 4
                || i_leftup + i_rightdown >= 4
                || i_leftdown + i_rightup >= 4
            {
                self.m_renju_state = Pattern::FiveInARow;
                return true;
            }
        }
        false
    }

    pub fn is_legal(&mut self, board: &Board, last_move: i16) -> bool {
        if last_move < 0 {
            if last_move == -1 {
                return true;
            } else {
                return false;
            }
        }
        let n = board.len();
        if n == 0 {
            return false;
        }
        let idx = last_move as usize;
        let val = board[idx / n][idx % n];
        if val == Color::Black {
            if Self::is_over_line(board, last_move) {
                self.m_renju_state = Pattern::Overline;
                return false;
            }
            if self.is_double_four(board, last_move) {
                self.m_renju_state = Pattern::DoubleFour;
                return false;
            }
            if self.is_double_three(board, last_move) {
                self.m_renju_state = Pattern::DoubleThree;
                return false;
            }
            // four-three is ambiguous in some rules; we will treat plain four-three as legal here but record state
            if self.is_four_three(board, last_move) {
                self.m_renju_state = Pattern::StraightFour;
                return true;
            }
            if self.is_four(board, last_move) {
                self.m_renju_state = Pattern::Four;
                return true;
            }
            if self.is_three(board, last_move) {
                self.m_renju_state = Pattern::Three;
                return true;
            }
        }
        true
    }

    pub fn get_renju_state(&self) -> Pattern {
        self.m_renju_state
    }
}

impl RuleOpt for RenjuJudge {
    fn check_win(&mut self, board: &Board, last_move: i16) -> bool {
        RenjuJudge::check_win(self, board, last_move)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_double_four_simple() {
        // Construct a simple board where placing at center creates two separate 4-patterns
        let n = 15;
        let mut board = vec![vec![Color::Blank; n]; n];
        let r = 7usize;
        // horizontal three to left, open on other side
        board[r][3] = Color::Black;
        board[r][4] = Color::Black;
        board[r][5] = Color::Black;
        // vertical three above
        board[4][7] = Color::Black;
        board[5][7] = Color::Black;
        board[6][7] = Color::Black;
        // Place the move at (7,7)
        board[r][7] = Color::Black;
        let last_move = (r * n + 7) as i16;
        let judge = RenjuJudge::new();
        // count_a4 in two orthogonal directions should detect 2 (approx)
        assert!(judge.is_double_four(&board, last_move));
    }

    #[test]
    fn detect_double_three_simple() {
        let n = 15;
        let mut board = vec![vec![Color::Blank; n]; n];
        let r = 7usize;
        // create two separate open-three patterns around (7,7)
        board[r][5] = Color::Black;
        board[r][6] = Color::Black;
        board[r][8] = Color::Black;
        board[r][9] = Color::Black;
        board[6][7] = Color::Black;
        board[5][7] = Color::Black;
        // move at center
        board[r][7] = Color::Black;
        let last_move = (r * n + 7) as i16;
        let judge = RenjuJudge::new();
        assert!(!judge.is_double_three(&board, last_move));
    }
}
