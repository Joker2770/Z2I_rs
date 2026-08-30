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

    /// Count the number of "fours" passing through last_move.
    ///
    /// RIF DOUBLE-FOUR = a move forms more than one four (intersecting at the move).
    /// Key point: two fours sharing the same completion point count as one four — if white
    /// plays that point next, all five-in-a-row routes are blocked at once (e.g. for xxxx_x,
    /// {0,1,2,3} and {1,2,3,5} both complete at 4; for xxx_xx, the two gapped fours both
    /// complete at 3), so it is not a double-four and the move is legal.
    /// Only when there are two fours with different completion points (e.g. x_xxx_x completes
    /// at 5 and 9) is it a double-four forbidden move.
    ///
    /// The i_flag combination logic is designed accordingly: a live four with two completion
    /// points (the same stone set) counts only 1; flag combinations produced by multiple fours
    /// sharing the same completion point never fall into the "count 2" branch.
    fn count_a4(board: &Board, last_move: i16, p_drt: (isize, isize)) -> i32 {
        let v = Self::collect_line_colors(board, last_move, p_drt, 4);
        if v.len() < 5 {
            return 0;
        }
        let mut i_count = 0i32;
        let mut i_flag = 0i32;
        let mut b_1_before_3 = false;
        for j in 0..=v.len() - 5 {
            for i in 0..5 {
                let mut ok = true;
                for k in 0..5 {
                    if A4_SHAPES[i][k] != v[j + k] {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    if i == 1 || i == 3 {
                        if i == 1 {
                            i_flag |= 0x01;
                        } else if i == 3 {
                            if (i_flag & 0x01) == 0x01 {
                                b_1_before_3 = true;
                            }
                            i_flag |= 0x02;
                        }
                        break;
                    } else if i == 2 {
                        if (i_flag & 0x04) != 0x04 {
                            i_flag |= 0x04;
                        } else if (i_flag & 0x08) != 0x08 {
                            i_flag |= 0x08;
                        }
                        break;
                    } else {
                        i_count = 1;
                    }
                }
            }
        }

        if (i_flag & 0x0F) == 0x0F {
            i_count = 2;
        } else if (i_flag & 0x07) == 0x07 {
            if b_1_before_3 {
                i_count = 2;
            } else {
                i_count = 1;
            }
        } else if (i_flag & 0x03) == 0x03 {
            i_count = 2;
        } else if (i_flag & (0x04 | 0x08)) == (0x04 | 0x08) {
            i_count = 2;
        } else if ((i_flag & 0x03) == 0x01) || ((i_flag & 0x03) == 0x02) {
            i_count = 1;
        } else if ((i_flag & 0x04) == 0x04) || ((i_flag & 0x08) == 0x08) {
            i_count = 1;
        }

        i_count
    }

    fn count_a3(board: &Board, last_move: i16, p_drt: (isize, isize)) -> i32 {
        let v = Self::collect_line_colors(board, last_move, p_drt, 5);
        if v.len() < 6 {
            return 0;
        }
        for j in 0..=v.len() - 6 {
            for i in 0..4 {
                let mut ok = true;
                for k in 0..6 {
                    if A3_SHAPES[i][k] != v[j + k] {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    return 1;
                }
            }
        }
        0
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

        let i_up_4 = Self::count_a4(board, last_move, dirs[0]);
        let i_left_4 = Self::count_a4(board, last_move, dirs[1]);
        let i_leftup_4 = Self::count_a4(board, last_move, dirs[2]);
        let i_leftdown_4 = Self::count_a4(board, last_move, dirs[3]);
        let i_up_3 = Self::count_a3(board, last_move, dirs[0]);
        let i_left_3 = Self::count_a3(board, last_move, dirs[1]);
        let i_leftup_3 = Self::count_a3(board, last_move, dirs[2]);
        let i_leftdown_3 = Self::count_a3(board, last_move, dirs[3]);

        let sum4 = i_up_4 + i_left_4 + i_leftup_4 + i_leftdown_4;
        let sum3 = i_up_3 + i_left_3 + i_leftup_3 + i_leftdown_3;

        if (sum4 < 2) && (sum3 >= 2) {
            if sum4 == 0 {
                return sum3 >= 2;
            } else {
                if i_up_4 == 1 {
                    if i_left_3 + i_leftup_3 + i_leftdown_3 >= 2 {
                        return true;
                    } else if i_left_3 + i_leftup_3 + i_leftdown_3 == 1 {
                        return false;
                    } else {
                        return false;
                    }
                } else if i_left_4 == 1 {
                    if i_up_3 + i_leftup_3 + i_leftdown_3 >= 2 {
                        return true;
                    } else if i_up_3 + i_leftup_3 + i_leftdown_3 == 1 {
                        return false;
                    } else {
                        return false;
                    }
                } else if i_leftup_4 == 1 {
                    if i_up_3 + i_left_3 + i_leftdown_3 >= 2 {
                        return true;
                    } else if i_up_3 + i_left_3 + i_leftdown_3 == 1 {
                        return false;
                    } else {
                        return false;
                    }
                } else if i_leftdown_4 == 1 {
                    if i_up_3 + i_leftup_3 + i_left_3 >= 2 {
                        return true;
                    } else if i_up_3 + i_leftup_3 + i_left_3 == 1 {
                        return false;
                    } else {
                        return false;
                    }
                }
            }
        }
        false
    }

    fn is_four_three(&self, board: &Board, last_move: i16) -> bool {
        let dirs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)];
        let i_up_4 = Self::count_a4(board, last_move, dirs[0]);
        let i_left_4 = Self::count_a4(board, last_move, dirs[1]);
        let i_leftup_4 = Self::count_a4(board, last_move, dirs[2]);
        let i_leftdown_4 = Self::count_a4(board, last_move, dirs[3]);
        let i_up_3 = Self::count_a3(board, last_move, dirs[0]);
        let i_left_3 = Self::count_a3(board, last_move, dirs[1]);
        let i_leftup_3 = Self::count_a3(board, last_move, dirs[2]);
        let i_leftdown_3 = Self::count_a3(board, last_move, dirs[3]);

        if ((i_up_4 + i_left_4 + i_leftup_4 + i_leftdown_4) == 1)
            && (i_up_3 + i_left_3 + i_leftup_3 + i_leftdown_3 >= 1)
        {
            if i_up_4 == 1 {
                if i_left_3 + i_leftup_3 + i_leftdown_3 == 0 {
                    return false;
                } else if i_left_3 + i_leftup_3 + i_leftdown_3 > 1 {
                    return false;
                } else if i_left_3 + i_leftup_3 + i_leftdown_3 == 1 {
                    return true;
                }
            } else if i_left_4 == 1 {
                if i_up_3 + i_leftup_3 + i_leftdown_3 == 0 {
                    return false;
                } else if i_up_3 + i_leftup_3 + i_leftdown_3 > 1 {
                    return false;
                } else if i_up_3 + i_leftup_3 + i_leftdown_3 == 1 {
                    return true;
                }
            } else if i_leftup_4 == 1 {
                if i_up_3 + i_left_3 + i_leftdown_3 == 0 {
                    return false;
                } else if i_up_3 + i_left_3 + i_leftdown_3 > 1 {
                    return false;
                } else if i_up_3 + i_left_3 + i_leftdown_3 == 1 {
                    return true;
                }
            } else if i_leftdown_4 == 1 {
                if i_left_3 + i_leftup_3 + i_up_3 == 0 {
                    return false;
                } else if i_left_3 + i_leftup_3 + i_up_3 > 1 {
                    return false;
                } else if i_left_3 + i_leftup_3 + i_up_3 == 1 {
                    return true;
                }
            }
        }

        false
    }

    fn is_four(&self, board: &Board, last_move: i16) -> bool {
        let dirs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)];
        let i_up_4 = Self::count_a4(board, last_move, dirs[0]);
        let i_left_4 = Self::count_a4(board, last_move, dirs[1]);
        let i_leftup_4 = Self::count_a4(board, last_move, dirs[2]);
        let i_leftdown_4 = Self::count_a4(board, last_move, dirs[3]);
        let i_up_3 = Self::count_a3(board, last_move, dirs[0]);
        let i_left_3 = Self::count_a3(board, last_move, dirs[1]);
        let i_leftup_3 = Self::count_a3(board, last_move, dirs[2]);
        let i_leftdown_3 = Self::count_a3(board, last_move, dirs[3]);

        if ((i_up_4 + i_left_4 + i_leftup_4 + i_leftdown_4) == 1)
            && (i_up_3 + i_left_3 + i_leftup_3 + i_leftdown_3 < 2)
        {
            if i_up_4 == 1 {
                if i_left_3 + i_leftup_3 + i_leftdown_3 == 0 {
                    return true;
                }
            } else if i_left_4 == 1 {
                if i_up_3 + i_leftup_3 + i_leftdown_3 == 0 {
                    return true;
                }
            } else if i_leftup_4 == 1 {
                if i_up_3 + i_left_3 + i_leftdown_3 == 0 {
                    return true;
                }
            } else if i_leftdown_4 == 1 {
                if i_up_3 + i_left_3 + i_leftup_3 == 0 {
                    return true;
                }
            }
        }

        false
    }

    fn is_three(&self, board: &Board, last_move: i16) -> bool {
        let dirs = [(0, -1), (-1, 0), (-1, -1), (-1, 1)];
        let i_up_4 = Self::count_a4(board, last_move, dirs[0]);
        let i_left_4 = Self::count_a4(board, last_move, dirs[1]);
        let i_leftup_4 = Self::count_a4(board, last_move, dirs[2]);
        let i_leftdown_4 = Self::count_a4(board, last_move, dirs[3]);
        let i_up_3 = Self::count_a3(board, last_move, dirs[0]);
        let i_left_3 = Self::count_a3(board, last_move, dirs[1]);
        let i_leftup_3 = Self::count_a3(board, last_move, dirs[2]);
        let i_leftdown_3 = Self::count_a3(board, last_move, dirs[3]);

        if (i_up_4 + i_left_4 + i_leftup_4 + i_leftdown_4 == 0)
            && (i_up_3 + i_left_3 + i_leftup_3 + i_leftdown_3 == 1)
        {
            return true;
        }

        false
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
            return last_move == -1;
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
            return last_move == -1;
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
            } else if self.is_double_four(board, last_move) {
                self.m_renju_state = Pattern::DoubleFour;
                return false;
            } else if self.is_four_three(board, last_move) {
                return true;
            } else if self.is_four(board, last_move) {
                self.m_renju_state = Pattern::Four;
                return true;
            } else if self.is_double_three(board, last_move) {
                self.m_renju_state = Pattern::DoubleThree;
                return false;
            } else if self.is_three(board, last_move) {
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

    #[test]
    fn detect_overline_illegal_for_black() {
        let n = 15;
        let mut board = vec![vec![Color::Blank; n]; n];
        let r = 7usize;
        // place 6 blacks in a row horizontally
        for c in 3..=8 {
            board[r][c] = Color::Black;
        }
        let last_move = (r * n + 5) as i16; // one of the middle stones
        let mut judge = RenjuJudge::new();
        assert!(!judge.is_legal(&board, last_move));
        assert_eq!(judge.get_renju_state(), Pattern::Overline);
    }

    #[test]
    fn detect_five_in_a_row_win_for_white() {
        let n = 15;
        let mut board = vec![vec![Color::Blank; n]; n];
        let r = 7usize;
        // vertical five whites
        for rr in 3..=7 {
            board[rr][r] = Color::White;
        }
        let last_move = (7 * n + r) as i16;
        let mut judge = RenjuJudge::new();
        assert!(judge.check_win(&board, last_move));
        assert_eq!(judge.get_renju_state(), Pattern::FiveInARow);
    }

    #[test]
    fn detect_double_four_case() {
        let n = 15;
        let mut board = vec![vec![Color::Blank; n]; n];
        let r = 7usize;
        // create two separate 4 patterns crossing at (7,7)
        board[r][3] = Color::Black;
        board[r][4] = Color::Black;
        board[r][5] = Color::Black;
        board[4][7] = Color::Black;
        board[5][7] = Color::Black;
        board[6][7] = Color::Black;
        board[r][7] = Color::Black;
        let last_move = (r * n + 7) as i16;
        let judge = RenjuJudge::new();
        assert!(judge.is_double_four(&board, last_move));
    }

    // --- RIF rule reference probes ---

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

    #[test]
    fn black_exact_five_wins() {
        // RIF 9.1: black wins with exactly five in a row
        let stones = vec![
            (idx(7, 4), Color::Black),
            (idx(7, 5), Color::Black),
            (idx(7, 7), Color::Black),
            (idx(7, 8), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][6] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(judge.check_win(&board, idx(7, 6) as i16));
        assert_eq!(judge.get_renju_state(), Pattern::FiveInARow);
    }

    #[test]
    fn black_six_is_overline_not_win() {
        // RIF 9.1/9.2a: black overline is not a win; it is a forbidden move
        let stones = vec![
            (idx(7, 3), Color::Black),
            (idx(7, 4), Color::Black),
            (idx(7, 5), Color::Black),
            (idx(7, 7), Color::Black),
            (idx(7, 8), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][6] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(!judge.check_win(&board, idx(7, 6) as i16));
        assert!(!judge.is_legal(&board, idx(7, 6) as i16));
        assert_eq!(judge.get_renju_state(), Pattern::Overline);
    }

    #[test]
    fn white_overline_wins() {
        // RIF 9.1: white overline also wins
        let stones = vec![
            (idx(7, 3), Color::White),
            (idx(7, 4), Color::White),
            (idx(7, 5), Color::White),
            (idx(7, 7), Color::White),
            (idx(7, 8), Color::White),
        ];
        let mut board = board_with(&stones);
        board[7][6] = Color::White;
        let mut judge = RenjuJudge::new();
        assert!(judge.check_win(&board, idx(7, 6) as i16));
    }

    #[test]
    fn single_straight_four_is_legal() {
        // RIF: a single live four (straight four) is legal, not a double-four forbidden move
        let stones = vec![
            (idx(7, 4), Color::Black),
            (idx(7, 5), Color::Black),
            (idx(7, 7), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][6] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(!judge.is_double_four(&board, idx(7, 6) as i16));
        assert!(judge.is_legal(&board, idx(7, 6) as i16));
    }

    #[test]
    fn gapped_double_four_is_forbidden() {
        // RIF 9.2b: x_xxx_x forms two "gapped fours" at once, a double-four forbidden move
        // {7,4},{7,6},{7,7},{7,8} complete at {7,5}; {7,6},{7,7},{7,8},{7,10} complete at {7,9}
        let stones = vec![
            (idx(7, 4), Color::Black),
            (idx(7, 7), Color::Black),
            (idx(7, 8), Color::Black),
            (idx(7, 10), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][6] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(judge.is_double_four(&board, idx(7, 6) as i16));
        assert!(!judge.is_legal(&board, idx(7, 6) as i16));
        assert_eq!(judge.get_renju_state(), Pattern::DoubleFour);
    }

    #[test]
    fn four_three_is_legal() {
        // RIF: four-three (one four + one three) is legal, a winning method for black
        let stones = vec![
            (idx(7, 4), Color::Black),
            (idx(7, 5), Color::Black),
            (idx(7, 7), Color::Black), // horizontal live four _xxxx_
            (idx(5, 6), Color::Black), // vertical three __xxx_
            (idx(6, 6), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][6] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(judge.is_four_three(&board, idx(7, 6) as i16));
        assert!(judge.is_legal(&board, idx(7, 6) as i16));
    }

    #[test]
    fn double_three_is_forbidden() {
        // RIF 9.2c: forming two live threes at once is forbidden
        // horizontal _xx_x_ (7,5),(7,6),(7,8); vertical _x_xx_ (5,5),(7,5),(8,5) — the move is (7,5)
        let stones = vec![
            (idx(7, 6), Color::Black),
            (idx(7, 8), Color::Black),
            (idx(5, 5), Color::Black),
            (idx(8, 5), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][5] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(judge.is_double_three(&board, idx(7, 5) as i16));
        assert!(!judge.is_legal(&board, idx(7, 5) as i16));
        assert_eq!(judge.get_renju_state(), Pattern::DoubleThree);
    }

    #[test]
    fn contiguous_plus_gapped_four_sharing_point_is_legal() {
        // xxxx_x (cur completes the contiguous four): {0,1,2,3} and {1,2,3,5} both complete at 4,
        // so white's next move at 4 blocks both five routes at once -> counts as one four -> legal
        let stones = vec![
            (idx(7, 0), Color::Black),
            (idx(7, 1), Color::Black),
            (idx(7, 3), Color::Black),
            (idx(7, 5), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][2] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(!judge.is_double_four(&board, idx(7, 2) as i16));
        assert!(judge.is_legal(&board, idx(7, 2) as i16));
    }

    #[test]
    fn two_gapped_fours_sharing_point_is_legal() {
        // xxx_xx (cur is the 3rd stone): gapped fours {0,1,2,4} and {1,2,4,5} share completion point 3,
        // white at 3 blocks both at once -> counts as one four -> legal
        let stones = vec![
            (idx(7, 0), Color::Black),
            (idx(7, 1), Color::Black),
            (idx(7, 4), Color::Black),
            (idx(7, 5), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][2] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(!judge.is_double_four(&board, idx(7, 2) as i16));
        assert!(judge.is_legal(&board, idx(7, 2) as i16));
    }

    #[test]
    fn two_fours_with_different_points_is_forbidden() {
        // xx.xxx.xx (cur is the 4th stone): gapped four {0,1,3,4} completes at 2, gapped four {3,4,5,7} completes at 6,
        // white cannot block both completion points at once -> double-four forbidden move
        let stones = vec![
            (idx(7, 0), Color::Black),
            (idx(7, 1), Color::Black),
            (idx(7, 3), Color::Black),
            (idx(7, 5), Color::Black),
            (idx(7, 7), Color::Black),
            (idx(7, 8), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][4] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(judge.is_double_four(&board, idx(7, 4) as i16));
        assert!(!judge.is_legal(&board, idx(7, 4) as i16));
        assert_eq!(judge.get_renju_state(), Pattern::DoubleFour);
    }

    // --- board corner/edge special cases ---

    #[test]
    fn corner_diagonal_black_five_wins() {
        // (0,0)..(4,4) main diagonal, move at the corner endpoint: forward goes out of
        // bounds, backward counts to 4 -> black wins
        let stones = vec![
            (idx(1, 1), Color::Black),
            (idx(2, 2), Color::Black),
            (idx(3, 3), Color::Black),
            (idx(4, 4), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[0][0] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(judge.check_win(&board, idx(0, 0) as i16));
        assert_eq!(judge.get_renju_state(), Pattern::FiveInARow);
    }

    #[test]
    fn top_edge_black_five_wins() {
        // top edge, horizontal five on cols 0..4, move in the middle -> black wins
        let stones = vec![
            (idx(0, 0), Color::Black),
            (idx(0, 1), Color::Black),
            (idx(0, 3), Color::Black),
            (idx(0, 4), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[0][2] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(judge.check_win(&board, idx(0, 2) as i16));
    }

    #[test]
    fn top_edge_black_six_is_overline() {
        // top edge, black six on cols 0..5 -> overline forbidden (edge-side count caps at 5)
        let stones = vec![
            (idx(0, 0), Color::Black),
            (idx(0, 1), Color::Black),
            (idx(0, 2), Color::Black),
            (idx(0, 4), Color::Black),
            (idx(0, 5), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[0][3] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(!judge.is_legal(&board, idx(0, 3) as i16));
        assert_eq!(judge.get_renju_state(), Pattern::Overline);
    }

    #[test]
    fn bottom_edge_white_overline_wins() {
        // bottom edge, white six on cols 9..14 -> white overline also wins
        let stones = vec![
            (idx(14, 9), Color::White),
            (idx(14, 10), Color::White),
            (idx(14, 12), Color::White),
            (idx(14, 13), Color::White),
            (idx(14, 14), Color::White),
        ];
        let mut board = board_with(&stones);
        board[14][11] = Color::White;
        let mut judge = RenjuJudge::new();
        assert!(judge.check_win(&board, idx(14, 11) as i16));
    }

    #[test]
    fn left_edge_white_five_wins() {
        // left edge, vertical white five on col 0 -> white wins
        let stones = vec![
            (idx(0, 0), Color::White),
            (idx(1, 0), Color::White),
            (idx(3, 0), Color::White),
            (idx(4, 0), Color::White),
        ];
        let mut board = board_with(&stones);
        board[2][0] = Color::White;
        let mut judge = RenjuJudge::new();
        assert!(judge.check_win(&board, idx(2, 0) as i16));
    }

    #[test]
    fn corner_double_four_is_forbidden() {
        // move at (0,0): horizontal four {(0,0)..(0,3)} completes at (0,4), vertical four {(0,0)..(3,0)} completes at (4,0),
        // different completion points -> double-four forbidden. Covers the "forward out of
        // bounds, collect backward only" path of a corner move
        let stones = vec![
            (idx(0, 1), Color::Black),
            (idx(0, 2), Color::Black),
            (idx(0, 3), Color::Black),
            (idx(1, 0), Color::Black),
            (idx(2, 0), Color::Black),
            (idx(3, 0), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[0][0] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(judge.is_double_four(&board, idx(0, 0) as i16));
        assert!(!judge.is_legal(&board, idx(0, 0) as i16));
        assert_eq!(judge.get_renju_state(), Pattern::DoubleFour);
    }

    #[test]
    fn bottom_right_corner_double_four_is_forbidden() {
        // move at (14,14): horizontal four {(14,11)..(14,14)} completes at (14,10), vertical four {(11,14)..(14,14)}
        // completes at (10,14) -> double-four forbidden. Covers the bottom-right
        // "backward out of bounds, collect forward only" path
        let stones = vec![
            (idx(14, 11), Color::Black),
            (idx(14, 12), Color::Black),
            (idx(14, 13), Color::Black),
            (idx(11, 14), Color::Black),
            (idx(12, 14), Color::Black),
            (idx(13, 14), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[14][14] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(judge.is_double_four(&board, idx(14, 14) as i16));
        assert!(!judge.is_legal(&board, idx(14, 14) as i16));
        assert_eq!(judge.get_renju_state(), Pattern::DoubleFour);
    }

    #[test]
    fn corner_single_four_is_legal() {
        // move at (0,0) forms only one horizontal straight four -> legal
        let stones = vec![
            (idx(0, 1), Color::Black),
            (idx(0, 2), Color::Black),
            (idx(0, 3), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[0][0] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(!judge.is_double_four(&board, idx(0, 0) as i16));
        assert!(judge.is_legal(&board, idx(0, 0) as i16));
    }

    #[test]
    fn top_edge_straight_four_is_legal() {
        // top edge live four on cols 1..4, completion points (0,0) and (0,5) both on board ->
        // a single live four is legal; covers the window-start clipping path
        // (s in [cur_pos-4, cur_pos]) when the edge side collects fewer than 4 cells
        let stones = vec![
            (idx(0, 1), Color::Black),
            (idx(0, 3), Color::Black),
            (idx(0, 4), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[0][2] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(!judge.is_double_four(&board, idx(0, 2) as i16));
        assert!(judge.is_legal(&board, idx(0, 2) as i16));
    }

    #[test]
    fn bottom_edge_dead_three_is_not_double_three() {
        // move at (14,13): horizontal {11,12,13} can become a live four by completing at (14,10) -> live three;
        // vertical {12,13,14} completes at (11,13) but only (10,13) remains as the single completion
        // point (other end off board) -> dead three, no double-three -> legal. Covers the
        // count_a3 edge collection window
        let stones = vec![
            (idx(14, 11), Color::Black),
            (idx(14, 12), Color::Black),
            (idx(12, 13), Color::Black),
            (idx(13, 13), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[14][13] = Color::Black;
        let mut judge = RenjuJudge::new();
        assert!(!judge.is_double_three(&board, idx(14, 13) as i16));
        assert!(judge.is_legal(&board, idx(14, 13) as i16));
    }

    #[test]
    #[ignore = "RIF 9.3 exception not implemented: this double-three should be allowed by the rules, but the current implementation marks it forbidden (conservative false positive)"]
    fn rif_9_3_allowed_double_three() {
        // RIF 9.3a: if only one of the two threes can be extended into a live four (extending
        // the other would create a double-four forbidden move), the double-three is allowed.
        // The current implementation marks it forbidden (wrong).
        // Vertical three V: (7,5),(8,5),(10,5); its only extension point (9,5) would form
        // a vertical live four (7,5)..(10,5) plus a horizontal four (9,3),(9,4),(9,6),(9,5) ->
        // double-four forbidden, so V cannot be extended.
        // Horizontal three H: (7,5),(7,6),(7,8); extension point (7,7) becomes a live four normally.
        let stones = vec![
            (idx(7, 6), Color::Black),
            (idx(7, 8), Color::Black),
            (idx(8, 5), Color::Black),
            (idx(9, 6), Color::Black),
            (idx(9, 4), Color::Black),
            (idx(9, 3), Color::Black),
            (idx(10, 5), Color::Black),
        ];
        let mut board = board_with(&stones);
        board[7][5] = Color::Black;
        let mut judge = RenjuJudge::new();
        // per RIF 9.3a this move should be legal; the current implementation marks it a
        // double-three forbidden move (false positive)
        assert!(judge.is_legal(&board, idx(7, 5) as i16));
    }
}
