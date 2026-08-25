// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use crate::{
    caro::CaroJudge,
    configuration::cfg,
    free_style::FreeStyleJudge,
    renju::RenjuJudge,
    rule::{Board, Color, RuleFlag},
    standard::StandardJudge,
};

#[derive(Clone)]
struct RuleObj {
    free_style_obj: Option<FreeStyleJudge>,
    stand_obj: Option<StandardJudge>,
    renju_obj: Option<RenjuJudge>,
    caro_obj: Option<CaroJudge>,
}

#[derive(Clone)]
struct CheckResult {
    rule_grp: RuleObj,
    chk_rst: (GameStage, Color),
}

impl CheckResult {
    fn new() -> Self {
        let obj = RuleObj {
            free_style_obj: None,
            stand_obj: None,
            renju_obj: None,
            caro_obj: None,
        };
        Self {
            rule_grp: obj,
            chk_rst: (GameStage::Running, Color::Blank),
        }
    }

    fn value(
        &mut self,
        rule_flag: &RuleFlag,
        board: &Board,
        board_size: u8,
        last_move: i16,
    ) -> &(GameStage, Color) {
        if last_move < 0 {
            return &(GameStage::Running, Color::Blank);
        }
        let mut is_win = false;
        let mut flag = RuleFlag::FreeStyle;
        match self.rule_grp.free_style_obj {
            None => {
                let o = FreeStyleJudge::new();
                self.rule_grp.free_style_obj = Some(o);
                is_win = o.check_win(board, last_move);
            }
            Some(f) => {
                is_win = f.check_win(board, last_move);
            }
        }
        if rule_flag.contains(RuleFlag::Standard) {
            match self.rule_grp.stand_obj {
                Some(s) => {
                    if s.check_win(board, last_move) {
                        flag |= RuleFlag::Standard;
                    } else {
                        is_win = false;
                    }
                }
                None => {
                    let o = StandardJudge::new();
                    self.rule_grp.stand_obj = Some(o);
                    if o.check_win(board, last_move) {
                        flag |= RuleFlag::Standard;
                    } else {
                        is_win = false;
                    }
                }
            }
        }
        if rule_flag.contains(RuleFlag::Renju) {
            match self.rule_grp.renju_obj {
                Some(mut r) => {
                    if r.check_win(board, last_move) {
                        flag |= RuleFlag::Renju;
                    } else {
                        is_win = false;
                    }
                }
                None => {
                    let mut o = RenjuJudge::new();
                    if o.check_win(board, last_move) {
                        flag |= RuleFlag::Renju;
                    } else {
                        is_win = false;
                    }
                    self.rule_grp.renju_obj = Some(o);
                }
            }
        }
        if rule_flag.contains(RuleFlag::Caro) {
            match self.rule_grp.caro_obj {
                Some(c) => {
                    if c.check_win(board, last_move) {
                        flag |= RuleFlag::Caro;
                    } else {
                        is_win = false;
                    }
                }
                None => {
                    let o = CaroJudge::new();
                    self.rule_grp.caro_obj = Some(o);
                    if o.check_win(board, last_move) {
                        flag |= RuleFlag::Caro;
                    } else {
                        is_win = false;
                    }
                }
            }
        }

        if RuleFlag::FreeStyle != flag {
            is_win = *rule_flag & flag == flag;
        }

        if is_win {
            let idx = last_move as usize;
            let s = board_size as usize;
            let row = (idx / s) as isize;
            let col = (idx % s) as isize;

            self.chk_rst = (GameStage::End, board[row as usize][col as usize]);
        } else if rule_flag.contains(RuleFlag::Renju) {
            if let Some(mut r) = self.rule_grp.renju_obj
                && !(r.is_legal(board, last_move))
            {
                self.chk_rst = (GameStage::End, Color::White);
            }
        } else {
            self.chk_rst = (GameStage::Running, Color::Blank);
        }

        &self.chk_rst
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GameStage {
    Running = 0,
    End = 1,
}

#[derive(Clone)]
pub struct Gomoku {
    board_size: u8,
    board: Board,
    cur_color: Color,
    last_move: i16,
    n_in_row: u8,
    rule_flag: RuleFlag,
    sum_cur_actions: u16,
    legal_moves_hash_tab: Vec<u8>,
    check_result: CheckResult,
}

impl Gomoku {
    pub fn new(b_s: u8, n_in_row: u8) -> Option<Self> {
        let rule_flag = if let Some(rf) = RuleFlag::from_bits(cfg::DEFAULT_RULE_FLAG) {
            rf
        } else {
            RuleFlag::FreeStyle
        };
        if b_s > n_in_row && n_in_row >= cfg::MIN_BOARD_SIZE && b_s <= cfg::MAX_BOARD_SIZE {
            let g = Gomoku {
                board_size: b_s,
                cur_color: Color::Black,
                last_move: -1,
                board: vec![vec![Color::Blank; b_s as usize]; b_s as usize],
                n_in_row,
                rule_flag,
                sum_cur_actions: 0,
                legal_moves_hash_tab: vec![1; (b_s * b_s) as usize],
                check_result: CheckResult::new(),
            };
            Some(g)
        } else {
            None
        }
    }

    pub fn get_action_size(&self) -> u16 {
        (self.board_size * self.board_size) as u16
    }

    pub fn get_board(&self) -> &Board {
        &self.board
    }

    pub fn get_last_move(&self) -> i16 {
        self.last_move
    }

    pub fn get_rule(&self) -> &RuleFlag {
        &self.rule_flag
    }

    pub fn get_cur_color(&self) -> &Color {
        &self.cur_color
    }

    pub fn get_legal_moves(&self) -> &Vec<u8> {
        &self.legal_moves_hash_tab
    }

    pub fn has_blank_pos(&self) -> bool {
        self.sum_cur_actions < self.get_action_size()
    }

    pub fn is_illegal(&self, x: u8, y: u8) -> bool {
        if x > (self.board_size - 1) || y > (self.board_size - 1) {
            true
        } else {
            self.board[x as usize][y as usize] != Color::Blank
        }
    }

    pub fn set_rule(&mut self, rule_flag: RuleFlag) -> bool {
        if -1 == self.last_move {
            self.rule_flag = rule_flag;
            true
        } else {
            false
        }
    }

    pub fn load_position(&mut self, stones: &[(u16, Color)], next_color: Color) -> bool {
        let board_size = self.board_size as u16;
        let mut board =
            vec![vec![Color::Blank; self.board_size as usize]; self.board_size as usize];
        let mut legal_moves = vec![1; self.get_action_size() as usize];

        for &(move_idx, color) in stones {
            if move_idx >= self.get_action_size() || color == Color::Blank {
                return false;
            }
            let row = (move_idx / board_size) as usize;
            let col = (move_idx % board_size) as usize;
            if board[row][col] != Color::Blank {
                return false;
            }
            board[row][col] = color;
            legal_moves[move_idx as usize] = 0;
        }

        self.board = board;
        self.legal_moves_hash_tab = legal_moves;
        self.sum_cur_actions = stones.len() as u16;
        self.last_move = stones.last().map_or(-1, |(move_idx, _)| *move_idx as i16);
        self.cur_color = next_color;
        self.check_result = CheckResult::new();
        true
    }

    pub fn execute_move(&mut self, move_idx: u16) -> bool {
        let i = (move_idx / self.board_size as u16) as u8;
        let j = (move_idx % self.board_size as u16) as u8;

        if self.is_illegal(i, j) {
            false
        } else {
            let p_state = self.board[i as usize][j as usize];
            if p_state != Color::Blank {
                println!("Board[{}][{}] = {:?}!!!", i, j, p_state);
                false
            } else {
                self.board[i as usize][j as usize] = self.cur_color;
                self.legal_moves_hash_tab[move_idx as usize] = 0;
                self.sum_cur_actions += 1;
                self.last_move = move_idx as i16;
                self.cur_color = if Color::White == self.cur_color {
                    Color::Black
                } else if Color::Black == self.cur_color {
                    Color::White
                } else {
                    Color::Blank
                };

                true
            }
        }
    }

    pub fn get_game_status(&mut self) -> &(GameStage, Color) {
        if self.n_in_row == 5 {
            if self.sum_cur_actions >= 9 {
                let _s_c = self.check_result.value(
                    &self.rule_flag,
                    &self.board,
                    self.board_size,
                    self.last_move,
                );

                if self.check_result.chk_rst.0 == GameStage::End {
                    return &self.check_result.chk_rst;
                }

                if self.has_blank_pos() {
                    self.check_result.chk_rst = (GameStage::Running, Color::Blank);
                } else {
                    self.check_result.chk_rst = (GameStage::End, Color::Blank);
                }
            } else {
                self.check_result.chk_rst = (GameStage::Running, Color::Blank);
            }
        } else if let Some(winner) = self.find_n_in_row_winner() {
            self.check_result.chk_rst = (GameStage::End, winner);
        } else if self.has_blank_pos() {
            self.check_result.chk_rst = (GameStage::Running, Color::Blank);
        } else {
            self.check_result.chk_rst = (GameStage::End, Color::Blank);
        }

        &self.check_result.chk_rst
    }

    fn render_to_string(&self) -> String {
        let last_move = self.last_move;
        let last_pos = if last_move >= 0 {
            Some((
                (last_move as usize) / (self.board_size as usize),
                (last_move as usize) % (self.board_size as usize),
            ))
        } else {
            None
        };

        let mut out = String::new();
        for r in 0..self.board_size as usize {
            for c in 0..self.board_size as usize {
                let symbol = match self.board[r][c] {
                    Color::Black => 'x',
                    Color::White => 'o',
                    Color::Blank => '.',
                };
                let symbol = if let Some((lr, lc)) = last_pos {
                    if lr == r && lc == c {
                        match symbol {
                            'x' => 'X',
                            'o' => 'O',
                            other => other,
                        }
                    } else {
                        symbol
                    }
                } else {
                    symbol
                };
                out.push(symbol);
                if c + 1 < self.board_size as usize {
                    out.push(' ');
                }
            }
            out.push('\n');
        }
        out
    }

    pub fn render(&self) {
        print!("{}", self.render_to_string());
    }

    fn find_n_in_row_winner(&self) -> Option<Color> {
        let board_size = self.board_size as usize;
        let n_in_row = self.n_in_row as usize;

        for row in 0..board_size {
            for col in 0..board_size {
                if self.board[row][col] == Color::Blank {
                    continue;
                }

                let directions = [(0isize, 1isize), (1, 0), (1, 1), (1, -1)];
                for (row_step, col_step) in directions {
                    let end_row = row as isize + (n_in_row - 1) as isize * row_step;
                    let end_col = col as isize + (n_in_row - 1) as isize * col_step;
                    if end_row < 0
                        || end_row >= board_size as isize
                        || end_col < 0
                        || end_col >= board_size as isize
                    {
                        continue;
                    }

                    let mut sum = 0;
                    for offset in 0..n_in_row {
                        let check_row = (row as isize + offset as isize * row_step) as usize;
                        let check_col = (col as isize + offset as isize * col_step) as usize;
                        sum += self.board[check_row][check_col] as i32;
                    }
                    if sum.abs() == self.n_in_row as i32 {
                        return Some(self.board[row][col]);
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_empty_board() {
        let gomoku = Gomoku::new(4, 3).unwrap();
        let expected = ". . . .\n. . . .\n. . . .\n. . . .\n";
        assert_eq!(gomoku.render_to_string(), expected);
    }

    #[test]
    fn render_board_with_last_move_highlight() {
        let mut gomoku = Gomoku::new(4, 3).unwrap();
        gomoku.board[1][1] = Color::White;
        gomoku.last_move = 5;
        let expected = ". . . .\n. O . .\n. . . .\n. . . .\n";
        assert_eq!(gomoku.render_to_string(), expected);
    }

    #[test]
    fn render_large_gomoku_board() {
        let mut gomoku = Gomoku::new(15, 5).unwrap();
        gomoku.board[0][0] = Color::Black;
        gomoku.board[0][1] = Color::White;
        gomoku.board[1][0] = Color::White;
        gomoku.board[14][14] = Color::Black;
        gomoku.last_move = 14 * 15 + 14;

        let mut expected = String::new();
        expected.push_str("x o . . . . . . . . . . . . .\n");
        expected.push_str("o . . . . . . . . . . . . . .\n");
        for _ in 2..14 {
            expected.push_str(". . . . . . . . . . . . . . .\n");
        }
        expected.push_str(". . . . . . . . . . . . . . X\n");

        assert_eq!(gomoku.render_to_string(), expected);
    }

    #[test]
    fn n_in_row_detects_all_directions() {
        let cases = [
            vec![(0, Color::Black), (1, Color::Black), (2, Color::Black)],
            vec![(0, Color::Black), (5, Color::Black), (10, Color::Black)],
            vec![(0, Color::Black), (6, Color::Black), (12, Color::Black)],
            vec![(2, Color::Black), (6, Color::Black), (10, Color::Black)],
        ];

        for stones in cases {
            let mut gomoku = Gomoku::new(5, 3).unwrap();
            assert!(gomoku.load_position(&stones, Color::White));
            assert_eq!(gomoku.get_game_status(), &(GameStage::End, Color::Black));
        }
    }

    #[test]
    fn n_in_row_without_winner_is_running() {
        let mut gomoku = Gomoku::new(5, 3).unwrap();
        assert!(gomoku.load_position(
            &[(0, Color::Black), (1, Color::Black), (7, Color::Black)],
            Color::White,
        ));
        assert_eq!(
            gomoku.get_game_status(),
            &(GameStage::Running, Color::Blank)
        );
    }
}
