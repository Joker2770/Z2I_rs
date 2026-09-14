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

    /// Free-style win check (five or more in a row), lazily constructing the judge.
    fn free_style_win(&mut self, board: &Board, last_move: i16) -> bool {
        match self.rule_grp.free_style_obj {
            None => {
                let o = FreeStyleJudge::new();
                let is_win = o.check_win(board, last_move);
                self.rule_grp.free_style_obj = Some(o);
                is_win
            }
            Some(f) => f.check_win(board, last_move),
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
            // Empty board (no move played yet): no win and no forbidden move.
            // Update chk_rst so callers reading self.chk_rst see the same result
            // as the returned reference.
            self.chk_rst = (GameStage::Running, Color::Blank);
            return &self.chk_rst;
        }

        // INFO rule 0 (free-style): the winner is decided solely by the free-style
        // judge — five or more stones in a row win, with no forbidden-move restrictions.
        if *rule_flag == RuleFlag::FreeStyle {
            if self.free_style_win(board, last_move) {
                let idx = last_move as usize;
                let s = board_size as usize;
                self.chk_rst = (GameStage::End, board[idx / s][idx % s]);
            } else {
                self.chk_rst = (GameStage::Running, Color::Blank);
            }
            return &self.chk_rst;
        }

        // Other rules: free-style five-in-a-row is a necessary baseline; each
        // additional sub-rule (standard exactly-five / renju / caro) further constrains it.
        let mut is_win = self.free_style_win(board, last_move);
        let mut flag = RuleFlag::FreeStyle;
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
            // Combined rules (e.g. standard-caro = 0b1001) require all mandatory sub-rules to win simultaneously:
            // flag is the union of all winning sub-rules this turn and must contain every bit of rule_flag.
            is_win = *rule_flag & flag == *rule_flag;
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
            } else {
                // Legal Renju move (including any White move, which can never be a
                // forbidden move): the game is still running. Update chk_rst so it
                // never keeps a stale terminal value from a previous check.
                self.chk_rst = (GameStage::Running, Color::Blank);
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
        if b_s >= n_in_row && n_in_row >= cfg::MIN_BOARD_SIZE && b_s <= cfg::MAX_BOARD_SIZE {
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

    pub fn get_board_size(&self) -> u8 {
        self.board_size
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

    /// Remove the points the active rule forbids **the side to move** from the playable set.
    ///
    /// The stored table only encodes "empty cell" (`load_position`, `execute_move`): nothing in
    /// it knows about Renju's forbidden moves, so without this the search priors, the Dirichlet
    /// noise, the stored training target and the move actually played all disagree with the
    /// judge that decides the game. A Black move the judge rejects ends the game with White
    /// winning (`CheckResult::value`), i.e. the "mover loses on its own move" ending that turns
    /// the value labels into a function of the side to move and collapses a value head onto the
    /// colour plane.
    ///
    /// It asks the same `RenjuJudge::is_legal` the judge asks, so the two cannot disagree.
    /// Only Renju with Black to move does any work: every other rule (and White to move under
    /// Renju, since White has no forbidden moves) returns immediately with the table untouched,
    /// which keeps those rules bit-for-bit as they were.
    ///
    /// Call it once per move, before starting a search and after `execute_move`: it costs one
    /// `is_legal` pass over the empty points, which is a decision-point price, not a per-node
    /// one -- the search keeps using the cheap table on its own clones.
    ///
    /// Returns how many points were removed.
    pub fn refresh_playable_moves(&mut self) -> usize {
        if !self.rule_flag.contains(RuleFlag::Renju) || self.cur_color != Color::Black {
            return 0;
        }
        let size = self.board_size as usize;
        let mut forbidden = 0usize;
        for index in 0..self.legal_moves_hash_tab.len() {
            if self.legal_moves_hash_tab[index] != 1 {
                continue;
            }
            let (row, col) = (index / size, index % size);
            // `RenjuJudge::is_legal` looks at the stone that is already on the board at the
            // point, so the candidate has to be placed for the question to mean anything. It is
            // a pure query: the board goes back to Blank right after, and the judge only
            // remembers the pattern it saw.
            self.board[row][col] = Color::Black;
            let is_legal = match self.check_result.rule_grp.renju_obj {
                Some(ref mut judge) => judge.is_legal(&self.board, index as i16),
                None => {
                    let mut judge = RenjuJudge::new();
                    let is_legal = judge.is_legal(&self.board, index as i16);
                    self.check_result.rule_grp.renju_obj = Some(judge);
                    is_legal
                }
            };
            self.board[row][col] = Color::Blank;
            if !is_legal {
                self.legal_moves_hash_tab[index] = 0;
                forbidden += 1;
            }
        }
        forbidden
    }

    /// Whether this finished game was decided by the mover's own forbidden move.
    ///
    /// Under Renju a forbidden Black move ends the game with White as the winner, which the
    /// status alone cannot tell apart from a White five; the board can: a five is always
    /// completed by the winner's own stone, so a White win whose last stone is Black's can only
    /// come from the forbidden-move rule. Read it after `get_game_status`.
    pub fn ended_by_forbidden_move(&self) -> bool {
        let (stage, winner) = self.check_result.chk_rst;
        if stage != GameStage::End || winner != Color::White || self.last_move < 0 {
            return false;
        }
        let size = self.board_size as usize;
        let index = self.last_move as usize;
        self.board[index / size][index % size] == Color::Black
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
        // the side to move may have forbidden points (Renju Black): make the playable set agree
        // with the judge from the start instead of only after the first search refresh
        self.refresh_playable_moves();
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

    /// The Renju double-four shape the judge rejects: six Black stones on row 7 leave `(7, 4)`
    /// as the point that would complete two fours at once, i.e. a forbidden move.
    /// Six White stones keep the stone counts even, so White has just moved and Black is up;
    /// they are spread out because six in a row would already be an overline win for White and
    /// the position would not be a live decision point any more.
    const FORBIDDEN_POINT: u16 = 7 * 15 + 4;

    fn double_four_stones() -> Vec<(u16, Color)> {
        let mut stones = Vec::new();
        for col in [0u16, 1, 3, 5, 7, 8] {
            stones.push((7 * 15 + col, Color::Black));
        }
        for col in [0u16, 2, 4, 6, 8, 10] {
            stones.push((col, Color::White));
        }
        stones
    }

    fn renju_game(stones: &[(u16, Color)], next_color: Color) -> Gomoku {
        let mut game = Gomoku::new(15, 5).expect("valid test board");
        assert!(game.set_rule(RuleFlag::Renju));
        assert!(game.load_position(stones, next_color));
        game
    }

    #[test]
    fn refresh_playable_moves_is_a_no_op_without_forbidden_moves() {
        let stones = double_four_stones();
        for rule in [RuleFlag::FreeStyle, RuleFlag::Standard, RuleFlag::Caro] {
            let mut game = Gomoku::new(15, 5).expect("valid test board");
            assert!(game.set_rule(rule));
            assert!(game.load_position(&stones, Color::Black));
            let before = game.get_legal_moves().clone();

            assert_eq!(game.refresh_playable_moves(), 0, "{rule:?} does no work");
            assert_eq!(
                game.get_legal_moves()[FORBIDDEN_POINT as usize],
                1,
                "{rule:?} has no forbidden moves, so the point stays playable"
            );
            assert_eq!(*game.get_legal_moves(), before, "{rule:?} is untouched");
        }
    }

    #[test]
    fn refresh_playable_moves_removes_only_renju_black_forbidden_points() {
        let stones = double_four_stones();

        // Black to move: the double-four point is gone and the refresh is idempotent.
        let black = renju_game(&stones, Color::Black);
        assert_eq!(
            black.get_legal_moves()[FORBIDDEN_POINT as usize],
            0,
            "the judge forbids it for Black"
        );
        let playable = black.get_legal_moves().iter().filter(|m| **m == 1).count();
        let removable = 15 * 15 - stones.len() - playable;
        assert!(
            removable >= 1,
            "at least the double-four point must be removed"
        );
        let mut black = black;
        assert_eq!(
            black.refresh_playable_moves(),
            0,
            "a second pass changes nothing"
        );
        assert_eq!(black.get_legal_moves()[FORBIDDEN_POINT as usize], 0);

        // White to move on the same shape: White has no forbidden moves at all.
        let mut white_stones = stones.clone();
        white_stones.push((14 * 15, Color::Black));
        let white = renju_game(&white_stones, Color::White);
        assert_eq!(
            white.get_legal_moves()[FORBIDDEN_POINT as usize],
            1,
            "White may play the point that is forbidden for Black"
        );
    }

    #[test]
    fn ended_by_forbidden_move_tells_apart_losses_from_fives() {
        let stones = double_four_stones();
        let mut game = renju_game(&stones, Color::Black);
        // `execute_move` only knows about occupancy, so the forbidden point lands on the board
        // exactly like it did in the poisoned self-play games; the judge then ends the game
        // with White winning.
        assert!(game.execute_move(FORBIDDEN_POINT));
        assert_eq!(*game.get_game_status(), (GameStage::End, Color::White));
        assert!(
            game.ended_by_forbidden_move(),
            "White won because Black's own move was forbidden"
        );

        // an ordinary five is not a forbidden-move ending, for either colour
        for (color, expected_winner) in [(Color::White, Color::White), (Color::Black, Color::Black)]
        {
            let mut five = Gomoku::new(15, 5).expect("valid test board");
            assert!(five.set_rule(RuleFlag::Renju));
            for col in 0..5usize {
                five.board[3][col] = color;
            }
            five.legal_moves_hash_tab[3 * 15 + 4] = 0;
            five.sum_cur_actions = 10;
            five.last_move = (3 * 15 + 4) as i16;
            assert_eq!(*five.get_game_status(), (GameStage::End, expected_winner));
            assert!(
                !five.ended_by_forbidden_move(),
                "a {color:?} five is an ordinary win"
            );
        }
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

    // --- INFO rule 9: standard caro (0b1001 = exactly-five(1) | caro(8)) ---

    fn standard_caro() -> RuleFlag {
        RuleFlag::Standard | RuleFlag::Caro
    }

    /// Build a 15x15 game, apply rule 9 and load the position.
    /// Note: `get_game_status` checks for a win at `last_move`, so the last element
    /// of stones must be the key move to judge; also the total move count must be >= 9
    /// for the rule check to trigger.
    fn rule9_game(stones: &[(u16, Color)]) -> Gomoku {
        let mut gomoku = Gomoku::new(15, 5).unwrap();
        assert!(gomoku.set_rule(standard_caro()));
        assert!(gomoku.load_position(stones, Color::White));
        gomoku
    }

    #[test]
    fn rule_9_is_standard_caro_bitmask() {
        // protocol: rule = bitmask, 1=exactly five, 8=caro, 9=sum of both
        assert_eq!(
            RuleFlag::from_bits_truncate(9),
            RuleFlag::Standard | RuleFlag::Caro
        );
    }

    #[test]
    fn standard_caro_open_five_wins() {
        // xxxxx, open at both ends -> wins
        let mut stones = vec![
            (0u16, Color::White),
            (1, Color::White),
            (2, Color::White),
            (3, Color::White),
        ];
        stones.extend((4..=8).map(|c| ((7 * 15 + c) as u16, Color::Black)));
        let mut gomoku = rule9_game(&stones);
        assert_eq!(gomoku.get_game_status(), &(GameStage::End, Color::Black));
    }

    #[test]
    fn standard_caro_five_blocked_at_one_end_wins() {
        // o xxxxx _, blocked at one end only -> wins
        let mut stones = vec![
            ((7 * 15 + 3) as u16, Color::White),
            (0u16, Color::White),
            (1, Color::White),
            (2, Color::White),
        ];
        stones.extend((4..=8).map(|c| ((7 * 15 + c) as u16, Color::Black)));
        let mut gomoku = rule9_game(&stones);
        assert_eq!(gomoku.get_game_status(), &(GameStage::End, Color::Black));
    }

    #[test]
    fn standard_caro_five_blocked_at_both_ends_is_not_win() {
        // o xxxxx o, exactly five but blocked at both ends -> not a win
        let mut stones = vec![
            ((7 * 15 + 3) as u16, Color::White),
            ((7 * 15 + 9) as u16, Color::White),
            (0u16, Color::White),
            (1, Color::White),
        ];
        stones.extend((4..=8).map(|c| ((7 * 15 + c) as u16, Color::Black)));
        let mut gomoku = rule9_game(&stones);
        assert_eq!(
            gomoku.get_game_status(),
            &(GameStage::Running, Color::Blank)
        );
    }

    #[test]
    fn standard_caro_six_in_a_row_is_not_win() {
        // xxxxxx, overline does not satisfy exactly five -> not a win
        let mut stones = vec![(0u16, Color::White), (1, Color::White), (2, Color::White)];
        stones.extend((3..=8).map(|c| ((7 * 15 + c) as u16, Color::Black)));
        let mut gomoku = rule9_game(&stones);
        assert_eq!(
            gomoku.get_game_status(),
            &(GameStage::Running, Color::Blank)
        );
    }

    // --- INFO rule 0: free-style (five or more in a row wins) ---

    fn rule0_game(stones: &[(u16, Color)]) -> Gomoku {
        let mut gomoku = Gomoku::new(15, 5).unwrap();
        assert!(gomoku.set_rule(RuleFlag::FreeStyle));
        assert!(gomoku.load_position(stones, Color::White));
        gomoku
    }

    #[test]
    fn rule_0_five_in_a_row_wins() {
        // Black fills row 7 columns 4..=8 (five in a row); white padding brings the
        // move count to 9 so the rule check triggers, and the last stone is black.
        let mut stones = vec![
            (0u16, Color::White),
            (1, Color::White),
            (2, Color::White),
            (3, Color::White),
        ];
        stones.extend((4..=8).map(|c| ((7 * 15 + c) as u16, Color::Black)));
        let mut gomoku = rule0_game(&stones);
        assert_eq!(gomoku.get_game_status(), &(GameStage::End, Color::Black));
    }

    #[test]
    fn rule_0_overline_wins() {
        // six in a row still wins under free-style
        let mut stones = vec![(0u16, Color::White), (1, Color::White), (2, Color::White)];
        stones.extend((3..=8).map(|c| ((7 * 15 + c) as u16, Color::Black)));
        let mut gomoku = rule0_game(&stones);
        assert_eq!(gomoku.get_game_status(), &(GameStage::End, Color::Black));
    }

    #[test]
    fn rule_0_four_in_a_row_is_not_win() {
        // only four black stones in a row: no win under free-style
        let mut stones = vec![
            (0u16, Color::White),
            (1, Color::White),
            (2, Color::White),
            (3, Color::White),
            (4, Color::White),
        ];
        stones.extend((4..=7).map(|c| ((7 * 15 + c) as u16, Color::Black)));
        let mut gomoku = rule0_game(&stones);
        assert_eq!(
            gomoku.get_game_status(),
            &(GameStage::Running, Color::Blank)
        );
    }

    // --- INFO rule 4: renju (forbidden moves) ---

    /// A legal Renju move must overwrite any stale terminal value left in `chk_rst`
    /// by a previous check (e.g. an earlier forbidden move). This guards against
    /// `CheckResult::value` returning a stale result when read directly.
    #[test]
    fn renju_legal_move_overwrites_stale_terminal_value() {
        let mut gomoku = Gomoku::new(15, 5).unwrap();
        assert!(gomoku.set_rule(RuleFlag::Renju));

        // First: a black overline (six in a row) is a forbidden move -> (End, White).
        let mut stones: Vec<(u16, Color)> = (0..4).map(|c| (c as u16, Color::White)).collect();
        stones.extend((4..=9).map(|c| ((7 * 15 + c) as u16, Color::Black)));
        assert!(gomoku.load_position(&stones, Color::White));
        let rule_flag = gomoku.rule_flag;
        let board = gomoku.board.clone();
        let last_move = gomoku.last_move;
        let result = gomoku
            .check_result
            .value(&rule_flag, &board, gomoku.board_size, last_move);
        assert_eq!(*result, (GameStage::End, Color::White));

        // Second: replace the board with a legal non-winning move (a lone black stone)
        // without resetting `check_result`, so the stale (End, White) is still present.
        gomoku.board = vec![vec![Color::Blank; 15]; 15];
        gomoku.board[7][7] = Color::Black;
        gomoku.last_move = (7 * 15 + 7) as i16;
        let board = gomoku.board.clone();
        let last_move = gomoku.last_move;
        let result = gomoku
            .check_result
            .value(&rule_flag, &board, gomoku.board_size, last_move);
        assert_eq!(*result, (GameStage::Running, Color::Blank));
    }
}
