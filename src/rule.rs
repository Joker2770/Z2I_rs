// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Joker2770

use bitflags::bitflags;

pub type Board = Vec<Vec<Color>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Color {
    Black = 1,
    Blank = 0,
    White = -1,
}

bitflags! {
    #[derive(Clone, Copy,Debug,PartialEq)]
    pub struct RuleFlag: u8 {
       const Caro = 0b1000;
       const FreeStyle = 0b0000;
       const Renju = 0b0100;
       const Standard = 0b0001;
    }
}

pub trait RuleOpt {
    /// Check whether last_move causes a win. Implementations may mutate self.
    fn check_win(&mut self, board: &Board, last_move: i16) -> bool;
}
