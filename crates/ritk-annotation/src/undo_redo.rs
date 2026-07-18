//! Generic undo/redo stack for command-pattern state management.
//!
//! # Mathematical Specification
//!
//! The stack maintains a history sequence H = [s0, s1, ..., sn] and a future
//! sequence F = [f0, f1, ..., fm]. Invariants:
//! - H is non-empty: current() = H.last() is always defined.
//! - undo(): n >= 1 => push sn onto F, pop H -> H = [s0,...,sn-1]; returns true.
//!   n = 0 (only s0 remains) => returns false (cannot undo initial state).
//! - redo(): m >= 1 => push f0 (top of F) onto H, pop F; returns true.
//!   m = 0 => returns false.
//! - push(s): appends s to H, clears F entirely.
//!   New actions discard the redo future (branching undo semantics).
//!
//! # Type Parameter
//! `S: Clone` --- states are cloned on push to ensure the stack owns each snapshot.

/// A generic undo/redo history stack.
///
/// `S` must implement `Clone` so each snapshot is independently owned.
#[derive(Debug, Clone)]
pub struct UndoRedoStack<S: Clone> {
    /// History of states. Non-empty invariant: always has at least the initial state.
    history: Vec<S>,
    /// Future states available for redo.
    future: Vec<S> }

impl<S: Clone> UndoRedoStack<S> {
    /// Construct a new stack with the given initial state.
    /// `history = [initial]`, `future = []`.
    pub fn new(initial: S) -> Self {
        Self {
            history: vec![initial],
            future: Vec::new() }
    }

    /// Push a new state onto the history. Clears the redo future.
    pub fn push(&mut self, state: S) {
        self.future.clear();
        self.history.push(state);
    }

    /// Undo one step.
    ///
    /// Returns `true` if undo was possible (history had > 1 state).
    /// Returns `false` if already at the initial state.
    pub fn undo(&mut self) -> bool {
        if self.history.len() <= 1 {
            return false;
        }
        let current = self.history.pop().expect("history non-empty; qed");
        self.future.push(current);
        true
    }

    /// Redo one step.
    ///
    /// Returns `true` if redo was possible (future was non-empty).
    /// Returns `false` if no future state is available.
    pub fn redo(&mut self) -> bool {
        match self.future.pop() {
            Some(state) => {
                self.history.push(state);
                true
            }
            None => false }
    }

    /// Return a reference to the current state (most recently pushed or undone-to).
    pub fn current(&self) -> &S {
        self.history
            .last()
            .expect("UndoRedoStack history invariant: non-empty")
    }

    /// `true` if undo is possible (history depth > 1).
    pub fn can_undo(&self) -> bool {
        self.history.len() > 1
    }

    /// `true` if redo is possible (future is non-empty).
    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }

    /// Number of states in history (including current).
    pub fn history_depth(&self) -> usize {
        self.history.len()
    }

    /// Number of states available for redo.
    pub fn future_depth(&self) -> usize {
        self.future.len()
    }
}

#[cfg(test)]
#[path = "tests_undo_redo.rs"]
mod tests;
