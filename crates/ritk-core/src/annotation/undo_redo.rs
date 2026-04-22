//! Generic undo/redo stack for command-pattern state management.
//!
//! # Mathematical Specification
//!
//! The stack maintains a history sequence H = [s0, s1, ..., sn] and a future
//! sequence F = [f0, f1, ..., fm]. Invariants:
//! - H is non-empty: current() = H.last() is always defined.
//! - undo(): n >= 1 => push sn onto F, pop H -> H = [s0,...,sn-1]; returns true.
//!           n = 0 (only s0 remains) => returns false (cannot undo initial state).
//! - redo(): m >= 1 => push f0 (top of F) onto H, pop F; returns true.
//!           m = 0 => returns false.
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
    future: Vec<S>,
}

impl<S: Clone> UndoRedoStack<S> {
    /// Construct a new stack with the given initial state.
    /// `history = [initial]`, `future = []`.
    pub fn new(initial: S) -> Self {
        Self { history: vec![initial], future: Vec::new() }
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
            Some(state) => { self.history.push(state); true }
            None => false,
        }
    }

    /// Return a reference to the current state (most recently pushed or undone-to).
    pub fn current(&self) -> &S {
        self.history.last().expect("UndoRedoStack history invariant: non-empty")
    }

    /// `true` if undo is possible (history depth > 1).
    pub fn can_undo(&self) -> bool { self.history.len() > 1 }

    /// `true` if redo is possible (future is non-empty).
    pub fn can_redo(&self) -> bool { !self.future.is_empty() }

    /// Number of states in history (including current).
    pub fn history_depth(&self) -> usize { self.history.len() }

    /// Number of states available for redo.
    pub fn future_depth(&self) -> usize { self.future.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undo_redo_stack_initial() {
        let stack: UndoRedoStack<i32> = UndoRedoStack::new(42);
        assert_eq!(stack.current(), &42);
        assert!(!stack.can_undo());
        assert!(!stack.can_redo());
        assert_eq!(stack.history_depth(), 1);
        assert_eq!(stack.future_depth(), 0);
    }

    #[test]
    fn test_undo_redo_push_increments_history() {
        let mut stack = UndoRedoStack::new(0i32);
        stack.push(10);
        stack.push(20);
        assert_eq!(stack.history_depth(), 3);
        assert_eq!(stack.current(), &20);
    }

    #[test]
    fn test_undo_one_step() {
        let mut stack = UndoRedoStack::new(0i32);
        stack.push(1);
        stack.push(2);
        let result = stack.undo();
        assert!(result, "undo must return true when history has > 1 state");
        assert_eq!(stack.current(), &1);
    }

    #[test]
    fn test_undo_at_initial_returns_false() {
        let mut stack = UndoRedoStack::new(0i32);
        let result = stack.undo();
        assert!(!result, "undo at initial state must return false");
        assert_eq!(stack.current(), &0);
    }

    #[test]
    fn test_redo_after_undo() {
        let mut stack = UndoRedoStack::new(0i32);
        stack.push(1);
        stack.push(2);
        stack.undo();
        let result = stack.redo();
        assert!(result, "redo must return true when future is non-empty");
        assert_eq!(stack.current(), &2);
    }

    #[test]
    fn test_redo_with_no_future_returns_false() {
        let mut stack = UndoRedoStack::new(0i32);
        let result = stack.redo();
        assert!(!result, "redo with empty future must return false");
        assert_eq!(stack.current(), &0);
    }

    #[test]
    fn test_push_clears_future() {
        let mut stack = UndoRedoStack::new(0i32);
        stack.push(1);
        stack.push(2);
        stack.undo();
        stack.push(3);
        assert!(!stack.can_redo(), "future must be empty after push");
        assert_eq!(stack.current(), &3);
        assert_eq!(stack.future_depth(), 0);
    }

    #[test]
    fn test_multiple_undo_redo_cycle() {
        let mut stack = UndoRedoStack::new(0i32);
        for i in 1..=5 {
            stack.push(i);
        }
        // history: [0,1,2,3,4,5]
        stack.undo(); stack.undo(); stack.undo();
        // history: [0,1,2], future: [5,4,3]
        assert_eq!(stack.current(), &2);
        stack.redo(); stack.redo();
        // history: [0,1,2,3,4], future: [5]
        assert_eq!(stack.current(), &4);
        assert_eq!(stack.history_depth(), 5);
        assert_eq!(stack.future_depth(), 1);
    }

    #[test]
    fn test_undo_redo_with_string_state() {
        let mut stack = UndoRedoStack::new(String::from("initial"));
        stack.push(String::from("hello"));
        stack.push(String::from("world"));
        stack.undo();
        assert_eq!(stack.current().as_str(), "hello");
        stack.undo();
        assert_eq!(stack.current().as_str(), "initial");
    }

    #[test]
    fn test_future_depth_tracking() {
        let mut stack = UndoRedoStack::new(0i32);
        stack.push(1);
        stack.push(2);
        stack.undo();
        assert_eq!(stack.future_depth(), 1);
        assert_eq!(stack.current(), &1);
        stack.redo();
        assert_eq!(stack.future_depth(), 0);
        assert_eq!(stack.current(), &2);
    }
}
