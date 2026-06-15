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
    stack.undo();
    stack.undo();
    stack.undo();
    // history: [0,1,2], future: [5,4,3]
    assert_eq!(stack.current(), &2);
    stack.redo();
    stack.redo();
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
