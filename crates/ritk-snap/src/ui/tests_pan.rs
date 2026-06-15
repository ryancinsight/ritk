use super::*;

#[test]
fn identity_zero_delta_preserves_viewport_origin() {
    let origin = Pos2::new(100.0, 50.0);
    let start = Pos2::new(200.0, 150.0);
    let result = pan_from_drag_delta(origin, start, start);
    assert_eq!(result.x, 100.0);
    assert_eq!(result.y, 50.0);
}

#[test]
fn rightward_drag_increases_x_offset() {
    let origin = Pos2::new(0.0, 0.0);
    let start = Pos2::new(100.0, 100.0);
    let current = Pos2::new(130.0, 100.0);
    let result = pan_from_drag_delta(origin, start, current);
    assert_eq!(result.x, 30.0);
    assert_eq!(result.y, 0.0);
}

#[test]
fn leftward_drag_decreases_x_offset() {
    let origin = Pos2::new(100.0, 100.0);
    let start = Pos2::new(100.0, 100.0);
    let current = Pos2::new(60.0, 100.0);
    let result = pan_from_drag_delta(origin, start, current);
    assert_eq!(result.x, 60.0);
    assert_eq!(result.y, 100.0);
}

#[test]
fn downward_drag_increases_y_offset() {
    let origin = Pos2::new(0.0, 0.0);
    let start = Pos2::new(100.0, 100.0);
    let current = Pos2::new(100.0, 130.0);
    let result = pan_from_drag_delta(origin, start, current);
    assert_eq!(result.x, 0.0);
    assert_eq!(result.y, 30.0);
}

#[test]
fn upward_drag_decreases_y_offset() {
    let origin = Pos2::new(100.0, 100.0);
    let start = Pos2::new(100.0, 100.0);
    let current = Pos2::new(100.0, 60.0);
    let result = pan_from_drag_delta(origin, start, current);
    assert_eq!(result.x, 100.0);
    assert_eq!(result.y, 60.0);
}

#[test]
fn diagonal_drag_updates_both_components_independently() {
    let origin = Pos2::new(50.0, 75.0);
    let start = Pos2::new(200.0, 150.0);
    let current = Pos2::new(220.0, 130.0);
    let result = pan_from_drag_delta(origin, start, current);
    // delta = (20.0, -20.0)
    // new_pan = (70.0, 55.0)
    assert_eq!(result.x, 70.0);
    assert_eq!(result.y, 55.0);
}

#[test]
fn large_positive_drag_produces_proportional_offset_increase() {
    let origin = Pos2::new(0.0, 0.0);
    let start = Pos2::new(0.0, 0.0);
    let current = Pos2::new(100.0, 200.0);
    let result = pan_from_drag_delta(origin, start, current);
    assert_eq!(result.x, 100.0);
    assert_eq!(result.y, 200.0);
}

#[test]
fn large_negative_drag_produces_proportional_offset_decrease() {
    let origin = Pos2::new(500.0, 400.0);
    let start = Pos2::new(100.0, 100.0);
    let current = Pos2::new(0.0, 0.0);
    let result = pan_from_drag_delta(origin, start, current);
    assert_eq!(result.x, 400.0);
    assert_eq!(result.y, 300.0);
}

#[test]
fn fractional_delta_is_preserved() {
    let origin = Pos2::new(10.0, 20.0);
    let start = Pos2::new(50.0, 50.0);
    let current = Pos2::new(52.5, 48.3);
    let result = pan_from_drag_delta(origin, start, current);
    // delta = (2.5, -1.7)
    // new_pan = (12.5, 18.3)
    assert_eq!(result.x, 12.5);
    assert!((result.y - 18.3).abs() < 1e-6);
}
