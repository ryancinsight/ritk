use super::*;

/// Zero drag leaves (center, width) unchanged.
#[test]
fn zero_drag_is_identity() {
    let (c, w) = window_level_from_drag_delta(40.0, 400.0, 0.0, 0.0, WINDOW_LEVEL_SENSITIVITY);
    assert_eq!(c, 40.0);
    assert_eq!(w, 400.0);
}

/// Positive horizontal drag (right) increases window width proportionally.
#[test]
fn rightward_drag_increases_width() {
    let (c, w) = window_level_from_drag_delta(40.0, 400.0, 10.0, 0.0, WINDOW_LEVEL_SENSITIVITY);
    // new_width = 400.0 + 10.0 * 4.0 = 440.0
    assert_eq!(w, 440.0);
    assert_eq!(c, 40.0);
}

/// Negative horizontal drag (left) decreases window width, floored at MIN.
#[test]
fn leftward_drag_decreases_width_clamped() {
    // Starting width = 400, dx = -200 → raw = 400 − 800 = −400 → clamp to 1.0
    let (_c, w) = window_level_from_drag_delta(40.0, 400.0, -200.0, 0.0, WINDOW_LEVEL_SENSITIVITY);
    assert_eq!(w, MIN_WINDOW_WIDTH);
}

/// Positive vertical drag (down) decreases window center (y-axis inverted).
#[test]
fn downward_drag_decreases_center() {
    let (c, w) = window_level_from_drag_delta(40.0, 400.0, 0.0, 10.0, WINDOW_LEVEL_SENSITIVITY);
    // new_center = 40.0 − 10.0 * 4.0 = 0.0
    assert_eq!(c, 0.0);
    assert_eq!(w, 400.0);
}

/// Negative vertical drag (up) increases window center.
#[test]
fn upward_drag_increases_center() {
    let (c, _w) = window_level_from_drag_delta(40.0, 400.0, 0.0, -10.0, WINDOW_LEVEL_SENSITIVITY);
    // new_center = 40.0 − (−10.0) * 4.0 = 80.0
    assert_eq!(c, 80.0);
}

/// Width is monotone non-decreasing in dx for dx ≥ 0.
#[test]
fn width_monotone_nondecreasing_in_positive_dx() {
    let widths: Vec<f64> = (0..=20)
        .map(|i| {
            let (_, w) =
                window_level_from_drag_delta(0.0, 100.0, i as f32, 0.0, WINDOW_LEVEL_SENSITIVITY);
            w
        })
        .collect();
    for pair in widths.windows(2) {
        assert!(
            pair[1] >= pair[0],
            "width decreased: {} → {}",
            pair[0],
            pair[1]
        );
    }
}

/// Center is monotone decreasing in dy for dy > 0 (downward = lower center).
#[test]
fn center_monotone_decreasing_in_positive_dy() {
    let centers: Vec<f64> = (0..=20)
        .map(|i| {
            let (c, _) =
                window_level_from_drag_delta(0.0, 100.0, 0.0, i as f32, WINDOW_LEVEL_SENSITIVITY);
            c
        })
        .collect();
    for pair in centers.windows(2) {
        assert!(
            pair[1] <= pair[0],
            "center increased on downward drag: {} → {}",
            pair[0],
            pair[1]
        );
    }
}

/// Diagonal drag updates both center and width independently (no coupling).
#[test]
fn diagonal_drag_updates_center_and_width_independently() {
    let (c, w) = window_level_from_drag_delta(100.0, 200.0, 5.0, -5.0, WINDOW_LEVEL_SENSITIVITY);
    // new_width  = 200 + 5*4 = 220
    assert_eq!(w, 220.0);
    // new_center = 100 − (−5)*4 = 120
    assert_eq!(c, 120.0);
}

/// clamp_window_width enforces MIN_WINDOW_WIDTH.
#[test]
fn clamp_window_width_enforces_minimum() {
    assert_eq!(clamp_window_width(0.0), MIN_WINDOW_WIDTH);
    assert_eq!(clamp_window_width(-100.0), MIN_WINDOW_WIDTH);
    assert_eq!(clamp_window_width(0.5), MIN_WINDOW_WIDTH);
    assert_eq!(clamp_window_width(1.0), 1.0);
    assert_eq!(clamp_window_width(500.0), 500.0);
}
