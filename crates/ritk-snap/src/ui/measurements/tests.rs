use super::*;

/// `perpendicular_offset` for a horizontal right-pointing line must return
/// a straight-up vector.
///
/// Analytical: direction = (1, 0) → perpendicular CCW = (0, 1) rotated =
/// (-0, 1) → normalised (-dy, dx) = (0, 1).
/// With distance = 10: result = (0, 10).
#[test]
fn test_perpendicular_offset_horizontal() {
    let p1 = [0.0, 0.0];
    let p2 = [10.0, 0.0];
    let off = perpendicular_offset(p1, p2, 10.0);
    // (-dy/|d|, dx/|d|) * distance = (-0/10, 10/10) * 10 = (0.0, 1.0) * 10
    assert!(
        off[0].abs() < 1e-4,
        "horizontal line perpendicular x must be 0, got {}",
        off[0]
    );
    assert!(
        (off[1] - 10.0).abs() < 1e-4,
        "horizontal line perpendicular y must be 10, got {}",
        off[1]
    );
}

/// `perpendicular_offset` for a vertical downward line must return a
/// right-pointing vector.
///
/// Analytical: direction = (0, 1) → (-dy, dx) = (-1, 0).
/// With distance = 5: result = (-5, 0).
#[test]
fn test_perpendicular_offset_vertical() {
    let p1 = [0.0, 0.0];
    let p2 = [0.0, 10.0];
    let off = perpendicular_offset(p1, p2, 5.0);
    // direction = (0, 1), normalised = (0, 1).
    // (-dy, dx) = (-1, 0) * 5 = (-5, 0).
    assert!(
        (off[0] - (-5.0)).abs() < 1e-4,
        "vertical line perpendicular x must be -5, got {}",
        off[0]
    );
    assert!(
        off[1].abs() < 1e-4,
        "vertical line perpendicular y must be 0, got {}",
        off[1]
    );
}

/// `perpendicular_offset` for a degenerate (zero-length) line must return
/// a straight-up fallback `(0, -d)`.
#[test]
fn test_perpendicular_offset_degenerate() {
    let p = [5.0, 5.0];
    let off = perpendicular_offset(p, p, 8.0);
    assert!(
        off[0].abs() < 1e-4,
        "degenerate line perpendicular x must be 0, got {}",
        off[0]
    );
    assert!(
        (off[1] - (-8.0)).abs() < 1e-4,
        "degenerate line perpendicular y must be -8, got {}",
        off[1]
    );
}
