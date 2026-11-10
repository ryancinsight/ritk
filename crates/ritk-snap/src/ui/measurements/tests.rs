use super::*;

/// A rightward line has direction `(1, 0)`, so `(-dy, dx) = (0, 1)` and the
/// offset is `(0, distance)`. The operands are exact in binary.
#[test]
fn a_rightward_line_is_offset_along_positive_y() {
    assert_eq!(
        perpendicular_offset([0.0, 0.0], [10.0, 0.0], 10.0),
        [0.0, 10.0]
    );
}

/// A downward line has direction `(0, 1)`, so `(-dy, dx) = (-1, 0)`.
#[test]
fn a_downward_line_is_offset_along_negative_x() {
    assert_eq!(
        perpendicular_offset([0.0, 0.0], [0.0, 10.0], 5.0),
        [-5.0, 0.0]
    );
}

/// A 3-4-5 line scales the unit perpendicular `(-4, 3) / 5` by the distance.
#[test]
fn a_diagonal_offset_has_the_requested_length() {
    assert_eq!(
        perpendicular_offset([1.0, 1.0], [4.0, 5.0], 10.0),
        [-8.0, 6.0]
    );
}

/// Equal endpoints have no direction and fall back to straight up.
#[test]
fn a_degenerate_line_is_offset_straight_up() {
    assert_eq!(
        perpendicular_offset([5.0, 5.0], [5.0, 5.0], 8.0),
        [0.0, -8.0]
    );
}
