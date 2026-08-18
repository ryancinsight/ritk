//! Tests for the hoisted Cartesian index/world transform pair.

use super::CartesianGridGeometry;
use crate::{CoordinateMap, CurvilinearArray, Direction, Point, Spacing};

/// A quarter-turn fixture keeps every transform operation exactly representable.
///
/// - origin `(10, 20, 30)` mm LPS
/// - spacing `[4, 2, 8]` in tensor-axis order
/// - direction rotates the (x, y) plane by 90 degrees, with columns that differ
///   from rows so direction application and axis selection remain observable.
fn quarter_turn() -> CartesianGridGeometry<3> {
    let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(
        direction.is_orthogonal(),
        "fixture direction must be orthonormal"
    );
    CartesianGridGeometry::new(
        &Point::new([10.0, 20.0, 30.0]),
        &Spacing::try_new([4.0, 2.0, 8.0]).expect("invariant: fixture spacing is positive"),
        &direction,
        &CoordinateMap::Cartesian,
    )
    .expect("invariant: fixture is Cartesian with an orthonormal direction")
}

#[test]
fn rotated_index_maps_to_the_hand_computed_point() {
    // index [1, 1, 1] -> scaled (4, 2, 8) -> direction (-2, 4, 8)
    // -> point (8, 24, 38). Without the direction the point would be
    // (14, 22, 38), so the rotation is value-observable.
    assert_eq!(quarter_turn().point([1.0, 1.0, 1.0]), [8.0, 24.0, 38.0]);
}

#[test]
fn rotated_point_maps_back_to_the_hand_computed_index() {
    assert_eq!(quarter_turn().index([8.0, 24.0, 38.0]), [1.0, 1.0, 1.0]);
}

#[test]
fn index_and_point_round_trip_on_a_non_lattice_coordinate() {
    // Binary fractions exercise the inverse away from a voxel centre. The
    // quarter-turn inverse, integer spacing, and origin leave the exact index.
    let geometry = quarter_turn();
    let index = [0.25, -1.5, 3.75];
    assert_eq!(geometry.index(geometry.point(index)), index);
}

#[test]
fn a_displacement_rotates_without_the_origin_translation() {
    // A displacement is a free vector: the index offset it induces must not
    // pick up the origin. Displacement (-2, 4, 8) is exactly D*S*[1, 1, 1].
    let geometry = quarter_turn();
    let base = [3.0, -2.0, 11.0];
    let shifted = [base[0] - 2.0, base[1] + 4.0, base[2] + 8.0];
    let a = geometry.index(shifted);
    let b = geometry.index(base);
    let difference = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    assert_eq!(difference, [1.0, 1.0, 1.0]);
}

#[test]
fn identity_direction_reduces_to_the_axis_aligned_affine() {
    // The bit-identity guard: with the identity direction the transform must be
    // exactly `origin + spacing (.) index`, so an axis-aligned volume is
    // unaffected by this type's introduction.
    let geometry = CartesianGridGeometry::new(
        &Point::new([10.0, 20.0, 30.0]),
        &Spacing::try_new([4.0, 3.0, 2.0]).expect("invariant: fixture spacing is positive"),
        &Direction::identity(),
        &CoordinateMap::Cartesian,
    )
    .expect("invariant: identity direction is invertible");
    let index = [2.0, 3.0, 4.0];
    assert_eq!(
        geometry.point(index),
        [10.0 + 2.0 * 4.0, 20.0 + 3.0 * 3.0, 30.0 + 4.0 * 2.0],
        "identity direction must be bit-identical to the direction-free affine"
    );
}

#[test]
fn the_two_dimensional_instantiation_applies_the_same_formula() {
    // The type is generic over rank; the 2-D instantiation is the same formula.
    // The exact quarter-turn maps scaled index (4, 2) to (-2, 4), so the point
    // is (8, 24) and the inverse returns index (1, 1).
    let geometry = CartesianGridGeometry::<2>::new(
        &Point::new([10.0, 20.0]),
        &Spacing::try_new([4.0, 2.0]).expect("invariant: fixture spacing is positive"),
        &Direction::from_rows([[0.0, -1.0], [1.0, 0.0]]),
        &CoordinateMap::Cartesian,
    )
    .expect("invariant: fixture is Cartesian with an orthonormal direction");

    assert_eq!(geometry.point([1.0, 1.0]), [8.0, 24.0]);
    assert_eq!(geometry.index([8.0, 24.0]), [1.0, 1.0]);
}

#[test]
fn a_nonsingular_nonorthogonal_direction_uses_its_inverse() {
    let direction = Direction::from_rows([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(!direction.is_orthogonal());
    let geometry = CartesianGridGeometry::new(
        &Point::new([3.0, 5.0, 7.0]),
        &Spacing::try_new([2.0, 4.0, 1.0]).expect("invariant: fixture spacing is positive"),
        &direction,
        &CoordinateMap::Cartesian,
    )
    .expect("invariant: nonsingular Cartesian direction has an inverse");

    // The scaled index [2, 8, 3] maps to [6, 8, 3]; adding the origin gives
    // [9, 13, 10]. Transposing the shear would produce a different point and
    // would not recover the original index.
    assert_eq!(geometry.point([1.0, 2.0, 3.0]), [9.0, 13.0, 10.0]);
    assert_eq!(geometry.index([9.0, 13.0, 10.0]), [1.0, 2.0, 3.0]);
}

#[test]
fn axis_direction_returns_the_direction_column_not_the_row() {
    // For a quarter-turn, column 0 is (0, 1, 0) and row 0 is (0, -1, 0).
    assert_eq!(quarter_turn().axis_direction(0), [0.0, 1.0, 0.0]);
    assert_eq!(quarter_turn().axis_direction(1), [-1.0, 0.0, 0.0]);
}

#[test]
fn a_beam_space_acquisition_is_rejected_rather_than_mapped_affinely() {
    // The whole reason the constructor takes a map: a curvilinear index pair is
    // a beam and a sample, and the affine formula would answer with a point in
    // no physical space at all.
    let beam = CurvilinearArray::try_new(1.0, 0.5, 0.01, -0.5).expect("valid curvilinear geometry");
    let error = CartesianGridGeometry::<3>::new(
        &Point::origin(),
        &Spacing::try_new([1.0, 1.0, 1.0]).expect("invariant: unit spacing is positive"),
        &Direction::identity(),
        &CoordinateMap::CurvilinearArray(beam),
    )
    .expect_err("a beam-space acquisition must be rejected");
    assert!(
        error.to_string().contains("Cartesian coordinate map"),
        "error should name the coordinate map, got {error}"
    );
}

#[test]
fn a_singular_direction_is_reported_rather_than_panicking() {
    let singular = Direction::from_rows([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let error = CartesianGridGeometry::new(
        &Point::origin(),
        &Spacing::try_new([1.0, 1.0, 1.0]).expect("invariant: unit spacing is positive"),
        &singular,
        &CoordinateMap::Cartesian,
    )
    .expect_err("a singular direction must be rejected");
    assert!(
        error.to_string().contains("singular"),
        "error should name the singular direction, got {error}"
    );
}
