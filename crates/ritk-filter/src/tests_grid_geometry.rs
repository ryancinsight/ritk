//! Tests for the hoisted Cartesian index/world transform pair.

use super::CartesianGridGeometry;
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};

/// Deliberately oblique fixture with exact-rational geometry.
///
/// - origin `(10, 20, 30)` mm LPS
/// - spacing `[4, 3, 2]` in tensor-axis order, i.e. `dz = 4`, `dy = 3`, `dx = 2`
/// - direction rotating the (x, y) plane by the exact 3-4-5 angle
///   (`cos = 0.6`, `sin = 0.8`), so column 0 — the physical direction of the
///   slowest index axis — is `(0.6, 0.8, 0)` rather than a coordinate axis.
fn oblique() -> CartesianGridGeometry {
    let direction = Direction::from_rows([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]);
    assert!(
        direction.is_orthogonal(),
        "fixture direction must be orthonormal"
    );
    CartesianGridGeometry::new(
        &Point::new([10.0, 20.0, 30.0]),
        &Spacing::try_new([4.0, 3.0, 2.0]).expect("invariant: fixture spacing is positive"),
        &direction,
        &CoordinateMap::Cartesian,
    )
    .expect("invariant: fixture is Cartesian with an orthonormal direction")
}

#[test]
fn oblique_index_maps_to_the_hand_computed_point() {
    // index [1, 1, 1] -> scaled (4, 3, 2)
    //   D * (4, 3, 2) = 4*(0.6, 0.8, 0) + 3*(-0.8, 0.6, 0) + 2*(0, 0, 1)
    //                 = (2.4, 3.2, 0) + (-2.4, 1.8, 0) + (0, 0, 2)
    //                 = (0, 5, 2)
    //   point = origin + (0, 5, 2) = (10, 25, 32)
    //
    // Dropping the direction would instead give origin + (4, 3, 2) =
    // (14, 23, 32) — a 4.5 mm displacement, and the x component moves the
    // wrong way entirely.
    let got = oblique().point([1.0, 1.0, 1.0]);
    let want = [10.0, 25.0, 32.0];
    for k in 0..3 {
        assert!(
            (got[k] - want[k]).abs() < 1e-12,
            "component {k}: got {got:?}, want {want:?}"
        );
    }
}

#[test]
fn oblique_point_maps_back_to_the_hand_computed_index() {
    let got = oblique().index([10.0, 25.0, 32.0]);
    let want = [1.0, 1.0, 1.0];
    for k in 0..3 {
        assert!(
            (got[k] - want[k]).abs() < 1e-12,
            "component {k}: got {got:?}, want {want:?}"
        );
    }
}

#[test]
fn index_and_point_round_trip_on_a_non_lattice_coordinate() {
    // A fractional index exercises the inverse on a point that is not a voxel
    // centre, where a transpose-vs-inverse mistake shows up as a scale error.
    let geometry = oblique();
    let index = [0.25, -1.5, 3.75];
    let round_tripped = geometry.index(geometry.point(index));
    for k in 0..3 {
        assert!(
            (round_tripped[k] - index[k]).abs() < 1e-12,
            "component {k}: round trip gave {round_tripped:?}, want {index:?}"
        );
    }
}

#[test]
fn a_displacement_rotates_without_the_origin_translation() {
    // A displacement is a free vector: the index offset it induces must not
    // pick up the origin. Displacement (0, 5, 2) is exactly the D*S image of
    // index offset [1, 1, 1].
    let geometry = oblique();
    let base = [3.0, -2.0, 11.0];
    let shifted = [base[0] + 0.0, base[1] + 5.0, base[2] + 2.0];
    let a = geometry.index(shifted);
    let b = geometry.index(base);
    let difference = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    for k in 0..3 {
        assert!(
            (difference[k] - 1.0).abs() < 1e-12,
            "component {k}: got {difference:?}, want [1, 1, 1]"
        );
    }
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
