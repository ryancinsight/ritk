use super::*;

/// Cube cardinality formula: `(2r+1)³` for r ∈ {0, 1, 2, 3, 5}.
#[test]
fn cube_cardinality_matches_formula() {
    for r in 0..=5usize {
        let expected = (2 * r + 1).pow(3);
        assert_eq!(
            Cube::offsets(r).len(),
            expected,
            "cube(r={r}) must have (2r+1)³ = {expected} offsets"
        );
        assert_eq!(
            cube_cardinality(r),
            expected,
            "cube_cardinality(r={r}) must equal (2r+1)³"
        );
    }
}

/// Cross cardinality formula: `3(2r+1) - 2`.
#[test]
fn cross_cardinality_matches_formula() {
    for r in 0..=5usize {
        let expected = 3 * (2 * r + 1) - 2;
        assert_eq!(
            Cross::offsets(r).len(),
            expected,
            "cross(r={r}) must have 3(2r+1)-2 = {expected} offsets"
        );
        assert_eq!(
            cross_cardinality(r),
            expected,
            "cross_cardinality(r={r}) must equal 3(2r+1)-2"
        );
    }
}

/// Ball cardinality for small radii matches known values.
///
/// # Derivation
/// `ball(r)` = `#{x ∈ ℤ³ : ‖x‖₂² ≤ r²}`. The count by squared distance:
/// - d²=0: 1 (origin)
/// - d²=1: 6 (face-centre points (±1,0,0), (0,±1,0), (0,0,±1))
/// - d²=2: 12 (edge-midpoints (±1,±1,0) etc, 3 axes × 4 sign combos)
/// - d²=3: 8 (corner points (±1,±1,±1))
/// - d²=4: 6 (face-centre points (±2,0,0) etc)
///
/// Total for r=1: 1+6 = 7. Total for r=2: 1+6+12+8+6 = 33.
#[test]
fn ball_cardinality_matches_known_values() {
    assert_eq!(Ball::offsets(0).len(), 1, "ball(0) = origin");
    assert_eq!(
        Ball::offsets(1).len(),
        7,
        "ball(1) = origin + 6 face centres"
    );
    assert_eq!(
        Ball::offsets(2).len(),
        33,
        "ball(2) = origin + 6 d²=1 + 12 d²=2 + 8 d²=3 + 6 d²=4"
    );
}

/// All three shape markers are Zero-Sized Types.
#[test]
fn shape_markers_are_zsts() {
    assert_eq!(std::mem::size_of::<Cube>(), 0, "Cube must be a ZST");
    assert_eq!(std::mem::size_of::<Cross>(), 0, "Cross must be a ZST");
    assert_eq!(std::mem::size_of::<Ball>(), 0, "Ball must be a ZST");
}

/// The origin `(0, 0, 0)` is included in every non-empty shape.
#[test]
fn origin_is_included_in_every_shape() {
    let origin = Offset3D::new(0, 0, 0);
    for r in 0..=3usize {
        assert!(
            Cube::offsets(r).contains(&origin),
            "cube r={r} must contain origin"
        );
        assert!(
            Cross::offsets(r).contains(&origin),
            "cross r={r} must contain origin"
        );
        assert!(
            Ball::offsets(r).contains(&origin),
            "ball r={r} must contain origin"
        );
    }
}

/// Offsets are deterministic across multiple invocations.
#[test]
fn offsets_are_deterministic() {
    let a = Cube::offsets(2);
    let b = Cube::offsets(2);
    assert_eq!(a, b, "Cube::offsets must be deterministic");

    let a = Cross::offsets(2);
    let b = Cross::offsets(2);
    assert_eq!(a, b, "Cross::offsets must be deterministic");

    let a = Ball::offsets(2);
    let b = Ball::offsets(2);
    assert_eq!(a, b, "Ball::offsets must be deterministic");
}
