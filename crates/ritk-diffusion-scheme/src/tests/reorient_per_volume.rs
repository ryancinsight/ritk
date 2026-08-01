//! Per-volume gradient reorientation.
//!
//! Motion and eddy-current correction registers each volume independently, so
//! each acquires its own rotation. The cases below pin the behaviours that
//! separate a correct implementation from the ones that still produce a
//! plausible-looking scheme: applying the right rotation to the right volume,
//! passing b = 0 through without inventing an orientation, and refusing a
//! rotation list that does not cover the series.

use ritk_spatial::Vector;

use crate::{GradientFrame, GradientScheme, GradientSchemeError};

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Right-handed rotation by `angle` about z.
fn rotation_z(angle: f64) -> [[f64; 3]; 3] {
    let (sin, cos) = angle.sin_cos();
    [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]
}

/// Right-handed rotation by `angle` about y.
fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (sin, cos) = angle.sin_cos();
    [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]]
}

fn scheme_from(pairs: Vec<(f64, [f64; 3])>) -> GradientScheme {
    GradientScheme::from_seconds_per_square_millimeter(
        pairs
            .into_iter()
            .map(|(b, direction)| (b, Vector::new(direction)))
            .collect(),
        GradientFrame::ImageAxis,
    )
    .expect("valid scheme")
}

fn directions_of(scheme: &GradientScheme) -> Vec<[f64; 3]> {
    scheme
        .directions()
        .iter()
        .map(|entry| entry.direction().to_array())
        .collect()
}

fn assert_close(actual: [f64; 3], expected: [f64; 3], context: &str) {
    // Rotation of a unit vector is exact to a few ulp; 1e-12 is orders of
    // magnitude above that and orders below any real orientation error.
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() < 1e-12,
            "{context}: axis {axis} is {} but expected {}",
            actual[axis],
            expected[axis]
        );
    }
}

#[test]
fn each_volume_receives_its_own_rotation() {
    // Three volumes along +x, each rotated about z by a different angle. If the
    // implementation applied one rotation to all of them, or paired them by the
    // wrong index, at most one volume would land correctly.
    let scheme = scheme_from(vec![
        (1000.0, [1.0, 0.0, 0.0]),
        (1000.0, [1.0, 0.0, 0.0]),
        (1000.0, [1.0, 0.0, 0.0]),
    ]);
    let quarter = std::f64::consts::FRAC_PI_2;
    let rotations = [rotation_z(0.0), rotation_z(quarter), rotation_z(-quarter)];

    let corrected = scheme
        .reorient_per_volume(&rotations)
        .expect("valid per-volume rotations");
    let actual = directions_of(&corrected);

    assert_close(actual[0], [1.0, 0.0, 0.0], "volume 0 is unrotated");
    assert_close(actual[1], [0.0, 1.0, 0.0], "volume 1 turns +x onto +y");
    assert_close(actual[2], [0.0, -1.0, 0.0], "volume 2 turns +x onto -y");
}

#[test]
fn rotations_about_different_axes_stay_independent() {
    // Mixing axes catches an implementation that accumulates rotations across
    // volumes instead of applying each to its own.
    let scheme = scheme_from(vec![(800.0, [1.0, 0.0, 0.0]), (800.0, [1.0, 0.0, 0.0])]);
    let quarter = std::f64::consts::FRAC_PI_2;

    let corrected = scheme
        .reorient_per_volume(&[rotation_z(quarter), rotation_y(quarter)])
        .expect("valid rotations");
    let actual = directions_of(&corrected);

    assert_close(actual[0], [0.0, 1.0, 0.0], "z-rotation sends +x to +y");
    assert_close(actual[1], [0.0, 0.0, -1.0], "y-rotation sends +x to -z");
}

#[test]
fn unweighted_volumes_pass_through_but_keep_their_slot() {
    // A b = 0 volume has no orientation to rotate. It must still occupy its
    // index, or every gradient after it would be paired with the wrong
    // rotation.
    let scheme = scheme_from(vec![
        (0.0, [0.0, 0.0, 0.0]),
        (1000.0, [1.0, 0.0, 0.0]),
        (0.0, [0.0, 0.0, 0.0]),
        (1000.0, [0.0, 1.0, 0.0]),
    ]);
    let quarter = std::f64::consts::FRAC_PI_2;
    let rotations = [
        rotation_z(quarter),
        rotation_z(quarter),
        rotation_z(quarter),
        rotation_z(quarter),
    ];

    let corrected = scheme
        .reorient_per_volume(&rotations)
        .expect("valid rotations");
    let actual = directions_of(&corrected);

    assert_eq!(actual.len(), 4, "every volume keeps its slot");
    assert_close(actual[1], [0.0, 1.0, 0.0], "weighted volume 1 rotates");
    assert_close(actual[3], [-1.0, 0.0, 0.0], "weighted volume 3 rotates");
    assert!(
        corrected.directions()[0].weighting().is_unweighted()
            && corrected.directions()[2].weighting().is_unweighted(),
        "unweighted volumes stay unweighted"
    );
}

#[test]
fn identity_rotations_leave_the_scheme_unchanged() {
    // The no-motion case. A correction that finds no rotation must be a no-op,
    // not a source of drift.
    let scheme = scheme_from(vec![
        (0.0, [0.0, 0.0, 0.0]),
        (1000.0, [0.0, 1.0, 0.0]),
        (2000.0, [0.0, 0.0, 1.0]),
    ]);

    let corrected = scheme
        .reorient_per_volume(&[IDENTITY, IDENTITY, IDENTITY])
        .expect("identity is a valid rotation");

    assert_eq!(
        corrected, scheme,
        "identity rotations must reproduce the scheme exactly"
    );
}

#[test]
fn uniform_rotations_agree_with_the_single_rotation_path() {
    // The two entry points must not disagree where their domains overlap:
    // supplying the same rotation for every volume is exactly `reorient`.
    let scheme = scheme_from(vec![
        (0.0, [0.0, 0.0, 0.0]),
        (1000.0, [1.0, 0.0, 0.0]),
        (1000.0, [0.0, 1.0, 0.0]),
    ]);
    let rotation = rotation_z(0.7);

    let uniform = scheme
        .reorient_per_volume(&[rotation, rotation, rotation])
        .expect("valid rotations");
    let single = scheme.reorient(rotation).expect("valid rotation");

    assert_eq!(
        uniform, single,
        "uniform per-volume reorientation is reorient"
    );
}

#[test]
fn inverse_rotations_restore_the_original_scheme() {
    // Applying R then Rᵀ per volume is the identity, which checks that the
    // rotation is applied in the stated sense rather than transposed. A
    // transposed implementation passes every fixed-angle test that happens to
    // use a symmetric case, so the round trip is the discriminating property.
    let scheme = scheme_from(vec![
        (1000.0, [1.0, 0.0, 0.0]),
        (1000.0, [0.0, 1.0, 0.0]),
        (1000.0, [0.6, 0.0, 0.8]),
    ]);
    let rotations = [rotation_z(0.3), rotation_y(-0.9), rotation_z(2.1)];
    let inverses: Vec<[[f64; 3]; 3]> = rotations
        .iter()
        .map(|rotation| {
            let mut transposed = [[0.0; 3]; 3];
            for row in 0..3 {
                for column in 0..3 {
                    transposed[row][column] = rotation[column][row];
                }
            }
            transposed
        })
        .collect();

    let restored = scheme
        .reorient_per_volume(&rotations)
        .expect("valid rotations")
        .reorient_per_volume(&inverses)
        .expect("valid inverse rotations");

    for (index, (actual, expected)) in directions_of(&restored)
        .into_iter()
        .zip(directions_of(&scheme))
        .enumerate()
    {
        assert_close(actual, expected, &format!("volume {index} round trip"));
    }
}

#[test]
fn a_short_rotation_list_is_rejected() {
    // Zipping to the shorter length would leave the tail unrotated — a
    // partially corrected scheme that reports success.
    let scheme = scheme_from(vec![
        (1000.0, [1.0, 0.0, 0.0]),
        (1000.0, [0.0, 1.0, 0.0]),
        (1000.0, [0.0, 0.0, 1.0]),
    ]);

    let error = scheme
        .reorient_per_volume(&[IDENTITY, IDENTITY])
        .expect_err("two rotations cannot cover three volumes");

    assert!(
        matches!(
            error,
            GradientSchemeError::RotationCountMismatch {
                expected: 3,
                actual: 2
            }
        ),
        "error must name both counts, got {error}"
    );
}

#[test]
fn a_long_rotation_list_is_rejected() {
    let scheme = scheme_from(vec![(1000.0, [1.0, 0.0, 0.0])]);

    let error = scheme
        .reorient_per_volume(&[IDENTITY, IDENTITY])
        .expect_err("more rotations than volumes is a caller error");

    assert!(matches!(
        error,
        GradientSchemeError::RotationCountMismatch {
            expected: 1,
            actual: 2
        }
    ));
}

#[test]
fn a_non_orthonormal_rotation_is_rejected() {
    // An affine correction's linear part includes scale and shear. Accepting it
    // as a rotation would change gradient magnitudes, silently reweighting the
    // acquisition. Callers must extract the rotational part first.
    let scheme = scheme_from(vec![(1000.0, [1.0, 0.0, 0.0]), (1000.0, [0.0, 1.0, 0.0])]);
    let scaled = [[1.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let error = scheme
        .reorient_per_volume(&[IDENTITY, scaled])
        .expect_err("a scaling matrix is not a rotation");

    assert!(
        matches!(error, GradientSchemeError::InvalidRotation(_)),
        "error must name the rotation contract, got {error}"
    );
}

#[test]
fn a_reflection_is_rejected() {
    // Determinant -1 preserves lengths and angles but flips handedness, which
    // would mirror every fitted orientation. Orthonormality alone does not
    // catch it.
    let scheme = scheme_from(vec![(1000.0, [1.0, 0.0, 0.0])]);
    let reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let error = scheme
        .reorient_per_volume(&[reflection])
        .expect_err("an improper rotation is not a valid correction");

    assert!(matches!(error, GradientSchemeError::InvalidRotation(_)));
}

#[test]
fn a_non_finite_rotation_is_rejected() {
    // A failed registration can produce NaN in its transform. Propagating it
    // would poison the gradient silently.
    let scheme = scheme_from(vec![(1000.0, [1.0, 0.0, 0.0])]);
    let poisoned = [[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let error = scheme
        .reorient_per_volume(&[poisoned])
        .expect_err("NaN cannot be a rotation");

    assert!(matches!(error, GradientSchemeError::InvalidRotation(_)));
}

#[test]
fn validation_precedes_partial_application() {
    // An invalid rotation late in the list must reject the whole call rather
    // than return a scheme with a rotated prefix.
    let scheme = scheme_from(vec![
        (1000.0, [1.0, 0.0, 0.0]),
        (1000.0, [0.0, 1.0, 0.0]),
        (1000.0, [0.0, 0.0, 1.0]),
    ]);
    let reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let error = scheme
        .reorient_per_volume(&[
            rotation_z(std::f64::consts::FRAC_PI_2),
            rotation_z(std::f64::consts::FRAC_PI_2),
            reflection,
        ])
        .expect_err("the third rotation is improper");

    assert!(matches!(error, GradientSchemeError::InvalidRotation(_)));
    assert_eq!(
        directions_of(&scheme)[0],
        [1.0, 0.0, 0.0],
        "the source scheme is unchanged by a failed call"
    );
}

#[test]
fn weightings_are_preserved_through_reorientation() {
    // Reorientation changes direction only. A b-value altered here would
    // rescale the fitted diffusivity.
    let scheme = scheme_from(vec![
        (0.0, [0.0, 0.0, 0.0]),
        (1000.0, [1.0, 0.0, 0.0]),
        (2500.0, [0.0, 1.0, 0.0]),
    ]);

    let corrected = scheme
        .reorient_per_volume(&[rotation_z(1.1), rotation_z(1.1), rotation_y(0.4)])
        .expect("valid rotations");

    let expected: Vec<f64> = scheme
        .directions()
        .iter()
        .map(|entry| entry.weighting().seconds_per_square_millimeter())
        .collect();
    let actual: Vec<f64> = corrected
        .directions()
        .iter()
        .map(|entry| entry.weighting().seconds_per_square_millimeter())
        .collect();

    assert_eq!(actual, expected, "reorientation must not touch weightings");
}

#[test]
fn rotation_preserves_unit_length() {
    // A gradient direction is a unit vector by contract. Rotation is
    // length-preserving, so any drift here signals a malformed matrix reaching
    // the arithmetic.
    let scheme = scheme_from(vec![(1000.0, [0.6, 0.0, 0.8]), (1000.0, [0.0, 0.6, 0.8])]);

    let corrected = scheme
        .reorient_per_volume(&[rotation_z(0.9), rotation_y(-1.7)])
        .expect("valid rotations");

    for (index, direction) in directions_of(&corrected).into_iter().enumerate() {
        let norm = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "volume {index} direction norm is {norm}, expected 1"
        );
    }
}

#[test]
fn an_empty_scheme_cannot_occur_so_zero_rotations_are_rejected() {
    // `GradientScheme` cannot be empty by construction, so a zero-length
    // rotation list is always a count mismatch.
    let scheme = scheme_from(vec![(1000.0, [1.0, 0.0, 0.0])]);

    let error = scheme
        .reorient_per_volume(&[])
        .expect_err("no rotations cannot cover one volume");

    assert!(matches!(
        error,
        GradientSchemeError::RotationCountMismatch {
            expected: 1,
            actual: 0
        }
    ));
}
