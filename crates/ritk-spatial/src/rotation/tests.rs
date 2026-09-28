use super::*;

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn rotation_z(angle: f64) -> [[f64; 3]; 3] {
    let (sin, cos) = angle.sin_cos();
    [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]
}

fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (sin, cos) = angle.sin_cos();
    [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]]
}

fn multiply(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut product = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            product[row][column] = (0..3).map(|k| left[row][k] * right[k][column]).sum::<f64>();
        }
    }
    product
}

fn assert_matrices_close(actual: [[f64; 3]; 3], expected: [[f64; 3]; 3], context: &str) {
    for row in 0..3 {
        for column in 0..3 {
            assert!(
                (actual[row][column] - expected[row][column]).abs() < 1e-10,
                "{context}: entry ({row}, {column}) is {} but expected {}",
                actual[row][column],
                expected[row][column]
            );
        }
    }
}

/// Assert orthonormality within the tolerance downstream reorientation uses.
fn assert_proper_rotation(matrix: [[f64; 3]; 3], context: &str) {
    let product = multiply(transpose(matrix), matrix);
    assert_matrices_close(product, IDENTITY, &format!("{context}: RᵀR"));
    let determinant = to_fixed(matrix).determinant();
    assert!(
        (determinant - 1.0).abs() < 1e-10,
        "{context}: determinant is {determinant}, expected 1"
    );
}

fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            result[row][column] = matrix[column][row];
        }
    }
    result
}

#[test]
fn a_rotation_is_returned_unchanged() {
    // The polar factor of an already-orthogonal matrix is itself, since
    // S = I. This is the fixed point the whole construction must satisfy.
    for angle in [0.0, 0.3, 1.1, -2.4, std::f64::consts::PI] {
        let rotation = rotation_z(angle);
        let extracted = rotation_from_linear(rotation).expect("a rotation is invertible");
        assert_matrices_close(
            extracted,
            rotation,
            &format!("z-rotation by {angle} is its own polar factor"),
        );
    }
}

#[test]
fn identity_extracts_to_identity() {
    assert_matrices_close(
        rotation_from_linear(IDENTITY).expect("identity is invertible"),
        IDENTITY,
        "identity",
    );
}

#[test]
fn uniform_scale_is_removed() {
    // R S with S = kI: the rotation must come back exactly, since uniform
    // scaling commutes with rotation and carries no orientation of its own.
    let rotation = rotation_z(0.8);
    let scaled = multiply(
        rotation,
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
    );

    let extracted = rotation_from_linear(scaled).expect("invertible");
    assert_matrices_close(extracted, rotation, "uniform scale removed");
}

#[test]
fn anisotropic_scale_is_removed() {
    // The case that matters for eddy-current correction: per-axis scaling,
    // which does not commute with rotation. Constructing A = R S and
    // recovering R is the oracle, because R is known by construction.
    let rotation = multiply(rotation_z(0.6), rotation_y(-0.4));
    let stretch = [[1.4, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.15]];
    let linear = multiply(rotation, stretch);

    let extracted = rotation_from_linear(linear).expect("invertible");
    assert_matrices_close(extracted, rotation, "anisotropic scale removed");
    assert_proper_rotation(extracted, "anisotropic case");
}

#[test]
fn shear_is_removed() {
    // Shear is the other half of an eddy-current transform. A symmetric
    // positive definite shear is a valid S, so R is again known exactly.
    let rotation = rotation_y(1.2);
    let stretch = [[1.0, 0.2, 0.0], [0.2, 1.1, 0.05], [0.0, 0.05, 0.95]];
    let linear = multiply(rotation, stretch);

    let extracted = rotation_from_linear(linear).expect("invertible");
    assert_matrices_close(extracted, rotation, "shear removed");
}

#[test]
fn extraction_is_idempotent() {
    // Extracting from an already-extracted rotation must change nothing;
    // a drifting implementation would fail on the second pass.
    let linear = multiply(
        rotation_z(0.9),
        [[2.0, 0.1, 0.0], [0.1, 1.3, 0.2], [0.0, 0.2, 0.7]],
    );

    let once = rotation_from_linear(linear).expect("invertible");
    let twice = rotation_from_linear(once).expect("a rotation is invertible");
    assert_matrices_close(twice, once, "idempotent");
}

#[test]
fn output_is_orthonormal_within_the_reorientation_tolerance() {
    // Downstream gradient reorientation validates orthonormality at 1e-9
    // and rejects anything looser, so extraction must clear that bar for
    // realistically conditioned transforms.
    let cases = [
        multiply(
            rotation_z(0.2),
            [[1.05, 0.0, 0.0], [0.0, 0.98, 0.0], [0.0, 0.0, 1.02]],
        ),
        multiply(
            rotation_y(-1.4),
            [[1.0, 0.03, 0.0], [0.03, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        multiply(
            rotation_z(2.9),
            [[3.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
        ),
    ];

    for (index, linear) in cases.into_iter().enumerate() {
        let extracted = rotation_from_linear(linear).expect("invertible");
        assert_proper_rotation(extracted, &format!("case {index}"));
    }
}

#[test]
fn a_reflection_is_rejected_not_repaired() {
    // Handedness reversal between two images of one subject is a defect,
    // not noise to correct. Repairing it would return a plausible rotation
    // for a transform that is wrong.
    let reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let error = rotation_from_linear(reflection).expect_err("determinant is -1");
    assert!(
        matches!(
            error,
            RotationExtractionError::OrientationReversing { determinant } if determinant < 0.0
        ),
        "error must name the reversed orientation, got {error}"
    );
}

#[test]
fn a_rotation_composed_with_a_reflection_is_rejected() {
    // The reversal can hide inside an otherwise ordinary transform.
    let linear = multiply(
        rotation_z(0.7),
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    );

    assert!(matches!(
        rotation_from_linear(linear),
        Err(RotationExtractionError::OrientationReversing { .. })
    ));
}

#[test]
fn a_collapsed_axis_is_rejected() {
    // A zero row maps a whole direction to the origin. No orthogonal matrix
    // reproduces that, so there is no rotation to extract.
    let collapsed = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];

    let error = rotation_from_linear(collapsed).expect_err("rank 2 has no polar rotation");
    assert!(
        matches!(error, RotationExtractionError::RankDeficient { .. }),
        "error must name rank deficiency, got {error}"
    );
}

#[test]
fn a_near_singular_transform_is_rejected() {
    // Severe conditioning, not exact singularity: the smallest singular
    // value is far below the rank tolerance relative to the largest.
    let squashed = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-18]];

    assert!(matches!(
        rotation_from_linear(squashed),
        Err(RotationExtractionError::RankDeficient { .. })
    ));
}

#[test]
fn a_well_conditioned_small_scale_is_accepted() {
    // The rank test is relative to the largest singular value, so a
    // uniformly small transform must still succeed — otherwise a transform
    // in metres would behave differently from the same one in millimetres.
    let small = multiply(
        rotation_z(0.5),
        [[1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0], [0.0, 0.0, 1e-6]],
    );

    let extracted = rotation_from_linear(small).expect("uniform scale is well conditioned");
    assert_matrices_close(extracted, rotation_z(0.5), "small uniform scale");
}

#[test]
fn a_non_finite_entry_is_rejected() {
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut linear = IDENTITY;
        linear[1][2] = poison;
        assert_eq!(
            rotation_from_linear(linear),
            Err(RotationExtractionError::NonFinite),
            "non-finite {poison} must be rejected"
        );
    }
}

#[test]
fn extraction_minimizes_distance_to_the_input() {
    // The defining property: R is the closest orthogonal matrix to A in the
    // Frobenius norm. Perturbing R by any small rotation must move it away
    // from A, which distinguishes the polar factor from merely "some
    // orthogonal matrix derived from A".
    let linear = multiply(
        rotation_z(0.4),
        [[1.3, 0.1, 0.0], [0.1, 0.85, 0.0], [0.0, 0.0, 1.1]],
    );
    let extracted = rotation_from_linear(linear).expect("invertible");

    let distance = |candidate: [[f64; 3]; 3]| -> f64 {
        (0..3)
            .flat_map(|row| (0..3).map(move |column| (row, column)))
            .map(|(row, column)| {
                let difference = candidate[row][column] - linear[row][column];
                difference * difference
            })
            .sum::<f64>()
    };

    let baseline = distance(extracted);
    for perturbation in [0.01, -0.01, 0.05, -0.05] {
        let moved = multiply(extracted, rotation_z(perturbation));
        assert!(
            distance(moved) > baseline,
            "perturbing by {perturbation} must increase the distance to A"
        );
        let moved = multiply(extracted, rotation_y(perturbation));
        assert!(
            distance(moved) > baseline,
            "perturbing about y by {perturbation} must increase the distance"
        );
    }
}
