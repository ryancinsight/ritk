//! Extracting the rotation from a general linear transform.
//!
//! # Why this is not the upper-left 3×3
//!
//! Registration produces affine transforms. Their linear part `A` mixes
//! rotation with scale and shear, and several operations need the rotation
//! alone: reorienting diffusion gradient directions after motion correction,
//! reorienting tensors and orientation distribution functions after a warp,
//! recovering the anatomical orientation of a resampled grid.
//!
//! Using `A` directly for those is wrong in a way that produces no error. A
//! gradient direction scaled by an eddy-current shear is no longer a unit
//! vector, so it silently reweights the acquisition; a tensor rotated by `A`
//! acquires the transform's anisotropy on top of the tissue's.
//!
//! # The polar factor
//!
//! Every invertible `A` factors uniquely as `A = R S`, with `R` orthogonal and
//! `S` symmetric positive definite. `R` is the orthogonal matrix closest to `A`
//! in the Frobenius norm, which is exactly "the rotation `A` performs, with its
//! stretching removed".
//!
//! `S = (AᵀA)^{1/2}` follows from `AᵀA = SᵀRᵀR S = S²`, and `S` is recovered
//! from the symmetric eigendecomposition `AᵀA = Q Λ Qᵀ`:
//!
//! ```text
//! S⁻¹ = Q Λ^{-1/2} Qᵀ        R = A S⁻¹
//! ```
//!
//! `AᵀA` is symmetric positive definite whenever `A` is invertible, so the
//! square root is real and the decomposition is well posed. The `√λᵢ` are the
//! singular values of `A`.
//!
//! # What is rejected rather than repaired
//!
//! A reflection is refused instead of being corrected to the nearest proper
//! rotation. The Kabsch-style sign flip is right when fitting a rotation to
//! noisy point correspondences, where a reflected fit is a fitting artifact.
//! It is wrong here: a registration between two images of one subject cannot
//! legitimately reverse handedness, so `det(A) < 0` means the transform is
//! wrong, and quietly repairing it would hide the defect behind a plausible
//! result.
//!
//! # Reference
//!
//! Higham, "Computing the polar decomposition — with applications", *SIAM
//! Journal on Scientific and Statistical Computing* 7(4), 1986, §1 — the
//! existence and uniqueness of `A = R S` and the nearest-orthogonal-matrix
//! property `R` satisfies.

use leto::FixedMatrix;

/// Failure modes of rotation extraction.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RotationExtractionError {
    /// The linear part contains a non-finite value.
    ///
    /// A failed registration can emit NaN in its transform; propagating it
    /// would poison every reoriented vector without an error.
    #[error("linear transform contains a non-finite value")]
    NonFinite,

    /// The linear part is singular or numerically rank deficient.
    ///
    /// A collapsed axis has no recoverable rotation: directions in the null
    /// space map to zero, so no orthogonal matrix reproduces `A`'s action.
    #[error(
        "linear transform is rank deficient: smallest singular value {smallest} \
         is below the rank tolerance {tolerance}"
    )]
    RankDeficient {
        /// Smallest singular value of the linear part.
        smallest: f64,
        /// Rank tolerance the value fell below.
        tolerance: f64,
    },

    /// The linear part reverses orientation.
    ///
    /// Refused rather than repaired — see the module documentation.
    #[error("linear transform reverses orientation: determinant is {determinant}")]
    OrientationReversing {
        /// Determinant of the linear part.
        determinant: f64,
    },
}

/// Rank tolerance scale for a 3×3 matrix, in units of the largest singular
/// value.
///
/// The standard LAPACK-style rank criterion is `max(rows, columns) · ε · σ_max`;
/// a singular value below it is indistinguishable from zero at working
/// precision. For a 3×3 that is `3 ε`.
const RANK_TOLERANCE_SCALE: f64 = 3.0 * f64::EPSILON;

/// The rotation `R` from the polar decomposition `A = R S` of `linear`.
///
/// `linear` is row-major: `linear[row][column]`. The returned matrix is
/// orthonormal with determinant `+1`, and is the closest such matrix to
/// `linear` in the Frobenius norm.
///
/// # Errors
///
/// [`RotationExtractionError::NonFinite`] for a non-finite entry,
/// [`RotationExtractionError::RankDeficient`] when a singular value falls below
/// the rank tolerance, and [`RotationExtractionError::OrientationReversing`]
/// when the determinant is negative.
///
/// # Examples
///
/// ```
/// use ritk_spatial::rotation::rotation_from_linear;
///
/// // A quarter turn about z, scaled by 2 along x and sheared.
/// let linear = [[0.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
/// let rotation = rotation_from_linear(linear).expect("invertible");
///
/// // The scale is removed; the rotation remains.
/// assert!((rotation[0][1] + 1.0).abs() < 1e-12);
/// assert!((rotation[1][0] - 1.0).abs() < 1e-12);
/// ```
pub fn rotation_from_linear(
    linear: [[f64; 3]; 3],
) -> Result<[[f64; 3]; 3], RotationExtractionError> {
    if linear.iter().flatten().any(|value| !value.is_finite()) {
        return Err(RotationExtractionError::NonFinite);
    }

    let matrix = to_fixed(linear);
    let determinant = matrix.determinant();
    if determinant < 0.0 {
        return Err(RotationExtractionError::OrientationReversing { determinant });
    }

    // AᵀA is symmetric positive semi-definite; its eigenvalues are the squared
    // singular values of A, returned in descending order.
    let gram = matrix.transpose() * matrix;
    let (eigenvalues, eigenvectors) = gram.symmetric_eigen();

    let largest_singular = eigenvalues[0].max(0.0).sqrt();
    let smallest_singular = eigenvalues[2].max(0.0).sqrt();
    let tolerance = RANK_TOLERANCE_SCALE * largest_singular;
    if smallest_singular <= tolerance {
        return Err(RotationExtractionError::RankDeficient {
            smallest: smallest_singular,
            tolerance,
        });
    }

    // S⁻¹ = Q Λ^{-1/2} Qᵀ, formed by scaling each eigenvector column by the
    // reciprocal of its singular value before multiplying by Qᵀ.
    let mut scaled = FixedMatrix::<f64, 3, 3>::zeros();
    for column in 0..3 {
        let inverse_singular = 1.0 / eigenvalues[column].max(0.0).sqrt();
        for row in 0..3 {
            scaled[(row, column)] = eigenvectors[(row, column)] * inverse_singular;
        }
    }
    let inverse_stretch = scaled * eigenvectors.transpose();

    Ok(from_fixed(refine(matrix * inverse_stretch)))
}

/// One Newton step of the polar iteration, projecting `candidate` back onto the
/// orthogonal manifold.
///
/// The eigen route above is exact in principle but loses accuracy when `AᵀA`
/// has repeated eigenvalues, because the eigenvectors of a degenerate matrix
/// are not determined individually — the analytic cubic then computes them from
/// cross products of a near-zero matrix. That degeneracy is not an edge case:
/// it is exactly what an undistorted transform produces, since `AᵀA = I` when
/// `A` is already a rotation.
///
/// Higham's iteration `X ← ½(X + X⁻ᵀ)` converges quadratically to the
/// orthogonal polar factor and is a fixed point at an exactly orthogonal `X`.
/// Applied to a candidate already accurate to `δ`, one step delivers `δ²` —
/// machine precision for any `δ` the eigen route can produce.
fn refine(candidate: FixedMatrix<f64, 3, 3>) -> FixedMatrix<f64, 3, 3> {
    let Some(inverse_transpose) = inverse_transpose(candidate) else {
        // Unreachable for a candidate derived from a full-rank matrix; leaving
        // it unrefined is correct rather than failing, since the caller's rank
        // check has already passed.
        return candidate;
    };

    let mut refined = FixedMatrix::<f64, 3, 3>::zeros();
    for row in 0..3 {
        for column in 0..3 {
            refined[(row, column)] =
                0.5 * (candidate[(row, column)] + inverse_transpose[(row, column)]);
        }
    }
    refined
}

/// `X⁻ᵀ` for a 3×3 matrix, via the closed-form adjugate.
///
/// The inverse of a 3×3 is its adjugate over its determinant, and the transpose
/// of that is the adjugate's transpose over the same determinant. This is
/// arithmetic rather than a solve, so it stays local instead of routing through
/// a decomposition.
fn inverse_transpose(matrix: FixedMatrix<f64, 3, 3>) -> Option<FixedMatrix<f64, 3, 3>> {
    let determinant = matrix.determinant();
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }

    // Cofactor (row, column) is the signed 2×2 minor. The inverse is
    // adjugateᵀ/det, so the inverse-transpose is adjugate/det — that is, the
    // cofactor matrix itself over the determinant.
    let mut result = FixedMatrix::<f64, 3, 3>::zeros();
    for row in 0..3 {
        for column in 0..3 {
            let rows: Vec<usize> = (0..3).filter(|index| *index != row).collect();
            let columns: Vec<usize> = (0..3).filter(|index| *index != column).collect();
            let minor = matrix[(rows[0], columns[0])] * matrix[(rows[1], columns[1])]
                - matrix[(rows[0], columns[1])] * matrix[(rows[1], columns[0])];
            let sign = if (row + column) % 2 == 0 { 1.0 } else { -1.0 };
            result[(row, column)] = sign * minor / determinant;
        }
    }
    Some(result)
}

fn to_fixed(values: [[f64; 3]; 3]) -> FixedMatrix<f64, 3, 3> {
    let mut matrix = FixedMatrix::<f64, 3, 3>::zeros();
    for row in 0..3 {
        for column in 0..3 {
            matrix[(row, column)] = values[row][column];
        }
    }
    matrix
}

fn from_fixed(matrix: FixedMatrix<f64, 3, 3>) -> [[f64; 3]; 3] {
    let mut values = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            values[row][column] = matrix[(row, column)];
        }
    }
    values
}

#[cfg(test)]
mod tests {
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
}
