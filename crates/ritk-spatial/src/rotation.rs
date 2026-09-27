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
mod tests;
