//! Robust rigid fitting from bidirectional point correspondences.
//!
//! The estimator implements the rigid subset of the symmetric block-matching
//! update described by Modat et al. (2014), sections 2.1–2.3. Forward matches
//! are expressed fixed→moving and reverse matches moving→fixed. Reversing the
//! latter before one joint least-trimmed-squares fit is order-invariant for a
//! rigid transform because rotations preserve Euclidean residual norms.

use leto::{Array2, FixedMatrix, FixedVector};

use super::error::{RegistrationError, Result};
use super::spatial::{build_homogeneous_matrix, center_points, compute_centroid, kabsch_algorithm};
use crate::types::AffineTransform;

type Matrix3 = FixedMatrix<f64, 3, 3>;
type Vector3 = FixedVector<f64, 3>;

/// Maximum least-trimmed-squares refits used by NiftyReg's `reg_aladin`.
const REFIT_LIMIT: usize = 5;
/// Exact elemental candidates remain bounded for small correspondence sets.
const EXACT_CANDIDATE_LIMIT: usize = 4_096;
/// Deterministic elemental candidates for larger sets.
///
/// At the limiting 50% inlier fraction, 1,024 independent three-point draws
/// miss an all-inlier subset with probability `(7/8)^1024 < f64::EPSILON^2`.
/// The deterministic sequence makes registration reproducible; the bound
/// explains its breadth but is not claimed as a probabilistic guarantee for
/// adversarially ordered input.
const SAMPLED_CANDIDATE_LIMIT: usize = 1_024;
/// `sqrt(f64::EPSILON)`, used as a relative rank threshold for 3-D point sets.
const RANK_TOLERANCE: f64 = 1.490_116_119_384_765_6e-8;

/// One finite physical-space correspondence.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RigidCorrespondence {
    fixed_mm: [f64; 3],
    moving_mm: [f64; 3],
}

impl RigidCorrespondence {
    /// Construct a correspondence in millimetres.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when either point contains
    /// a non-finite coordinate.
    pub fn try_new(fixed_mm: [f64; 3], moving_mm: [f64; 3]) -> Result<Self> {
        if fixed_mm
            .iter()
            .chain(moving_mm.iter())
            .any(|value| !value.is_finite())
        {
            return Err(RegistrationError::InvalidInput(format!(
                "rigid correspondence must be finite, got fixed {fixed_mm:?}, moving {moving_mm:?}"
            )));
        }
        Ok(Self {
            fixed_mm,
            moving_mm,
        })
    }

    /// Return the point in fixed-image physical coordinates.
    #[must_use]
    pub const fn fixed_mm(self) -> [f64; 3] {
        self.fixed_mm
    }

    /// Return the corresponding point in moving-image physical coordinates.
    #[must_use]
    pub const fn moving_mm(self) -> [f64; 3] {
        self.moving_mm
    }
}

/// Result of a symmetric 50%-trimmed rigid fit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
#[non_exhaustive]
pub struct SymmetricRigidFit {
    /// Rigid transform mapping fixed to moving physical coordinates.
    pub transform: AffineTransform,
    /// Total number of forward and reverse correspondences considered.
    pub correspondence_count: usize,
    /// Number of correspondences retained by the 50% LTS rule.
    pub inlier_count: usize,
    /// Root-mean-square residual over the retained correspondences, in mm.
    pub inlier_rms_mm: f64,
}

/// Fit one rigid transform to bidirectional block correspondences.
///
/// `fixed_to_moving` stores matches measured from fixed-image blocks.
/// `moving_to_fixed` stores matches measured after swapping the image order;
/// this function reverses those pairs before fitting. At each iteration the
/// half with the smallest squared residual is retained, matching the 50% block
/// and inlier policy used by Modat et al. and NiftyReg `reg_aladin`.
///
/// The implementation stores one correspondence vector, one index vector, and
/// one residual vector, so auxiliary memory is linear in the supplied match
/// count. Its five-refit cap is fixed by the reference implementation rather
/// than by validation-subject tuning.
///
/// # Errors
///
/// Returns [`RegistrationError::InvalidInput`] when either direction supplies
/// fewer than three correspondences or the retained points are collinear.
/// Numerical SVD failures are propagated with their source context.
///
/// # References
///
/// Modat M, et al. “Global image registration using a symmetric block-matching
/// approach.” *Journal of Medical Imaging* 1(2), 2014, sections 2.1–2.3.
/// <https://doi.org/10.1117/1.JMI.1.2.024003>
pub fn fit_symmetric_trimmed_rigid(
    fixed_to_moving: &[RigidCorrespondence],
    moving_to_fixed: &[RigidCorrespondence],
) -> Result<SymmetricRigidFit> {
    if fixed_to_moving.len() < 3 || moving_to_fixed.len() < 3 {
        return Err(RegistrationError::InvalidInput(format!(
            "symmetric rigid fitting needs at least three correspondences per direction, got {} forward and {} reverse",
            fixed_to_moving.len(),
            moving_to_fixed.len()
        )));
    }

    let correspondence_count = fixed_to_moving
        .len()
        .checked_add(moving_to_fixed.len())
        .ok_or_else(|| {
            RegistrationError::InvalidInput(
                "bidirectional correspondence count overflows usize".to_owned(),
            )
        })?;
    let mut correspondences = Vec::new();
    correspondences
        .try_reserve_exact(correspondence_count)
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {correspondence_count} rigid correspondences: {error}"
            ))
        })?;
    correspondences.extend_from_slice(fixed_to_moving);
    correspondences.extend(moving_to_fixed.iter().map(|pair| RigidCorrespondence {
        fixed_mm: pair.moving_mm,
        moving_mm: pair.fixed_mm,
    }));
    correspondences.sort_by(compare_correspondences);

    let inlier_count = correspondence_count / 2;
    let mut active = initial_trimmed_subset(&correspondences, inlier_count)?;
    for _ in 0..REFIT_LIMIT {
        let transform = fit_indices(&correspondences, &active)?;
        let (_, next) = trimmed_subset(&transform, &correspondences, inlier_count)?;
        if next == active {
            break;
        }
        active = next;
    }

    let transform = fit_indices(&correspondences, &active)?;
    let mean_squared = active
        .iter()
        .map(|&index| {
            let pair = correspondences
                .get(index)
                .expect("invariant: retained correspondence index came from this slice");
            squared_residual(&transform, pair)
        })
        .sum::<f64>()
        / active.len() as f64;
    Ok(SymmetricRigidFit {
        transform,
        correspondence_count,
        inlier_count: active.len(),
        inlier_rms_mm: mean_squared.sqrt(),
    })
}

fn compare_correspondences(
    left: &RigidCorrespondence,
    right: &RigidCorrespondence,
) -> std::cmp::Ordering {
    left.fixed_mm
        .into_iter()
        .chain(left.moving_mm)
        .zip(right.fixed_mm.into_iter().chain(right.moving_mm))
        .find_map(|(left, right)| {
            let ordering = left.total_cmp(&right);
            (ordering != std::cmp::Ordering::Equal).then_some(ordering)
        })
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn initial_trimmed_subset(
    correspondences: &[RigidCorrespondence],
    inlier_count: usize,
) -> Result<Vec<usize>> {
    let candidate_count = combination_count_capped(correspondences.len(), EXACT_CANDIDATE_LIMIT);
    let mut best: Option<(f64, Vec<usize>)> = None;

    if candidate_count <= EXACT_CANDIDATE_LIMIT {
        for first in 0..correspondences.len().saturating_sub(2) {
            for second in (first + 1)..correspondences.len().saturating_sub(1) {
                for third in (second + 1)..correspondences.len() {
                    consider_candidate(
                        correspondences,
                        [first, second, third],
                        inlier_count,
                        &mut best,
                    )?;
                }
            }
        }
    } else {
        for candidate in 0..SAMPLED_CANDIDATE_LIMIT {
            let indices = sampled_triplet(correspondences.len(), candidate);
            consider_candidate(correspondences, indices, inlier_count, &mut best)?;
        }
    }

    best.map(|(_, indices)| indices).ok_or_else(|| {
        RegistrationError::InvalidInput(
            "rigid correspondences contain no non-collinear elemental subset".to_owned(),
        )
    })
}

fn consider_candidate(
    correspondences: &[RigidCorrespondence],
    indices: [usize; 3],
    inlier_count: usize,
    best: &mut Option<(f64, Vec<usize>)>,
) -> Result<()> {
    let transform = match fit_indices(correspondences, &indices) {
        Ok(transform) => transform,
        Err(RegistrationError::InvalidInput(_)) => return Ok(()),
        Err(error) => return Err(error),
    };
    let (score, subset) = trimmed_subset(&transform, correspondences, inlier_count)?;
    let replaces = best.as_ref().is_none_or(|(best_score, best_subset)| {
        score.total_cmp(best_score).is_lt()
            || (score.total_cmp(best_score).is_eq() && subset < *best_subset)
    });
    if replaces {
        *best = Some((score, subset));
    }
    Ok(())
}

fn trimmed_subset(
    transform: &AffineTransform,
    correspondences: &[RigidCorrespondence],
    inlier_count: usize,
) -> Result<(f64, Vec<usize>)> {
    let mut residuals = Vec::new();
    residuals
        .try_reserve_exact(correspondences.len())
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {} rigid residuals: {error}",
                correspondences.len()
            ))
        })?;
    for (index, pair) in correspondences.iter().enumerate() {
        let residual = squared_residual(transform, pair);
        if !residual.is_finite() {
            return Err(RegistrationError::NumericalFailure(
                "rigid correspondence residual is non-finite".to_owned(),
            ));
        }
        residuals.push((residual, index));
    }
    residuals.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let score = residuals
        .iter()
        .take(inlier_count)
        .map(|&(residual, _)| residual)
        .sum();
    let mut subset: Vec<usize> = residuals
        .iter()
        .take(inlier_count)
        .map(|&(_, index)| index)
        .collect();
    subset.sort_unstable();
    Ok((score, subset))
}

fn combination_count_capped(count: usize, cap: usize) -> usize {
    let Some(first) = count.checked_mul(count.saturating_sub(1)) else {
        return cap.saturating_add(1);
    };
    let Some(product) = first.checked_mul(count.saturating_sub(2)) else {
        return cap.saturating_add(1);
    };
    (product / 6).min(cap.saturating_add(1))
}

fn sampled_triplet(count: usize, candidate: usize) -> [usize; 3] {
    let seed = u64::try_from(candidate).unwrap_or(u64::MAX);
    let mut indices = [
        sample_index(splitmix64(seed.wrapping_mul(3)), count),
        sample_index(splitmix64(seed.wrapping_mul(3).wrapping_add(1)), count),
        sample_index(splitmix64(seed.wrapping_mul(3).wrapping_add(2)), count),
    ];
    while indices[1] == indices[0] {
        indices[1] = (indices[1] + 1) % count;
    }
    while indices[2] == indices[0] || indices[2] == indices[1] {
        indices[2] = (indices[2] + 1) % count;
    }
    indices.sort_unstable();
    indices
}

fn sample_index(value: u64, count: usize) -> usize {
    let count = u64::try_from(count).unwrap_or(u64::MAX);
    usize::try_from(value % count).unwrap_or(usize::MAX)
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn fit_indices(
    correspondences: &[RigidCorrespondence],
    indices: &[usize],
) -> Result<AffineTransform> {
    if indices.len() < 3 {
        return Err(RegistrationError::InvalidInput(format!(
            "rigid fitting needs at least three retained points, got {}",
            indices.len()
        )));
    }
    let value_count = indices.len().checked_mul(3).ok_or_else(|| {
        RegistrationError::InvalidInput("rigid coordinate count overflows usize".to_owned())
    })?;
    let mut fixed_values = Vec::new();
    fixed_values
        .try_reserve_exact(value_count)
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {value_count} fixed rigid coordinates: {error}"
            ))
        })?;
    let mut moving_values = Vec::new();
    moving_values
        .try_reserve_exact(value_count)
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {value_count} moving rigid coordinates: {error}"
            ))
        })?;
    for &index in indices {
        let pair = correspondences
            .get(index)
            .expect("invariant: retained correspondence index came from this slice");
        fixed_values.extend_from_slice(&pair.fixed_mm);
        moving_values.extend_from_slice(&pair.moving_mm);
    }
    let fixed = Array2::from_vec([indices.len(), 3], fixed_values).map_err(|error| {
        RegistrationError::NumericalFailure(format!(
            "cannot lay out fixed rigid correspondences: {error}"
        ))
    })?;
    let moving = Array2::from_vec([indices.len(), 3], moving_values).map_err(|error| {
        RegistrationError::NumericalFailure(format!(
            "cannot lay out moving rigid correspondences: {error}"
        ))
    })?;
    let fixed_centroid = compute_centroid(&fixed);
    let moving_centroid = compute_centroid(&moving);
    let fixed_centered = center_points(&fixed, &fixed_centroid);
    let moving_centered = center_points(&moving, &moving_centroid);
    ensure_non_collinear(&fixed_centered, "fixed")?;
    ensure_non_collinear(&moving_centered, "moving")?;

    // `kabsch_algorithm(target, source)` maps source to target. The public
    // correspondence convention here is fixed→moving.
    let rotation = kabsch_algorithm(&moving_centered, &fixed_centered)?;
    let matrix = Matrix3::from_rows([
        [rotation[0], rotation[1], rotation[2]],
        [rotation[3], rotation[4], rotation[5]],
        [rotation[6], rotation[7], rotation[8]],
    ]);
    let translation = moving_centroid - matrix * fixed_centroid;
    let transform =
        build_homogeneous_matrix(&rotation, &[translation[0], translation[1], translation[2]]);
    if transform.as_array().iter().all(|value| value.is_finite()) {
        Ok(transform)
    } else {
        Err(RegistrationError::NumericalFailure(
            "rigid fit produced a non-finite transform".to_owned(),
        ))
    }
}

fn ensure_non_collinear(points: &Array2<f64>, context: &str) -> Result<()> {
    let mut covariance = Matrix3::zeros();
    for row in 0..points.shape()[0] {
        let point = Vector3::new([
            *points
                .get([row, 0])
                .expect("invariant: three-column point array"),
            *points
                .get([row, 1])
                .expect("invariant: three-column point array"),
            *points
                .get([row, 2])
                .expect("invariant: three-column point array"),
        ]);
        covariance += Matrix3::from_rows([
            [
                point[0] * point[0],
                point[0] * point[1],
                point[0] * point[2],
            ],
            [
                point[1] * point[0],
                point[1] * point[1],
                point[1] * point[2],
            ],
            [
                point[2] * point[0],
                point[2] * point[1],
                point[2] * point[2],
            ],
        ]);
    }
    let trace = covariance[(0, 0)] + covariance[(1, 1)] + covariance[(2, 2)];
    let frobenius_squared = (0..3)
        .flat_map(|row| (0..3).map(move |column| covariance[(row, column)].powi(2)))
        .sum::<f64>();
    // For PSD covariance with eigenvalues λᵢ, this is Σᵢ<ⱼ λᵢλⱼ.
    // It is zero exactly for rank < 2; sqrt(epsilon) rejects numerically
    // unresolved second axes without assigning a dimensional scale.
    let second_elementary = ((trace * trace - frobenius_squared) * 0.5).max(0.0);
    if trace <= 0.0 || second_elementary <= RANK_TOLERANCE * trace * trace {
        return Err(RegistrationError::InvalidInput(format!(
            "{context} rigid correspondences are collinear or numerically rank deficient"
        )));
    }
    Ok(())
}

fn squared_residual(transform: &AffineTransform, pair: &RigidCorrespondence) -> f64 {
    let matrix = transform.as_array();
    let mapped = [
        matrix[0] * pair.fixed_mm[0]
            + matrix[1] * pair.fixed_mm[1]
            + matrix[2] * pair.fixed_mm[2]
            + matrix[3],
        matrix[4] * pair.fixed_mm[0]
            + matrix[5] * pair.fixed_mm[1]
            + matrix[6] * pair.fixed_mm[2]
            + matrix[7],
        matrix[8] * pair.fixed_mm[0]
            + matrix[9] * pair.fixed_mm[1]
            + matrix[10] * pair.fixed_mm[2]
            + matrix[11],
    ];
    mapped
        .iter()
        .zip(pair.moving_mm.iter())
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum()
}

#[cfg(test)]
#[path = "robust_rigid_tests.rs"]
mod tests;
