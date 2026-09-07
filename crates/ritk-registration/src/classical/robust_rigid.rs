//! Robust rigid fitting from bidirectional point correspondences.
//!
//! The estimator implements the rigid subset of the symmetric block-matching
//! update described by Modat et al. (2014), sections 2.1–2.3. Forward matches
//! are expressed fixed→moving and reverse matches moving→fixed. Each direction
//! is fitted independently with 50%-trimmed least squares. The reverse fit is
//! inverted, then the two fixed→moving transforms are averaged as
//! `exp((log(F) + log(B⁻¹)) / 2)` in transformation space.

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
/// Rotations this close to the logarithm branch cut cannot yield a stable axis.
///
/// The skew part has magnitude `sin(theta)`. Below `sqrt(epsilon)`, normalizing
/// it loses at least half the significand, so the principal logarithm fails
/// closed rather than selecting an unstable sign at `theta = pi`.
const ROTATION_LOG_BRANCH_TOLERANCE: f64 = RANK_TOLERANCE;

/// One finite fixed-to-moving physical-space correspondence.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FixedToMovingCorrespondence {
    fixed_mm: [f64; 3],
    moving_mm: [f64; 3],
}

impl FixedToMovingCorrespondence {
    /// Construct a correspondence from fixed to moving millimetres.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when either point contains
    /// a non-finite coordinate.
    pub fn try_new(fixed_mm: [f64; 3], moving_mm: [f64; 3]) -> Result<Self> {
        validate_points(fixed_mm, moving_mm, "fixed-to-moving")?;
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

/// One finite moving-to-fixed physical-space correspondence.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct MovingToFixedCorrespondence {
    moving_mm: [f64; 3],
    fixed_mm: [f64; 3],
}

impl MovingToFixedCorrespondence {
    /// Construct a correspondence from moving to fixed millimetres.
    ///
    /// The argument order follows the measured direction and therefore differs
    /// intentionally from [`FixedToMovingCorrespondence::try_new`].
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when either point contains
    /// a non-finite coordinate.
    pub fn try_new(moving_mm: [f64; 3], fixed_mm: [f64; 3]) -> Result<Self> {
        validate_points(moving_mm, fixed_mm, "moving-to-fixed")?;
        Ok(Self {
            moving_mm,
            fixed_mm,
        })
    }

    /// Return the point in moving-image physical coordinates.
    #[must_use]
    pub const fn moving_mm(self) -> [f64; 3] {
        self.moving_mm
    }

    /// Return the corresponding point in fixed-image physical coordinates.
    #[must_use]
    pub const fn fixed_mm(self) -> [f64; 3] {
        self.fixed_mm
    }
}

fn validate_points(first_mm: [f64; 3], second_mm: [f64; 3], direction: &str) -> Result<()> {
    if first_mm
        .iter()
        .chain(second_mm.iter())
        .any(|value| !value.is_finite())
    {
        return Err(RegistrationError::InvalidInput(format!(
            "{direction} rigid correspondence must be finite, got source {first_mm:?}, target {second_mm:?}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RigidCorrespondence {
    source_mm: [f64; 3],
    target_mm: [f64; 3],
}

#[derive(Debug)]
struct DirectionalFit {
    transform: AffineTransform,
    inlier_count: usize,
    squared_residual_sum: f64,
}

/// Result of a symmetric 50%-trimmed rigid fit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
#[non_exhaustive]
pub struct SymmetricRigidFit {
    /// Rigid transform mapping fixed to moving physical coordinates.
    pub transform: AffineTransform,
    /// Total number of forward and reverse correspondences fitted.
    pub correspondence_count: usize,
    /// Total number retained by the two independent 50% LTS fits.
    pub inlier_count: usize,
    /// Root-mean-square residual over both directional inlier sets, in mm.
    ///
    /// Reverse residuals are measured in fixed space before inversion. Rigid
    /// distance preservation makes their magnitudes equal in moving space.
    pub inlier_rms_mm: f64,
}

/// Fit one rigid transform to bidirectional block correspondences.
///
/// `fixed_to_moving` stores matches measured from fixed-image blocks.
/// `moving_to_fixed` stores matches measured after swapping the image order;
/// each set receives an independent LTS fit retaining the half with the
/// smallest squared residuals. The reverse transform is inverted and the two
/// fixed-to-moving transforms are combined with the log/exp mean from Modat
/// et al., equations 4–5. Repeated pairs retain their supplied weight.
///
/// The implementation stores two correspondence vectors plus per-direction
/// index and residual vectors, so peak auxiliary memory is linear in the
/// larger directional match count. Its five-refit cap is fixed by the
/// reference implementation rather than by validation-subject tuning.
///
/// # Errors
///
/// Returns [`RegistrationError::InvalidInput`] when either direction supplies
/// fewer than six correspondences, because retaining 50% must leave at least
/// three non-collinear points, or when no retained subset has sufficient rank.
/// Returns [`RegistrationError::NumericalFailure`] when SVD fails or a fitted
/// rotation reaches the numerically unresolved branch of the principal matrix
/// logarithm at 180 degrees.
///
/// # References
///
/// Modat M, et al. “Global image registration using a symmetric block-matching
/// approach.” *Journal of Medical Imaging* 1(2), 2014, sections 2.1–2.3.
/// <https://doi.org/10.1117/1.JMI.1.2.024003>
pub fn fit_symmetric_trimmed_rigid(
    fixed_to_moving: &[FixedToMovingCorrespondence],
    moving_to_fixed: &[MovingToFixedCorrespondence],
) -> Result<SymmetricRigidFit> {
    if fixed_to_moving.len() < 6 || moving_to_fixed.len() < 6 {
        return Err(RegistrationError::InvalidInput(format!(
            "50%-trimmed symmetric rigid fitting needs at least six correspondences per direction, got {} forward and {} reverse",
            fixed_to_moving.len(),
            moving_to_fixed.len()
        )));
    }

    let forward = forward_correspondences(fixed_to_moving)?;
    let reverse = reverse_correspondences(moving_to_fixed)?;
    let forward_fit = fit_trimmed_direction(&forward)?;
    let reverse_fit = fit_trimmed_direction(&reverse)?;
    let reverse_inverse = invert_rigid(&reverse_fit.transform)?;
    let transform = log_euclidean_mean(&forward_fit.transform, &reverse_inverse)?;
    let correspondence_count = fixed_to_moving
        .len()
        .checked_add(moving_to_fixed.len())
        .ok_or_else(|| {
            RegistrationError::InvalidInput(
                "bidirectional correspondence count overflows usize".to_owned(),
            )
        })?;
    let inlier_count = forward_fit
        .inlier_count
        .checked_add(reverse_fit.inlier_count)
        .ok_or_else(|| {
            RegistrationError::InvalidInput("bidirectional inlier count overflows usize".to_owned())
        })?;
    let mean_squared =
        (forward_fit.squared_residual_sum + reverse_fit.squared_residual_sum) / inlier_count as f64;
    Ok(SymmetricRigidFit {
        transform,
        correspondence_count,
        inlier_count,
        inlier_rms_mm: mean_squared.sqrt(),
    })
}

fn forward_correspondences(
    supplied: &[FixedToMovingCorrespondence],
) -> Result<Vec<RigidCorrespondence>> {
    let mut correspondences = Vec::new();
    correspondences
        .try_reserve_exact(supplied.len())
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {} forward rigid correspondences: {error}",
                supplied.len()
            ))
        })?;
    correspondences.extend(supplied.iter().map(|pair| RigidCorrespondence {
        source_mm: pair.fixed_mm,
        target_mm: pair.moving_mm,
    }));
    correspondences.sort_by(compare_correspondences);
    Ok(correspondences)
}

fn reverse_correspondences(
    supplied: &[MovingToFixedCorrespondence],
) -> Result<Vec<RigidCorrespondence>> {
    let mut correspondences = Vec::new();
    correspondences
        .try_reserve_exact(supplied.len())
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {} reverse rigid correspondences: {error}",
                supplied.len()
            ))
        })?;
    correspondences.extend(supplied.iter().map(|pair| RigidCorrespondence {
        source_mm: pair.moving_mm,
        target_mm: pair.fixed_mm,
    }));
    correspondences.sort_by(compare_correspondences);
    Ok(correspondences)
}

fn compare_correspondences(
    left: &RigidCorrespondence,
    right: &RigidCorrespondence,
) -> std::cmp::Ordering {
    left.source_mm
        .into_iter()
        .chain(left.target_mm)
        .zip(right.source_mm.into_iter().chain(right.target_mm))
        .find_map(|(left, right)| {
            let ordering = left.total_cmp(&right);
            (ordering != std::cmp::Ordering::Equal).then_some(ordering)
        })
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn fit_trimmed_direction(correspondences: &[RigidCorrespondence]) -> Result<DirectionalFit> {
    let inlier_count = correspondences.len() / 2;
    let mut active = initial_trimmed_subset(correspondences, inlier_count)?;
    for _ in 0..REFIT_LIMIT {
        let transform = fit_indices(correspondences, &active)?;
        let (_, next) = trimmed_subset(&transform, correspondences, inlier_count)?;
        if next == active {
            break;
        }
        active = next;
    }

    let transform = fit_indices(correspondences, &active)?;
    let squared_residual_sum = active
        .iter()
        .map(|&index| {
            let pair = correspondences
                .get(index)
                .expect("invariant: retained correspondence index came from this slice");
            squared_residual(&transform, pair)
        })
        .sum();
    Ok(DirectionalFit {
        transform,
        inlier_count: active.len(),
        squared_residual_sum,
    })
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
        fixed_values.extend_from_slice(&pair.source_mm);
        moving_values.extend_from_slice(&pair.target_mm);
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

fn invert_rigid(transform: &AffineTransform) -> Result<AffineTransform> {
    let matrix = transform.as_array();
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(RegistrationError::NumericalFailure(
            "cannot invert a non-finite rigid transform".to_owned(),
        ));
    }
    let translation = [matrix[3], matrix[7], matrix[11]];
    let inverse_translation = [
        -(matrix[0] * translation[0] + matrix[4] * translation[1] + matrix[8] * translation[2]),
        -(matrix[1] * translation[0] + matrix[5] * translation[1] + matrix[9] * translation[2]),
        -(matrix[2] * translation[0] + matrix[6] * translation[1] + matrix[10] * translation[2]),
    ];
    Ok(AffineTransform([
        matrix[0],
        matrix[4],
        matrix[8],
        inverse_translation[0],
        matrix[1],
        matrix[5],
        matrix[9],
        inverse_translation[1],
        matrix[2],
        matrix[6],
        matrix[10],
        inverse_translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]))
}

fn log_euclidean_mean(
    forward: &AffineTransform,
    reverse_inverse: &AffineTransform,
) -> Result<AffineTransform> {
    let forward_log = rigid_logarithm(forward)?;
    let reverse_log = rigid_logarithm(reverse_inverse)?;
    let mean = std::array::from_fn(|index| (forward_log[index] + reverse_log[index]) * 0.5);
    rigid_exponential(mean)
}

/// Return the principal `se(3)` logarithm as angular and translational parts.
fn rigid_logarithm(transform: &AffineTransform) -> Result<[f64; 6]> {
    let matrix = transform.as_array();
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(RegistrationError::NumericalFailure(
            "rigid logarithm received a non-finite transform".to_owned(),
        ));
    }

    let sine_axis = [
        (matrix[9] - matrix[6]) * 0.5,
        (matrix[2] - matrix[8]) * 0.5,
        (matrix[4] - matrix[1]) * 0.5,
    ];
    let sine = vector_norm(sine_axis);
    let cosine = ((matrix[0] + matrix[5] + matrix[10] - 1.0) * 0.5).clamp(-1.0, 1.0);
    if cosine < 0.0 && sine <= ROTATION_LOG_BRANCH_TOLERANCE {
        return Err(RegistrationError::NumericalFailure(
            "principal rigid logarithm is unresolved at the 180-degree rotation branch".to_owned(),
        ));
    }
    let angle = sine.atan2(cosine);
    let angular = if sine == 0.0 {
        [0.0; 3]
    } else {
        let scale = angle / sine;
        sine_axis.map(|value| value * scale)
    };
    let translation = [matrix[3], matrix[7], matrix[11]];
    let angular_squared = dot(angular, angular);
    let jacobian_inverse_coefficient = if angular_squared <= RANK_TOLERANCE {
        // Taylor series for
        // `(1 - theta/2 * cot(theta/2)) / theta^2` avoids cancellation.
        1.0 / 12.0 + angular_squared / 720.0 + angular_squared.powi(2) / 30_240.0
    } else {
        let half_angle = angle * 0.5;
        (1.0 - half_angle * (half_angle.cos() / half_angle.sin())) / angular_squared
    };
    let cross_once = cross(angular, translation);
    let cross_twice = cross(angular, cross_once);
    let tangent_translation: [f64; 3] = std::array::from_fn(|axis| {
        translation[axis] - 0.5 * cross_once[axis]
            + jacobian_inverse_coefficient * cross_twice[axis]
    });
    let logarithm = [
        angular[0],
        angular[1],
        angular[2],
        tangent_translation[0],
        tangent_translation[1],
        tangent_translation[2],
    ];
    if logarithm.iter().all(|value| value.is_finite()) {
        Ok(logarithm)
    } else {
        Err(RegistrationError::NumericalFailure(
            "rigid logarithm produced a non-finite tangent vector".to_owned(),
        ))
    }
}

/// Exponentiate an `se(3)` tangent vector with stable small-angle series.
fn rigid_exponential(tangent: [f64; 6]) -> Result<AffineTransform> {
    if !tangent.iter().all(|value| value.is_finite()) {
        return Err(RegistrationError::NumericalFailure(
            "rigid exponential received a non-finite tangent vector".to_owned(),
        ));
    }
    let angular = [tangent[0], tangent[1], tangent[2]];
    let tangent_translation = [tangent[3], tangent[4], tangent[5]];
    let angle_squared = dot(angular, angular);
    let (rotation_linear, rotation_quadratic, translation_quadratic) =
        if angle_squared <= RANK_TOLERANCE {
            let angle_fourth = angle_squared * angle_squared;
            (
                1.0 - angle_squared / 6.0 + angle_fourth / 120.0,
                0.5 - angle_squared / 24.0 + angle_fourth / 720.0,
                1.0 / 6.0 - angle_squared / 120.0 + angle_fourth / 5_040.0,
            )
        } else {
            let angle = angle_squared.sqrt();
            (
                angle.sin() / angle,
                (1.0 - angle.cos()) / angle_squared,
                (angle - angle.sin()) / (angle_squared * angle),
            )
        };
    let [x, y, z] = angular;
    let angular_square = [
        [-(y * y + z * z), x * y, x * z],
        [x * y, -(x * x + z * z), y * z],
        [x * z, y * z, -(x * x + y * y)],
    ];
    let angular_matrix = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]];
    let rotation: [[f64; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (if row == column { 1.0 } else { 0.0 })
                + rotation_linear * angular_matrix[row][column]
                + rotation_quadratic * angular_square[row][column]
        })
    });
    let cross_once = cross(angular, tangent_translation);
    let cross_twice = cross(angular, cross_once);
    let translation: [f64; 3] = std::array::from_fn(|axis| {
        tangent_translation[axis]
            + rotation_quadratic * cross_once[axis]
            + translation_quadratic * cross_twice[axis]
    });
    let transform = AffineTransform([
        rotation[0][0],
        rotation[0][1],
        rotation[0][2],
        translation[0],
        rotation[1][0],
        rotation[1][1],
        rotation[1][2],
        translation[1],
        rotation[2][0],
        rotation[2][1],
        rotation[2][2],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    if transform.as_array().iter().all(|value| value.is_finite()) {
        Ok(transform)
    } else {
        Err(RegistrationError::NumericalFailure(
            "rigid exponential produced a non-finite transform".to_owned(),
        ))
    }
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn squared_residual(transform: &AffineTransform, pair: &RigidCorrespondence) -> f64 {
    let matrix = transform.as_array();
    let mapped = [
        matrix[0] * pair.source_mm[0]
            + matrix[1] * pair.source_mm[1]
            + matrix[2] * pair.source_mm[2]
            + matrix[3],
        matrix[4] * pair.source_mm[0]
            + matrix[5] * pair.source_mm[1]
            + matrix[6] * pair.source_mm[2]
            + matrix[7],
        matrix[8] * pair.source_mm[0]
            + matrix[9] * pair.source_mm[1]
            + matrix[10] * pair.source_mm[2]
            + matrix[11],
    ];
    mapped
        .iter()
        .zip(pair.target_mm.iter())
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum()
}

#[cfg(test)]
#[path = "robust_rigid_tests.rs"]
mod tests;
