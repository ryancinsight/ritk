use super::super::error::{RegistrationError, Result};
use super::correspondence::{DirectionalFit, RigidCorrespondence};
use super::limits::{EXACT_CANDIDATE_LIMIT, REFIT_LIMIT, SAMPLED_CANDIDATE_LIMIT};
use super::transform::{fit_indices, squared_residual};
use crate::types::AffineTransform;

pub(super) fn fit_trimmed_direction(
    correspondences: &[RigidCorrespondence],
) -> Result<DirectionalFit> {
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
