//! Deterministic fixed-domain center selection.

use std::collections::BinaryHeap;

use super::config::{MindSscSampling, SamplingSelection};
use super::MindSscError;

const SAMPLING_HASH_SEED: u64 = u64::from_be_bytes(*b"MIND-SSC");

pub(super) fn select_indices(
    policy: &MindSscSampling,
    shape: [usize; 3],
    halo: [usize; 3],
    spacing: [f64; 3],
    mask: Option<&[bool]>,
) -> Result<Vec<usize>, MindSscError> {
    let voxel_count = checked_voxel_count(shape)?;
    if let Some(mask) = mask {
        if mask.len() != voxel_count {
            return Err(MindSscError::MaskLength {
                expected: voxel_count,
                actual: mask.len(),
            });
        }
    }
    validate_shape(shape, halo)?;

    let indices = match policy.selection() {
        SamplingSelection::Indices(indices) => indices
            .iter()
            .copied()
            .map(|index| validate_caller_index(index, voxel_count, shape, halo, mask))
            .collect(),
        SamplingSelection::Dense => collect_eligible(shape, halo, mask),
        SamplingSelection::Stratified { max_samples } => {
            let eligible = count_eligible(shape, halo, mask)?;
            if eligible == 0 {
                return Err(MindSscError::EmptyFixedDomain);
            }
            if eligible <= *max_samples {
                collect_eligible(shape, halo, mask)
            } else {
                select_hash_ranked_strata(shape, halo, spacing, mask, eligible, *max_samples)
            }
        }
    }?;
    if indices.is_empty() {
        Err(MindSscError::EmptyFixedDomain)
    } else {
        Ok(indices)
    }
}

pub(super) fn checked_voxel_count(shape: [usize; 3]) -> Result<usize, MindSscError> {
    shape
        .into_iter()
        .try_fold(1_usize, |count, extent| count.checked_mul(extent))
        .ok_or(MindSscError::IndexOverflow)
}

fn validate_shape(shape: [usize; 3], halo: [usize; 3]) -> Result<(), MindSscError> {
    let supports = (0..3).all(|axis| {
        halo[axis]
            .checked_mul(2)
            .and_then(|width| width.checked_add(1))
            .is_some_and(|minimum| shape[axis] >= minimum)
    });
    if supports {
        Ok(())
    } else {
        Err(MindSscError::ImageTooSmall { shape, halo })
    }
}

fn validate_caller_index(
    index: usize,
    voxel_count: usize,
    shape: [usize; 3],
    halo: [usize; 3],
    mask: Option<&[bool]>,
) -> Result<usize, MindSscError> {
    if index >= voxel_count {
        return Err(MindSscError::SampleIndexOutOfBounds { index, voxel_count });
    }
    let center = decode_index(index, shape)?;
    if !has_complete_support(center, shape, halo) {
        return Err(MindSscError::SampleOutsideCompleteSupport { index });
    }
    if mask.is_some_and(|values| !values[index]) {
        return Err(MindSscError::SampleExcludedByMask { index });
    }
    Ok(index)
}

fn count_eligible(
    shape: [usize; 3],
    halo: [usize; 3],
    mask: Option<&[bool]>,
) -> Result<usize, MindSscError> {
    let mut count = 0_usize;
    visit_interior(shape, halo, |index| {
        if mask.is_none_or(|values| values[index]) {
            count = count.checked_add(1).ok_or(MindSscError::IndexOverflow)?;
        }
        Ok(())
    })?;
    Ok(count)
}

fn collect_eligible(
    shape: [usize; 3],
    halo: [usize; 3],
    mask: Option<&[bool]>,
) -> Result<Vec<usize>, MindSscError> {
    let mut selected = Vec::new();
    visit_interior(shape, halo, |index| {
        if mask.is_none_or(|values| values[index]) {
            selected.push(index);
        }
        Ok(())
    })?;
    Ok(selected)
}

fn select_hash_ranked_strata(
    shape: [usize; 3],
    halo: [usize; 3],
    spacing: [f64; 3],
    mask: Option<&[bool]>,
    eligible: usize,
    budget: usize,
) -> Result<Vec<usize>, MindSscError> {
    let interior = std::array::from_fn(|axis| shape[axis] - 2 * halo[axis]);
    let strata_shape = spatial_strata_shape(interior, spacing, budget)?;
    let strata_count = checked_voxel_count(strata_shape)?;
    let mut populations = vec![0_usize; strata_count];
    visit_interior(shape, halo, |index| {
        if mask.is_some_and(|values| !values[index]) {
            return Ok(());
        }
        let stratum = stratum_for_index(index, shape, halo, interior, strata_shape)?;
        populations[stratum] = populations[stratum]
            .checked_add(1)
            .ok_or(MindSscError::IndexOverflow)?;
        Ok(())
    })?;

    let quotas = proportional_quotas(&populations, eligible, budget)?;
    let mut ranked = quotas
        .iter()
        .map(|quota| BinaryHeap::with_capacity(*quota))
        .collect::<Vec<BinaryHeap<(u64, usize)>>>();
    visit_interior(shape, halo, |index| {
        if mask.is_some_and(|values| !values[index]) {
            return Ok(());
        }
        let stratum = stratum_for_index(index, shape, halo, interior, strata_shape)?;
        let quota = quotas[stratum];
        if quota == 0 {
            return Ok(());
        }
        let candidate = (sample_hash(index)?, index);
        let heap = &mut ranked[stratum];
        if heap.len() < quota {
            heap.push(candidate);
        } else if heap.peek().is_some_and(|largest| candidate < *largest) {
            heap.pop();
            heap.push(candidate);
        }
        Ok(())
    })?;

    let mut selected = ranked
        .into_iter()
        .flat_map(BinaryHeap::into_iter)
        .map(|(_, index)| index)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    if selected.len() != budget {
        return Err(MindSscError::IndexOverflow);
    }
    Ok(selected)
}

fn spatial_strata_shape(
    interior: [usize; 3],
    spacing: [f64; 3],
    budget: usize,
) -> Result<[usize; 3], MindSscError> {
    let physical_extent = [
        usize_to_f64(interior[0])? * spacing[0],
        usize_to_f64(interior[1])? * spacing[1],
        usize_to_f64(interior[2])? * spacing[2],
    ];
    let mut shape = [1_usize; 3];
    let mut count = 1_usize;
    loop {
        let axis = (0..3)
            .filter(|axis| {
                shape[*axis]
                    .checked_mul(2)
                    .is_some_and(|split| split <= interior[*axis])
                    && count.checked_mul(2).is_some_and(|next| next <= budget)
            })
            .max_by(|left, right| {
                let left_span = physical_extent[*left] / usize_to_f64_exact(shape[*left]);
                let right_span = physical_extent[*right] / usize_to_f64_exact(shape[*right]);
                left_span
                    .total_cmp(&right_span)
                    .then_with(|| right.cmp(left))
            });
        let Some(axis) = axis else {
            break;
        };
        shape[axis] = shape[axis]
            .checked_mul(2)
            .ok_or(MindSscError::IndexOverflow)?;
        count = count.checked_mul(2).ok_or(MindSscError::IndexOverflow)?;
    }
    Ok(shape)
}

fn proportional_quotas(
    populations: &[usize],
    total_population: usize,
    budget: usize,
) -> Result<Vec<usize>, MindSscError> {
    let denominator = u128::try_from(total_population).map_err(|_| MindSscError::IndexOverflow)?;
    let budget = u128::try_from(budget).map_err(|_| MindSscError::IndexOverflow)?;
    let mut quotas = Vec::with_capacity(populations.len());
    let mut remainders = Vec::with_capacity(populations.len());
    let mut assigned = 0_usize;
    for (stratum, population) in populations.iter().copied().enumerate() {
        let product = u128::try_from(population)
            .map_err(|_| MindSscError::IndexOverflow)?
            .checked_mul(budget)
            .ok_or(MindSscError::IndexOverflow)?;
        let quota =
            usize::try_from(product / denominator).map_err(|_| MindSscError::IndexOverflow)?;
        let remainder = product % denominator;
        assigned = assigned
            .checked_add(quota)
            .ok_or(MindSscError::IndexOverflow)?;
        quotas.push(quota);
        remainders.push((remainder, stratum));
    }
    remainders
        .sort_unstable_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let requested = usize::try_from(budget).map_err(|_| MindSscError::IndexOverflow)?;
    for &(_, stratum) in remainders.iter().take(requested - assigned) {
        quotas[stratum] = quotas[stratum]
            .checked_add(1)
            .ok_or(MindSscError::IndexOverflow)?;
    }
    Ok(quotas)
}

fn stratum_for_index(
    index: usize,
    shape: [usize; 3],
    halo: [usize; 3],
    interior: [usize; 3],
    strata_shape: [usize; 3],
) -> Result<usize, MindSscError> {
    let center = decode_index(index, shape)?;
    let mut stratum = [0_usize; 3];
    for axis in 0..3 {
        let local = center[axis] - halo[axis];
        let numerator = local
            .checked_mul(strata_shape[axis])
            .ok_or(MindSscError::IndexOverflow)?;
        stratum[axis] = (numerator / interior[axis]).min(strata_shape[axis] - 1);
    }
    linear_index(stratum, strata_shape)
}

fn sample_hash(index: usize) -> Result<u64, MindSscError> {
    let mut value =
        u64::try_from(index).map_err(|_| MindSscError::IndexOverflow)? ^ SAMPLING_HASH_SEED;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    Ok(value ^ (value >> 31))
}

fn visit_interior(
    shape: [usize; 3],
    halo: [usize; 3],
    mut visit: impl FnMut(usize) -> Result<(), MindSscError>,
) -> Result<(), MindSscError> {
    for z in halo[0]..shape[0] - halo[0] {
        for y in halo[1]..shape[1] - halo[1] {
            for x in halo[2]..shape[2] - halo[2] {
                visit(linear_index([z, y, x], shape)?)?;
            }
        }
    }
    Ok(())
}

pub(super) fn linear_index(index: [usize; 3], shape: [usize; 3]) -> Result<usize, MindSscError> {
    index[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_add(index[1]))
        .and_then(|value| value.checked_mul(shape[2]))
        .and_then(|value| value.checked_add(index[2]))
        .ok_or(MindSscError::IndexOverflow)
}

pub(super) fn decode_index(index: usize, shape: [usize; 3]) -> Result<[usize; 3], MindSscError> {
    let plane = shape[1]
        .checked_mul(shape[2])
        .ok_or(MindSscError::IndexOverflow)?;
    Ok([index / plane, index % plane / shape[2], index % shape[2]])
}

fn has_complete_support(center: [usize; 3], shape: [usize; 3], halo: [usize; 3]) -> bool {
    (0..3).all(|axis| center[axis] >= halo[axis] && center[axis] < shape[axis] - halo[axis])
}

fn usize_to_f64(value: usize) -> Result<f64, MindSscError> {
    u32::try_from(value)
        .map(f64::from)
        .map_err(|_| MindSscError::IndexOverflow)
}

fn usize_to_f64_exact(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("invariant: strata extent is bounded by image extent"))
}

#[cfg(test)]
pub(super) fn stratum_for_test(
    index: usize,
    shape: [usize; 3],
    halo: [usize; 3],
    spacing: [f64; 3],
    budget: usize,
) -> Result<usize, MindSscError> {
    let interior = std::array::from_fn(|axis| shape[axis] - 2 * halo[axis]);
    let strata_shape = spatial_strata_shape(interior, spacing, budget)?;
    stratum_for_index(index, shape, halo, interior, strata_shape)
}
