use ritk_diffusion::maps::DtiVolume;
use ritk_spatial::Point;

use crate::direction_fields::dti_volume_direction_field;
use crate::tracking::euler_tractography;
use crate::types::{DtiTractographyConfig, TractographyError, TractographyResult};

/// Select voxel-index seeds from a fitted DTI volume by inclusive FA threshold.
///
/// Unfitted voxels never qualify, including when `seed_anisotropy` is zero.
/// When a cap is present, the selected qualifying voxels are evenly strided
/// through storage order so the cap does not bias one end of the volume.
/// Returned points use the volume's `[depth, row, column]` index convention.
///
/// # Errors
///
/// Returns [`TractographyError::InvalidSeedAnisotropy`] for a threshold outside
/// `[0, 1]` and [`TractographyError::Allocation`] if the bounded output cannot
/// be reserved.
pub fn dti_volume_seed_points(
    volume: &DtiVolume,
    seed_anisotropy: f64,
    max_seeds: usize,
) -> Result<Box<[Point<3>]>, TractographyError> {
    dti_volume_seed_points_with_mask(volume, seed_anisotropy, max_seeds, None)
}

/// Select DTI-grid seed points with an optional region mask.
///
/// `seed_mask`, when present, must contain exactly one flag per DTI voxel in
/// the volume's `[depth, row, column]` storage order. A `false` flag excludes
/// that voxel from thresholding and seed-cap accounting; it does not alter the
/// volume's mask or the region through which selected streamlines may track.
/// `None` is equivalent to [`dti_volume_seed_points`].
///
/// # Errors
///
/// Returns [`TractographyError::InvalidSeedMaskLength`] when `seed_mask` does
/// not cover the volume's DTI voxels, [`TractographyError::InvalidSeedAnisotropy`]
/// for a threshold outside `[0, 1]`, and [`TractographyError::Allocation`] if
/// the bounded output cannot be reserved.
pub fn dti_volume_seed_points_with_mask(
    volume: &DtiVolume,
    seed_anisotropy: f64,
    max_seeds: usize,
    seed_mask: Option<&[bool]>,
) -> Result<Box<[Point<3>]>, TractographyError> {
    validate_seed_mask(volume, seed_mask)?;

    if !seed_anisotropy.is_finite() || !(0.0..=1.0).contains(&seed_anisotropy) {
        return Err(TractographyError::InvalidSeedAnisotropy {
            value: seed_anisotropy,
        });
    }

    let mask = volume.maps().mask();
    let qualifying_count = mask
        .iter()
        .enumerate()
        .filter(|(voxel, fitted)| {
            **fitted
                && seed_mask_allows(seed_mask, *voxel)
                && volume.maps().fractional_anisotropy_at(*voxel) >= seed_anisotropy
        })
        .count();
    if qualifying_count == 0 {
        return Ok(Box::new([]));
    }

    let stride = if max_seeds == 0 || qualifying_count <= max_seeds {
        1
    } else {
        qualifying_count.div_ceil(max_seeds)
    };
    let selected_count = qualifying_count.div_ceil(stride);
    let mut seeds = Vec::new();
    seeds
        .try_reserve_exact(selected_count)
        .map_err(|_| TractographyError::Allocation {
            requested: selected_count,
        })?;

    let [_, rows, columns] = volume.shape();
    let plane = rows
        .checked_mul(columns)
        .ok_or(TractographyError::Allocation {
            requested: usize::MAX,
        })?;
    let mut qualifying_index = 0_usize;
    for (voxel, fitted) in mask.iter().enumerate() {
        if !*fitted
            || !seed_mask_allows(seed_mask, voxel)
            || volume.maps().fractional_anisotropy_at(voxel) < seed_anisotropy
        {
            continue;
        }
        if qualifying_index.is_multiple_of(stride) {
            #[expect(
                clippy::cast_precision_loss,
                reason = "voxel indices are far below f64's exact-integer range"
            )]
            let index = [
                (voxel / plane) as f64,
                ((voxel % plane) / columns) as f64,
                (voxel % columns) as f64,
            ];
            seeds.push(Point::new(index));
        }
        qualifying_index += 1;
    }

    Ok(seeds.into_boxed_slice())
}

/// Seed and track a fitted DTI volume using one reusable policy.
///
/// The returned streamlines are in voxel-index space. Use
/// [`TractographyResult::map_points`](crate::TractographyResult::map_points)
/// with the reference image transform before writing a tractogram. A DTI
/// volume's own mask and anisotropy floor define where tracking may continue;
/// the configuration's FA threshold only selects starting voxels.
///
/// # Errors
///
/// Returns a typed configuration, allocation, no-seed, or integration error.
pub fn dti_volume_tractography(
    volume: &DtiVolume,
    config: DtiTractographyConfig,
) -> Result<TractographyResult, TractographyError> {
    dti_volume_tractography_with_mask(volume, config, None)
}

/// Seed and track a fitted DTI volume within an optional seed region.
///
/// The optional `seed_mask` is borrowed in DTI-grid `[depth, row, column]`
/// order and is applied only to candidate starting voxels. It must contain one
/// flag per DTI voxel; the volume's own fitted-voxel mask and anisotropy floor
/// remain authoritative for tracking after a seed is selected. `None` is
/// equivalent to [`dti_volume_tractography`].
///
/// # Errors
///
/// Returns [`TractographyError::InvalidSeedMaskLength`] for a malformed mask,
/// [`TractographyError::NoSeeds`] when no candidate survives the region and
/// threshold, or a typed configuration, allocation, or integration error.
pub fn dti_volume_tractography_with_mask(
    volume: &DtiVolume,
    config: DtiTractographyConfig,
    seed_mask: Option<&[bool]>,
) -> Result<TractographyResult, TractographyError> {
    let seeds = dti_volume_seed_points_with_mask(
        volume,
        config.seed_anisotropy(),
        config.max_seeds(),
        seed_mask,
    )?;
    if seeds.is_empty() {
        let maximum = volume
            .maps()
            .mask()
            .iter()
            .enumerate()
            .filter(|(voxel, fitted)| **fitted && seed_mask_allows(seed_mask, *voxel))
            .map(|(voxel, _)| volume.maps().fractional_anisotropy_at(voxel))
            .fold(0.0_f64, f64::max);
        return Err(TractographyError::NoSeeds {
            threshold: config.seed_anisotropy(),
            maximum,
        });
    }

    euler_tractography(
        &seeds,
        config.tracking(),
        dti_volume_direction_field(volume),
    )
}

fn validate_seed_mask(
    volume: &DtiVolume,
    seed_mask: Option<&[bool]>,
) -> Result<(), TractographyError> {
    let Some(seed_mask) = seed_mask else {
        return Ok(());
    };
    let expected = volume.maps().len();
    if seed_mask.len() != expected {
        return Err(TractographyError::InvalidSeedMaskLength {
            actual: seed_mask.len(),
            expected,
        });
    }
    Ok(())
}

fn seed_mask_allows(seed_mask: Option<&[bool]>, voxel: usize) -> bool {
    seed_mask.is_none_or(|mask| mask[voxel])
}
