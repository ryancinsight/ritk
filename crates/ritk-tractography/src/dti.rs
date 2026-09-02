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
            **fitted && volume.maps().fractional_anisotropy_at(*voxel) >= seed_anisotropy
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
        if !*fitted || volume.maps().fractional_anisotropy_at(voxel) < seed_anisotropy {
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
    let seeds = dti_volume_seed_points(volume, config.seed_anisotropy(), config.max_seeds())?;
    if seeds.is_empty() {
        let maximum = volume
            .maps()
            .mask()
            .iter()
            .enumerate()
            .filter(|(_, fitted)| **fitted)
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
