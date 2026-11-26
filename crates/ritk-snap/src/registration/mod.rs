//! Rigid registration of scalar viewer volumes in physical coordinates.
mod metric;
#[cfg(test)]
mod tests;

use crate::geometry::affine::AffineTransform as Grid;
use crate::render::reslice::sampling::{sample_volume, validate_volume};
use crate::render::ResliceInterpolation;
use crate::LoadedVolume;
use anyhow::{ensure, Context, Result};
use ritk_registration::classical::spatial::transform_point;
use ritk_registration::{search_rigid_pose, AffineTransform, RigidSearchAnchor, RigidSearchConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Capture the scalar image and geometry without duplicating a study's DICOM headers.
pub(crate) fn scalar_input(source: &LoadedVolume) -> Result<LoadedVolume> {
    validate_volume(source)?;
    ensure!(
        source.shape.iter().all(|&n| n > 1),
        "Registration requires two 3D series."
    );
    if source.metadata.is_some() {
        ensure!(
            matches!(source.measurement_spacing(), Some((_, Some(_)))),
            "Registration needs consistent DICOM slice positions, orientations and pixel spacing."
        );
        ensure!(
            source
                .metadata
                .as_ref()
                .is_some_and(|m| m.slices.len() == source.shape[0]),
            "Slice geometry is incomplete for this volume."
        );
    }
    Ok(LoadedVolume {
        data: Arc::clone(&source.data),
        shape: source.shape,
        channels: 1,
        spacing: source.spacing,
        origin: source.origin,
        direction: source.direction,
        modality: source.modality,
        metadata: None,
        source: None,
        patient_name: None,
        patient_id: None,
        study_date: None,
        series_description: None,
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    })
}

fn center(volume: &LoadedVolume, grid: Grid) -> [f64; 3] {
    grid.voxel_to_patient(volume.shape.map(|n| (n - 1) as f64 * 0.5))
}

pub(crate) fn run(
    fixed: &LoadedVolume,
    moving: &LoadedVolume,
    shared_frame: bool,
    cancelled: &AtomicBool,
) -> Result<LoadedVolume> {
    let fixed_grid = validate_volume(fixed)?;
    let moving_grid = validate_volume(moving)?;
    let transform = estimate(
        fixed,
        moving,
        fixed_grid,
        moving_grid,
        shared_frame,
        cancelled,
    )?;
    let outside = if moving.modality.as_deref() == Some("CT") {
        -1024.0
    } else {
        0.0
    };
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(fixed.data.len())
        .context("allocate registered volume")?;
    for z in 0..fixed.shape[0] {
        ensure!(
            !cancelled.load(Ordering::Relaxed),
            "Registration cancelled."
        );
        for y in 0..fixed.shape[1] {
            for x in 0..fixed.shape[2] {
                let point = fixed_grid.voxel_to_patient([z as f64, y as f64, x as f64]);
                let source = moving_grid.patient_to_voxel(transform_point(&point, &transform));
                let value = match sample_volume(moving, source, ResliceInterpolation::Linear) {
                    Ok(v) => v,
                    Err(crate::render::ResliceError::OutOfVolume { .. }) => outside,
                    Err(e) => return Err(e.into()),
                };
                pixels.push(value);
            }
        }
    }
    let mut result = scalar_input(fixed)?;
    result.data = Arc::new(pixels);
    result.modality = moving.modality;
    result.series_description = Some(LoadedVolume::REGISTRATION_PREVIEW.into());
    Ok(result)
}

fn estimate(
    fixed: &LoadedVolume,
    moving: &LoadedVolume,
    fixed_grid: Grid,
    moving_grid: Grid,
    shared_frame: bool,
    cancelled: &AtomicBool,
) -> Result<AffineTransform> {
    use metric::{Input, Samples, Stage};
    let fixed = Input::new(fixed, fixed_grid)?;
    let moving = Input::new(moving, moving_grid)?;
    let mut anchor = if shared_frame {
        RigidSearchAnchor::try_new(AffineTransform::IDENTITY, fixed.center)?
    } else {
        RigidSearchAnchor::from_centers(fixed.center, moving.center)?
    };
    let mut capture = Samples::new(&fixed, &moving, anchor.transform(), Stage::Capture)?;
    let mut score = |pose: &AffineTransform| capture.score(&moving, pose, cancelled);
    if !shared_frame {
        let centered = RigidSearchAnchor::from_centers(
            center(fixed.volume, fixed.grid),
            center(moving.volume, moving.grid),
        )?;
        if score(&centered.transform())? >= score(&anchor.transform())? {
            anchor = centered;
        }
    }

    // Fix the fine sampling domains at the captured pose, leaving a margin for
    // small residual rotations. Symmetric scoring reduces interpolation bias.
    let mut refinement = None;
    let refine = |pose: &AffineTransform| {
        let inverse = inverse_rigid(pose);
        let (forward, backward) = match &mut refinement {
            Some(domains) => domains,
            None => {
                let prepare = || -> Result<_> {
                    Ok((
                        Samples::new(&fixed, &moving, *pose, Stage::Refinement)?,
                        Samples::new(&moving, &fixed, inverse, Stage::Refinement)?,
                    ))
                };
                refinement.insert(prepare().map_err(|error| {
                    ritk_registration::classical::RegistrationError::InvalidInput(error.to_string())
                })?)
            }
        };
        Ok((forward.score(&moving, pose, cancelled)?
            + backward.score(&fixed, &inverse, cancelled)?)
            * 0.5)
    };
    let fit = search_rigid_pose(
        anchor,
        RigidSearchConfig::try_new(20.0, 32.0, 0.5, 0.75, 96)?.with_structural_half_range_cells(
            std::num::NonZeroU8::new(5).expect("positive cell count"),
        ),
        score,
        refine,
    )?;
    ensure!(
        !fit.capture_saturated,
        "The rigid fit reached its search limit. Check the selected anatomy and field of view."
    );
    ensure!(
        fit.structural_score > 0.0,
        "No usable intensity correspondence was found."
    );
    Ok(fit.structural_transform)
}

fn inverse_rigid(pose: &AffineTransform) -> AffineTransform {
    let matrix = pose.as_array();
    let mut inverse = AffineTransform::IDENTITY;
    for row in 0..3 {
        for column in 0..3 {
            inverse.0[row * 4 + column] = matrix[column * 4 + row];
        }
        inverse.0[row * 4 + 3] = -(0..3)
            .map(|k| matrix[k * 4 + row] * matrix[k * 4 + 3])
            .sum::<f64>();
    }
    inverse
}
