//! Resample the moving image onto the fixed image grid.
use super::estimate::estimate;
use super::volume::scalar_input;
use crate::render::reslice::sampling::{sample_volume, validate_volume};
use crate::render::ResliceInterpolation;
use crate::LoadedVolume;
use anyhow::{ensure, Context, Result};
use ritk_registration::classical::spatial::transform_point;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
