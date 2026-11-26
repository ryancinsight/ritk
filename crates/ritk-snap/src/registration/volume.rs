//! Prepare a scalar registration input in its calibrated patient coordinates.
use crate::geometry::affine::AffineTransform as Grid;
use crate::render::reslice::sampling::validate_volume;
use crate::LoadedVolume;
use anyhow::{ensure, Result};
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

pub(super) fn center(volume: &LoadedVolume, grid: Grid) -> [f64; 3] {
    grid.voxel_to_patient(volume.shape.map(|n| (n - 1) as f64 * 0.5))
}
