//! DICOM series loading into LoadedVolume.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use ritk_io::load_dicom_series_with_metadata;
use tracing::info;

use crate::LoadedVolume;

use super::convert::extract_spatial_metadata;
use super::B;

/// Load a DICOM series from `folder` into a [`LoadedVolume`].
///
/// # Algorithm
/// 1. Calls `ritk_io::load_dicom_series_with_metadata::<NdArray<f32>>`.
/// 2. Extracts the 3-D f32 tensor (shape `[depth, rows, cols]`).
/// 3. Reads spatial metadata (spacing, origin, direction) from the image.
/// 4. Populates optional DICOM-specific fields from [`DicomReadMetadata`].
/// 5. Sets the window/level hint from the first slice's `(WindowCenter,
///    WindowWidth)` tags when present.
///
/// # Errors
/// Propagates any error returned by `ritk_io`.
pub fn load_dicom_volume<P: AsRef<Path>>(folder: P) -> Result<LoadedVolume> {
    let folder = folder.as_ref();
    info!(path = %folder.display(), "loading DICOM volume");

    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let (image, meta) = load_dicom_series_with_metadata::<B, _>(folder, &device)
        .with_context(|| format!("failed to load DICOM series from '{}'", folder.display()))?;

    let shape = image.shape(); // [depth, rows, cols]
    let (spacing, origin, direction) = extract_spatial_metadata(&image);

    // Extract pixel data from the tensor.
    let tensor = image.into_tensor();
    let tensor_data = tensor.into_data();
    let pixels: Vec<f32> = tensor_data.into_vec::<f32>().map_err(|e| {
        anyhow::anyhow!("failed to extract f32 pixel data from DICOM tensor: {e:?}")
    })?;

    let modality = meta.modality.clone();
    let patient_name = meta.patient_name.clone();
    let patient_id = meta.patient_id.clone();
    let study_date = meta.study_date.clone();
    let series_description = meta.series_description.clone();

    Ok(LoadedVolume {
        data: Arc::new(pixels),
        shape,
        spacing,
        origin,
        direction,
        metadata: Some(Box::new(meta)),
        source: Some(folder.to_path_buf()),
        modality,
        patient_name,
        patient_id,
        study_date,
        series_description,
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    })
}
