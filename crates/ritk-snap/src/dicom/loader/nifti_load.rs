//! NIfTI volume loading into LoadedVolume.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use ritk_io::read_nifti;
use tracing::info;

use crate::LoadedVolume;

use super::convert::extract_spatial_metadata;
use super::B;

/// Load a NIfTI volume from `path` (`.nii` or `.nii.gz`) into a [`LoadedVolume`].
///
/// # Algorithm
/// 1. Calls `ritk_io::read_nifti::<NdArray<f32>>`.
/// 2. Extracts shape, spacing, origin, and direction from the returned
///    [`ritk_core::image::Image`].
/// 3. Copies pixel data from the tensor into a heap `Vec<f32>`.
///
/// NIfTI files carry no patient metadata; all optional DICOM fields are
/// left as `None`.
///
/// # Errors
/// Propagates any error returned by `ritk_io`.
pub fn load_nifti_volume<P: AsRef<Path>>(path: P) -> Result<LoadedVolume> {
    let path = path.as_ref();
    info!(path = %path.display(), "loading NIfTI volume");

    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let image = read_nifti::<B, _>(path, &device)
        .with_context(|| format!("failed to read NIfTI file '{}'", path.display()))?;

    let shape = image.shape(); // [depth, rows, cols] per RITK convention
    let (spacing, origin, direction) = extract_spatial_metadata(&image);

    let tensor = image.into_tensor();
    let tensor_data = tensor.into_data();
    let pixels: Vec<f32> = tensor_data.into_vec::<f32>().map_err(|e| {
        anyhow::anyhow!("failed to extract f32 pixel data from NIfTI tensor: {e:?}")
    })?;

    Ok(LoadedVolume {
        data: Arc::new(pixels),
        shape,
        spacing,
        origin,
        direction,
        metadata: None,
        source: Some(path.to_path_buf()),
        modality: None,
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
