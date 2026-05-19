//! DICOM series loading into LoadedVolume.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use ritk_io::{is_rgb_dicom_series, load_dicom_color_series, load_dicom_series_with_metadata};
use tracing::info;

use crate::LoadedVolume;

use super::convert::extract_spatial_metadata;
use super::B;

/// Load a DICOM series from `folder` into a [`LoadedVolume`].
///
/// # Algorithm
/// 1. Detects whether the series is RGB via [`is_rgb_dicom_series`].
/// 2. For RGB series: calls [`load_dicom_color_series`], converts the
///    `RgbVolume<B>` into a [`LoadedVolume`] with `channels: 3`.
/// 3. For scalar series: calls [`load_dicom_series_with_metadata`].
/// 4. Extracts spatial metadata (spacing, origin, direction) from the image.
/// 5. Populates optional DICOM-specific fields from [`DicomReadMetadata`].
///
/// # Errors
/// Propagates any error returned by `ritk_io`.
pub fn load_dicom_volume<P: AsRef<Path>>(folder: P) -> Result<LoadedVolume> {
    let folder = folder.as_ref();
    info!(path = %folder.display(), "loading DICOM volume");

    let device = <B as burn::tensor::backend::Backend>::Device::default();

    // Detect RGB colour series and route to the colour-volume loader.
    if is_rgb_dicom_series(folder).unwrap_or(false) {
        return load_dicom_color_volume::<B>(folder, &device);
    }

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
        channels: 1,
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

/// Load an RGB DICOM colour series into a [`LoadedVolume`] with `channels: 3`.
///
/// # Parameters
/// - `folder` — path to the DICOM series directory.
/// - `device` — burn backend device.  Only the `Default::default()` CPU
///   device is guaranteed to work.
fn load_dicom_color_volume<B: burn::tensor::backend::Backend>(
    folder: &Path,
    device: &B::Device,
) -> Result<LoadedVolume> {
    info!(path = %folder.display(), "loading DICOM RGB colour volume");

    let (rgb_vol, meta) = load_dicom_color_series::<B, _>(folder, device).with_context(|| {
        format!(
            "failed to load DICOM RGB series from '{}'",
            folder.display()
        )
    })?;

    let [depth, rows, cols] = rgb_vol.spatial_shape();
    let shape = [depth, rows, cols];

    let sp = rgb_vol.spacing();
    let spacing = [sp[0], sp[1], sp[2]];

    let orig = rgb_vol.origin();
    let origin = [orig.0[0], orig.0[1], orig.0[2]];

    let dir = rgb_vol.direction().0;
    let direction = [
        dir[(0, 0)],
        dir[(0, 1)],
        dir[(0, 2)],
        dir[(1, 0)],
        dir[(1, 1)],
        dir[(1, 2)],
        dir[(2, 0)],
        dir[(2, 1)],
        dir[(2, 2)],
    ];

    let pixels = rgb_vol.data_vec();

    let modality = meta.modality.clone();
    let patient_name = meta.patient_name.clone();
    let patient_id = meta.patient_id.clone();
    let study_date = meta.study_date.clone();
    let series_description = meta.series_description.clone();

    Ok(LoadedVolume {
        data: Arc::new(pixels),
        shape,
        channels: 3,
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
