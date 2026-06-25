//! Image-to-LoadedVolume conversion and spatial metadata extraction.

use std::path::PathBuf;
use std::sync::Arc;

use crate::LoadedVolume;
use anyhow::Result;

use super::B;

/// Extract spacing, origin, and direction from a 3-D image as typed arrays.
///
/// # Return order
/// `(spacing, origin, direction)` where:
/// - `spacing`: `[f64; 3]` — voxel pitch `[dz, dy, dx]` in mm/voxel.
/// - `origin`: `[f64; 3]` — physical coordinate of the first voxel.
/// - `direction`: `[f64; 9]` — row-major 3×3 direction cosine matrix.
///
/// # Contract
/// The `image` must be 3-dimensional. The direction matrix must be 3×3
/// (9 elements), which is guaranteed by `Direction<3>`.
pub(super) fn extract_spatial_metadata(
    image: &ritk_core::image::Image<B, 3>,
) -> ([f64; 3], [f64; 3], [f64; 9]) {
    let sp = image.spacing();
    let orig = image.origin();
    let dir = image.direction();

    let spacing = [sp[0], sp[1], sp[2]];
    let origin = [orig.0[0], orig.0[1], orig.0[2]];
    let direction = dir.to_row_major();

    (spacing, origin, direction)
}

/// Convert a generic `Image<B, 3>` (with no DICOM metadata) into a
/// [`LoadedVolume`], recording `source_path` as the origin.
///
/// This function is also used by `mod.rs` for MetaImage, NRRD, and MGH
/// format loading paths, which produce an `Image` without DICOM metadata.
pub(super) fn volume_from_image_no_meta(
    image: ritk_core::image::Image<B, 3>,
    source_path: PathBuf,
) -> Result<LoadedVolume> {
    let shape = image.shape();
    let (spacing, origin, direction) = extract_spatial_metadata(&image);

    let tensor = image.into_tensor();
    let tensor_data = tensor.into_data();
    let pixels: Vec<f32> = tensor_data.into_vec::<f32>().map_err(|e| {
        anyhow::anyhow!("failed to extract f32 pixel data from image tensor: {e:?}")
    })?;

    Ok(LoadedVolume {
        data: Arc::new(pixels),
        shape,
        channels: 1,
        spacing,
        origin,
        direction,
        metadata: None,
        source: Some(source_path),
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
