//! Volume loading from DICOM folders and NIfTI files.
//!
//! # Functions
//!
//! - [`load_dicom_volume`] — load a DICOM series folder into a [`LoadedVolume`].
//! - [`load_nifti_volume`] — load a NIfTI `.nii` / `.nii.gz` file.
//! - [`load_volume_from_path`] — auto-detect format and dispatch to the above.
//! - [`load_volume_from_bytes`] — load a pathless in-memory medical file payload.
//! - [`scan_folder_for_series`] — walk a directory tree and return a `SeriesTree`.
//!
//! # Backend
//!
//! All tensor operations use `burn_ndarray::NdArray<f32>` (CPU, synchronous).
//! This isolates the `<B: Backend>` type parameter to this module; callers
//! receive a format-erased [`LoadedVolume`].
//!
//! # Data layout
//!
//! Loaded pixel data is stored in row-major `[depth, rows, cols]` order,
//! matching the RITK tensor convention established in `ritk-io`.
//! Spacing is `[dz, dy, dx]` mm/voxel; origin and direction follow the
//! physical coordinate system of the source format.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn_ndarray::NdArray;

use crate::dicom::input_path::classify_dicom_input_path;
use crate::LoadedVolume;

mod bytes;
mod convert;
mod dicom_load;
mod nifti_load;
mod scan;
#[cfg(test)]
mod tests;

pub use dicom_load::load_dicom_volume;
pub use nifti_load::load_nifti_volume;
pub use scan::scan_folder_for_series;

/// CPU backend alias used for all loading operations in this module.
type B = NdArray<f32>;

/// Return a default device for the `B` backend.
///
/// Centralises the `<B as Backend>::Device::default()` spelling so callers
/// don't repeat the turbofish at every format branch.
#[inline]
fn device() -> <B as burn::tensor::backend::Backend>::Device {
    Default::default()
}

/// Auto-detect the volume format from `path` and load accordingly.
///
/// | Condition | Dispatched to |
/// |------------------------------------------------------|-------------------------|
/// | `path` is a directory | [`load_dicom_volume`] |
/// | Extension is `.nii` or the path ends in `.nii.gz` | [`load_nifti_volume`] |
/// | Extension is `.mha` or `.mhd` | MetaImage (via ritk_io) |
/// | Extension is `.nrrd` | NRRD (via ritk_io) |
/// | Extension is `.mgh` or `.mgz` | MGH (via ritk_io) |
/// | No extension / unknown extension | [`load_dicom_volume`] |
///
/// # Errors
///
/// Returns an error when the format is unsupported or when the underlying
/// loader fails.
pub fn load_volume_from_path<P: AsRef<Path>>(path: P) -> Result<LoadedVolume> {
    let path = path.as_ref();

    if classify_dicom_input_path(path).dicom_root().is_some() {
        return load_dicom_volume(path);
    }

    let path_str = path.to_string_lossy().to_lowercase();
    if path_str.ends_with(".nii.gz") || path_str.ends_with(".nii") {
        return load_nifti_volume(path);
    }
    if path_str.ends_with(".mha") || path_str.ends_with(".mhd") {
        let image = ritk_io::read_metaimage::<B, _>(path, &device())
            .with_context(|| format!("failed to read MetaImage '{}'", path.display()))?;
        return convert::volume_from_image_no_meta(image, path.to_path_buf());
    }
    if path_str.ends_with(".nrrd") || path_str.ends_with(".nhdr") {
        let image = ritk_io::read_nrrd::<B, _>(path, &device())
            .with_context(|| format!("failed to read NRRD '{}'", path.display()))?;
        return convert::volume_from_image_no_meta(image, path.to_path_buf());
    }
    if path_str.ends_with(".mgh") || path_str.ends_with(".mgz") {
        let image = ritk_io::read_mgh::<B, _>(path, &device())
            .with_context(|| format!("failed to read MGH '{}'", path.display()))?;
        return convert::volume_from_image_no_meta(image, path.to_path_buf());
    }

    // Fallback: treat as DICOM folder or single-file DICOM.
    load_dicom_volume(path)
}


/// Load a pathless in-memory medical payload.
///
/// Currently supports NIfTI byte payloads identified by `name_hint`
/// (`.nii` / `.nii.gz`).
pub fn load_volume_from_bytes(name_hint: &str, bytes: &[u8]) -> Result<LoadedVolume> {
    let name = name_hint.to_ascii_lowercase();

    if bytes::is_likely_dicom_bytes(name_hint, bytes) {
        return load_dicom_series_from_named_bytes(&[(name_hint.to_owned(), bytes)]);
    }

    if name.ends_with(".nii") || name.ends_with(".nii.gz") {
        let image = ritk_io::read_nifti_from_bytes::<B>(bytes, &device())
            .with_context(|| format!("failed to read dropped NIfTI bytes '{}'", name_hint))?;
        return convert::volume_from_image_no_meta(image, PathBuf::from(name_hint));
    }

    anyhow::bail!(
        "unsupported dropped in-memory file '{}' (supported: DICOM, .nii, .nii.gz)",
        name_hint
    )
}

/// Load a DICOM series from a pathless dropped in-memory byte batch.
///
/// Zero-disk implementation: the batch is scanned in-memory via
/// [`ritk_io::scan_dicom_part10_bytes`] and loaded via
/// [`ritk_io::load_dicom_from_series`], which decodes pixel data from
/// `part10_bytes` stored in each slice's metadata — no temporary files
/// are written to disk.
pub fn load_dicom_series_from_named_bytes(files: &[(String, &[u8])]) -> Result<LoadedVolume> {
    if files.is_empty() {
        anyhow::bail!("empty DICOM byte batch")
    }
    // Filter to DICOM-like payloads and build the scan input.
    let dicom_files: Vec<(&str, &[u8])> = files
        .iter()
        .filter(|(name, bytes)| bytes::is_likely_dicom_bytes(name, bytes))
        .map(|(name, bytes)| (name.as_str(), *bytes))
        .collect();
    if dicom_files.is_empty() {
        anyhow::bail!("no DICOM Part 10 payloads found in the dropped byte batch")
    }
    let series = ritk_io::scan_dicom_part10_bytes(&dicom_files)
        .with_context(|| "failed to scan dropped DICOM byte batch")?;
    dicom_load::load_volume_from_scanned_series(series)
}

/// Load a DICOM series from SCP-received `StoredInstance` values.
///
/// Zero-disk implementation: instances are scanned in-memory via
/// [`ritk_io::scan_dicom_instances`] and loaded via
/// [`ritk_io::load_dicom_from_series`], which decodes pixel data from
/// `part10_bytes` stored in each slice's metadata — no temporary files
/// are written to disk.
///
/// # Errors
///
/// Returns an error when:
/// - `instances` is empty.
/// - `scan_dicom_instances` fails to parse any instance.
/// - `load_dicom_from_series` fails to decode pixel data or reconstruct
///   the volume geometry.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_dicom_series_from_stored_instances(
    instances: &[ritk_io::StoredInstance],
) -> Result<LoadedVolume> {
    if instances.is_empty() {
        anyhow::bail!("no SCP-received DICOM instances to load");
    }
    let series = ritk_io::scan_dicom_instances(instances)
        .with_context(|| "failed to scan SCP-received DICOM instances")?;
    dicom_load::load_volume_from_scanned_series(series)
}
