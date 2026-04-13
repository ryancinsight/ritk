//! Python-exposed image I/O functions bridging ritk-io for NIfTI, PNG, DICOM,
//! MetaImage, and NRRD.
//!
//! # Supported formats
//!
//! | Extension(s)          | Read | Write | Backend              |
//! |-----------------------|------|-------|----------------------|
//! | `.nii`, `.nii.gz`     | ✓    | ✓     | ritk-io NIfTI        |
//! | `.png`                | ✓    | ✗     | ritk-io PNG          |
//! | directory (DICOM)     | ✓    | ✗     | ritk-io DICOM        |
//! | `.mha`, `.mhd`        | ✓    | ✓     | ritk-io MetaImage    |
//! | `.nrrd`               | ✓    | ✓     | ritk-io NRRD         |
//!
//! # Error mapping
//! All I/O errors are mapped to `PyIOError`.
//! Unsupported formats return `PyIOError` with an explicit message listing
//! the supported extensions.

use crate::image::{into_py_image, PyImage};
use burn_ndarray::{NdArray, NdArrayDevice};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use std::path::Path;

type Backend = NdArray<f32>;

// ── Public pyfunction: read_image ─────────────────────────────────────────────

/// Read a medical image from file.
///
/// Supports:
/// - NIfTI      (.nii, .nii.gz)  — full affine + voxel data
/// - PNG        (.png)           — single-slice [1, H, W], grayscale
/// - DICOM      (directory path) — reads the single series in the directory
/// - MetaImage  (.mha, .mhd)    — full affine + voxel data
/// - NRRD       (.nrrd)         — full affine + voxel data
///
/// Args:
///     path: File path (string). For DICOM pass the directory path.
///
/// Returns:
///     Image with voxel data, spacing, and origin set from file metadata.
///
/// Raises:
///     IOError: on read failure or unsupported format.
#[pyfunction]
pub fn read_image(path: &str) -> PyResult<PyImage> {
    let device = NdArrayDevice::default();
    let p = Path::new(path);

    let path_lower = path.to_lowercase();

    if path_lower.ends_with(".nii.gz") || path_lower.ends_with(".nii") {
        // ── NIfTI ────────────────────────────────────────────────────────────
        let image = ritk_io::read_nifti::<Backend, _>(p, &device)
            .map_err(|e| PyIOError::new_err(format!("NIfTI read error: {e}")))?;
        return Ok(into_py_image(image));
    }

    if path_lower.ends_with(".png") {
        // ── PNG (single slice) ────────────────────────────────────────────────
        let image = ritk_io::read_png_to_image::<Backend, _>(p, &device)
            .map_err(|e| PyIOError::new_err(format!("PNG read error: {e}")))?;
        return Ok(into_py_image(image));
    }

    if path_lower.ends_with(".mha") || path_lower.ends_with(".mhd") {
        let image = ritk_io::read_metaimage::<Backend, _>(p, &device)
            .map_err(|e| PyIOError::new_err(format!("MetaImage read error: {e}")))?;
        return Ok(into_py_image(image));
    }

    if path_lower.ends_with(".nrrd") {
        let image = ritk_io::read_nrrd::<Backend, _>(p, &device)
            .map_err(|e| PyIOError::new_err(format!("NRRD read error: {e}")))?;
        return Ok(into_py_image(image));
    }

    // ── DICOM directory (no recognised extension) ─────────────────────────────
    if p.is_dir() {
        let image = ritk_io::read_dicom_series::<Backend, _>(p, &device)
            .map_err(|e| PyIOError::new_err(format!("DICOM read error: {e}")))?;
        return Ok(into_py_image(image));
    }

    Err(PyIOError::new_err(format!(
        "Unsupported path '{}'. Supported: \
         .nii, .nii.gz (NIfTI), .png (single PNG slice), \
         .mha, .mhd (MetaImage), .nrrd (NRRD), \
         or a directory containing a DICOM series.",
        path
    )))
}

// ── Public pyfunction: write_image ────────────────────────────────────────────

/// Write a medical image to file.
///
/// The output format is inferred from the file extension.
///
/// Supported write formats:
/// - NIfTI     (.nii, .nii.gz)
/// - MetaImage (.mha, .mhd)
/// - NRRD      (.nrrd)
///
/// Args:
///     image: PyImage to write.
///     path:  Destination file path (string).
///
/// Raises:
///     IOError: on write failure or unsupported format.
#[pyfunction]
pub fn write_image(image: &PyImage, path: &str) -> PyResult<()> {
    let path_lower = path.to_lowercase();

    if path_lower.ends_with(".nii.gz") || path_lower.ends_with(".nii") {
        ritk_io::write_nifti(path, image.inner.as_ref())
            .map_err(|e| PyIOError::new_err(format!("NIfTI write error: {e}")))?;
        return Ok(());
    }

    if path_lower.ends_with(".png") {
        return Err(PyIOError::new_err(
            "PNG write is not yet implemented in ritk-io. \
             Use .nii, .nii.gz, .mha, .mhd, or .nrrd instead.",
        ));
    }

    if path_lower.ends_with(".mha") || path_lower.ends_with(".mhd") {
        ritk_io::write_metaimage(path, image.inner.as_ref())
            .map_err(|e| PyIOError::new_err(format!("MetaImage write error: {e}")))?;
        return Ok(());
    }

    if path_lower.ends_with(".nrrd") {
        ritk_io::write_nrrd(path, image.inner.as_ref())
            .map_err(|e| PyIOError::new_err(format!("NRRD write error: {e}")))?;
        return Ok(());
    }

    Err(PyIOError::new_err(format!(
        "Unsupported write extension for path '{}'. \
         Supported write formats: .nii, .nii.gz, .mha, .mhd, .nrrd",
        path
    )))
}

// ── Submodule registration ────────────────────────────────────────────────────

/// Register the `io` submodule with `read_image` and `write_image`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "io")?;
    m.add_function(wrap_pyfunction!(read_image, &m)?)?;
    m.add_function(wrap_pyfunction!(write_image, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
