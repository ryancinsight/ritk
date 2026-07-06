//! Python-exposed image and mesh I/O functions.
//!
//! # Image formats
//! | Extension(s)          | Read | Write | Backend              |
//! |-----------------------|------|-------|----------------------|
//! | `.nii`, `.nii.gz`     | ✓    | ✓     | ritk-io NIfTI        |
//! | `.png`                | ✓    | ✗     | ritk-io PNG          |
//! | directory (DICOM)     | ✓    | ✗     | ritk-io DICOM        |
//! | `.mha`, `.mhd`        | ✓    | ✓     | ritk-io MetaImage    |
//! | `.nrrd`               | ✓    | ✓     | ritk-io NRRD         |
//! | `.tif`, `.tiff`       | ✓    | ✓     | ritk-io TIFF         |
//! | `.vtk` (image)        | ✗    | ✗     | native path pending  |
//! | `.mgh`, `.mgz`        | ✓    | ✓     | ritk-io MGH          |
//! | `.hdr`, `.img`        | ✓    | ✓     | ritk-io Analyze      |
//! | `.jpg`, `.jpeg`       | ✓    | ✓     | ritk-io JPEG         |
//!
//! # Mesh formats
//! | Extension(s)       | Read | Write |
//! |--------------------|------|-------|
//! | `.obj`             | ✓    | ✓     |
//! | `.stl`             | ✓    | ✓     |
//! | `.ply`             | ✓    | ✓     |
//! | `.gltf`            | ✗    | ✓     |
//! | `.vtk` (polydata)  | ✓    | ✓     |
//! | `.vtp`             | ✓    | ✓     |

pub mod anonymize;
pub mod mesh;
pub mod transform;

pub use anonymize::anonymize_dicom_dir;
pub use mesh::{read_mesh, write_mesh, PyMesh};
pub use transform::{read_transform, write_transform};

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{native_into_py_image, py_image_to_native, PyImage};
use pyo3::prelude::*;
use std::path::Path;

#[inline]
fn io_err<E: std::fmt::Display>(label: &'static str) -> impl Fn(E) -> RitkPyError {
    move |e| RitkPyError::io(format!("{label}: {e}"))
}

// ── read_image ────────────────────────────────────────────────────────────────

/// Read a medical image from file.
///
/// Supports: .nii, .nii.gz, .png, .mha, .mhd, .nrrd, .tif, .tiff,
/// .mgh, .mgz, .hdr, .img, .jpg, .jpeg, or a DICOM directory.
///
/// Raises:
///     IOError: on read failure or unsupported format.
#[pyfunction]
pub fn read_image(py: Python<'_>, path: &str) -> RitkResult<PyImage> {
    let path_owned = path.to_string();
    py.allow_threads(move || {
        let p = Path::new(&path_owned);
        ritk_io::read_image_native(p)
            .map(native_into_py_image)
            .map_err(io_err("native image read error"))
    })
}

// ── write_image ───────────────────────────────────────────────────────────────

/// Write a medical image to file.  Format inferred from extension.
///
/// Supported: .nii, .nii.gz, .mha, .mhd, .nrrd, .tif, .tiff,
/// .mgh, .mgz, .hdr, .img, .jpg, .jpeg.
///
/// Raises:
///     IOError: on write failure or unsupported format.
#[pyfunction]
pub fn write_image(py: Python<'_>, image: &PyImage, path: &str) -> RitkResult<()> {
    let native = py_image_to_native(image)?;
    let path_owned = path.to_string();
    py.allow_threads(move || {
        ritk_io::write_image_native(&path_owned, &native)
            .map_err(io_err("native image write error"))
    })
}

// ── register ──────────────────────────────────────────────────────────────────

/// Register the `io` submodule with image I/O, mesh I/O, transform I/O,
/// and DICOM anonymization functions.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "io")?;
    m.add_class::<PyMesh>()?;
    m.add_function(wrap_pyfunction!(read_image, &m)?)?;
    m.add_function(wrap_pyfunction!(write_image, &m)?)?;
    m.add_function(wrap_pyfunction!(read_mesh, &m)?)?;
    m.add_function(wrap_pyfunction!(write_mesh, &m)?)?;
    m.add_function(wrap_pyfunction!(read_transform, &m)?)?;
    m.add_function(wrap_pyfunction!(write_transform, &m)?)?;
    m.add_function(wrap_pyfunction!(anonymize_dicom_dir, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
