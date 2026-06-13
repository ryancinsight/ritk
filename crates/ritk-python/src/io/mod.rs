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
//! | `.vtk` (image)        | ✓    | ✓     | ritk-io VTK          |
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
use crate::image::{image_to_vec, into_py_image, PyImage};
use burn_ndarray::{NdArray, NdArrayDevice};
use pyo3::prelude::*;
use std::path::Path;

type Backend = NdArray<f32>;

#[inline]
fn io_err<E: std::fmt::Display>(label: &'static str) -> impl Fn(E) -> RitkPyError {
    move |e| RitkPyError::io(format!("{label}: {e}"))
}

// ── read_image ────────────────────────────────────────────────────────────────

/// Read a medical image from file.
///
/// Supports: .nii, .nii.gz, .png, .mha, .mhd, .nrrd, .tif, .tiff, .vtk,
/// .mgh, .mgz, .hdr, .img, .jpg, .jpeg, or a DICOM directory.
///
/// Raises:
///     IOError: on read failure or unsupported format.
#[pyfunction]
pub fn read_image(py: Python<'_>, path: &str) -> RitkResult<PyImage> {
    let path_owned = path.to_string();
    py.allow_threads(move || {
        let device = NdArrayDevice::default();
        let p = Path::new(&path_owned);

        // DICOM directories are not extension-based — must be checked before format inference.
        if p.is_dir() {
            let image = ritk_io::read_dicom_series::<Backend, _>(p, &device)
                .map_err(|e| RitkPyError::io(format!("DICOM read error: {e}")))?;
            return Ok(into_py_image(image));
        }

        match ritk_io::ImageFormat::from_path(p) {
            Some(ritk_io::ImageFormat::NIfTI) => ritk_io::read_nifti::<Backend, _>(p, &device)
                .map(into_py_image)
                .map_err(io_err("NIfTI read error")),
            Some(ritk_io::ImageFormat::MetaImage) => {
                ritk_io::read_metaimage::<Backend, _>(p, &device)
                    .map(into_py_image)
                    .map_err(io_err("MetaImage read error"))
            }
            Some(ritk_io::ImageFormat::Nrrd) => ritk_io::read_nrrd::<Backend, _>(p, &device)
                .map(into_py_image)
                .map_err(io_err("NRRD read error")),
            Some(ritk_io::ImageFormat::Png) => ritk_io::read_png_to_image::<Backend, _>(p, &device)
                .map(into_py_image)
                .map_err(io_err("PNG read error")),
            Some(ritk_io::ImageFormat::Tiff) => ritk_io::read_tiff::<Backend, _>(p, &device)
                .map(into_py_image)
                .map_err(io_err("TIFF read error")),
            Some(ritk_io::ImageFormat::Vtk) => ritk_io::read_vtk::<Backend, _>(p, &device)
                .map(into_py_image)
                .map_err(io_err("VTK read error")),
            Some(ritk_io::ImageFormat::Mgh) => ritk_io::read_mgh::<Backend, _>(p, &device)
                .map(into_py_image)
                .map_err(io_err("MGH read error")),
            Some(ritk_io::ImageFormat::Analyze) => ritk_io::read_analyze::<Backend, _>(p, &device)
                .map(into_py_image)
                .map_err(io_err("Analyze read error")),
            Some(ritk_io::ImageFormat::Jpeg) => ritk_io::read_jpeg::<Backend, _>(p, &device)
                .map(into_py_image)
                .map_err(io_err("JPEG read error")),
            Some(ritk_io::ImageFormat::Dicom) | None => Err(RitkPyError::io(format!(
                "Unsupported path '{}'. Supported: .nii, .nii.gz, .png, \
                 .mha, .mhd, .nrrd, .tif, .tiff, .vtk, .mgh, .mgz, \
                 .hdr, .img, .jpg, .jpeg, or a DICOM directory.",
                path_owned
            ))),
        }
    })
}

// ── write_image ───────────────────────────────────────────────────────────────

/// Write a medical image to file.  Format inferred from extension.
///
/// Supported: .nii, .nii.gz, .mha, .mhd, .nrrd, .tif, .tiff, .vtk,
/// .mgh, .mgz, .hdr, .img, .jpg, .jpeg.
///
/// Raises:
///     IOError: on write failure or unsupported format.
#[pyfunction]
pub fn write_image(py: Python<'_>, image: &PyImage, path: &str) -> RitkResult<()> {
    let image = std::sync::Arc::clone(&image.inner);
    let path_owned = path.to_string();
    py.allow_threads(move || {
        let p = Path::new(&path_owned);
        match ritk_io::ImageFormat::from_path(p) {
            Some(ritk_io::ImageFormat::NIfTI) => ritk_io::write_nifti(&path_owned, image.as_ref())
                .map_err(io_err("NIfTI write error")),
            Some(ritk_io::ImageFormat::MetaImage) => {
                // Fast NdArray slice extraction (O(1) borrow + one copy) instead
                // of the generic `into_data()` materialization the writer would
                // otherwise run — ~10× faster binary write for large volumes.
                let (data, _shape) = image_to_vec(image.as_ref());
                ritk_io::write_metaimage_with_data(&path_owned, image.as_ref(), &data)
                    .map_err(io_err("MetaImage write error"))
            }
            Some(ritk_io::ImageFormat::Nrrd) => {
                let (data, _shape) = image_to_vec(image.as_ref());
                ritk_io::write_nrrd_with_data(&path_owned, image.as_ref(), &data)
                    .map_err(io_err("NRRD write error"))
            }
            Some(ritk_io::ImageFormat::Tiff) => {
                ritk_io::write_tiff(image.as_ref(), &path_owned).map_err(io_err("TIFF write error"))
            }
            Some(ritk_io::ImageFormat::Vtk) => {
                ritk_io::write_vtk(&path_owned, image.as_ref()).map_err(io_err("VTK write error"))
            }
            Some(ritk_io::ImageFormat::Mgh) => {
                ritk_io::write_mgh(image.as_ref(), &path_owned).map_err(io_err("MGH write error"))
            }
            Some(ritk_io::ImageFormat::Analyze) => {
                ritk_io::write_analyze(&path_owned, image.as_ref())
                    .map_err(io_err("Analyze write error"))
            }
            Some(ritk_io::ImageFormat::Jpeg) => {
                ritk_io::write_jpeg(&path_owned, image.as_ref()).map_err(io_err("JPEG write error"))
            }
            Some(ritk_io::ImageFormat::Png) => Err(RitkPyError::io(
                "PNG write not yet implemented. Use .nii, .nii.gz, .mha, .mhd, or .nrrd.",
            )),
            Some(ritk_io::ImageFormat::Dicom) | None => Err(RitkPyError::io(format!(
                "Unsupported write extension for '{}'. \
                 Supported: .nii, .nii.gz, .mha, .mhd, .nrrd, .tif, .tiff, \
                 .vtk, .mgh, .mgz, .hdr, .img, .jpg, .jpeg",
                path_owned
            ))),
        }
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
