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
use pyo3::types::{PyDict, PyList};
use ritk_core::transform::composite_io::{CompositeTransform, TransformDescription};
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

// ── Public pyfunction: read_transform ─────────────────────────────────────────

/// Read a composite transform from a JSON file.
///
/// The file format matches the `CompositeTransform` JSON schema produced by
/// `ritk_core::transform::composite_io`.
///
/// Args:
///     path: File path to the JSON transform file.
///
/// Returns:
///     A dict with keys:
///     - `"dimensionality"`: int (2 or 3)
///     - `"description"`: str
///     - `"transforms"`: list of dicts, each with a `"type"` key
///       (`"translation"`, `"rigid"`, `"affine"`, `"displacement_field"`,
///       `"bspline"`) and parameter keys matching the variant fields.
///
/// Raises:
///     IOError: on read failure or invalid JSON.
#[pyfunction]
pub fn read_transform(py: Python<'_>, path: String) -> PyResult<PyObject> {
    let composite = CompositeTransform::load_json(&path)
        .map_err(|e| PyIOError::new_err(format!("Transform read error: {e}")))?;

    let transforms_list = PyList::empty_bound(py);
    for t in &composite.transforms {
        let d = PyDict::new_bound(py);
        match t {
            TransformDescription::Translation { offset } => {
                d.set_item("type", "translation")?;
                d.set_item("offset", offset.clone())?;
            }
            TransformDescription::Rigid {
                rotation,
                translation,
            } => {
                d.set_item("type", "rigid")?;
                d.set_item("rotation", rotation.clone())?;
                d.set_item("translation", translation.clone())?;
            }
            TransformDescription::Affine { matrix } => {
                d.set_item("type", "affine")?;
                d.set_item("matrix", matrix.clone())?;
            }
            TransformDescription::DisplacementField {
                dims,
                origin,
                spacing,
                components,
            } => {
                d.set_item("type", "displacement_field")?;
                d.set_item("dims", dims.clone())?;
                d.set_item("origin", origin.clone())?;
                d.set_item("spacing", spacing.clone())?;
                d.set_item("components", components.clone())?;
            }
            TransformDescription::BSpline {
                grid_dims,
                grid_origin,
                grid_spacing,
                components,
            } => {
                d.set_item("type", "bspline")?;
                d.set_item("grid_dims", grid_dims.clone())?;
                d.set_item("grid_origin", grid_origin.clone())?;
                d.set_item("grid_spacing", grid_spacing.clone())?;
                d.set_item("components", components.clone())?;
            }
        }
        transforms_list.append(d)?;
    }

    let result = PyDict::new_bound(py);
    result.set_item("dimensionality", composite.dimensionality)?;
    result.set_item("description", &composite.description)?;
    result.set_item("transforms", transforms_list)?;

    Ok(result.into())
}

// ── Public pyfunction: write_transform ────────────────────────────────────────

/// Write a composite transform to a JSON file.
///
/// Args:
///     path:             Destination file path.
///     dimensionality:   Spatial dimensionality (2 or 3).
///     transforms:       List of dicts, each with a `"type"` key and parameter
///                       keys matching the transform variant fields.
///                       Supported types: `"translation"`, `"rigid"`,
///                       `"affine"`, `"displacement_field"`, `"bspline"`.
///     description:      Optional human-readable description (default `""`).
///
/// Raises:
///     IOError: on write failure or invalid transform dict structure.
#[pyfunction]
#[pyo3(signature = (path, dimensionality, transforms, description=""))]
pub fn write_transform(
    py: Python<'_>,
    path: String,
    dimensionality: usize,
    transforms: Vec<PyObject>,
    description: &str,
) -> PyResult<()> {
    let mut composite = CompositeTransform::new(dimensionality);
    composite.description = description.to_string();

    for obj in &transforms {
        let dict = obj
            .downcast_bound::<PyDict>(py)
            .map_err(|_| PyIOError::new_err("each transform must be a dict"))?;

        let type_str: String = dict
            .get_item("type")?
            .ok_or_else(|| PyIOError::new_err("transform dict missing 'type' key"))?
            .extract()?;

        let td = match type_str.as_str() {
            "translation" => {
                let offset: Vec<f64> = dict
                    .get_item("offset")?
                    .ok_or_else(|| PyIOError::new_err("translation missing 'offset'"))?
                    .extract()?;
                TransformDescription::Translation { offset }
            }
            "rigid" => {
                let rotation: Vec<f64> = dict
                    .get_item("rotation")?
                    .ok_or_else(|| PyIOError::new_err("rigid missing 'rotation'"))?
                    .extract()?;
                let translation: Vec<f64> = dict
                    .get_item("translation")?
                    .ok_or_else(|| PyIOError::new_err("rigid missing 'translation'"))?
                    .extract()?;
                TransformDescription::Rigid {
                    rotation,
                    translation,
                }
            }
            "affine" => {
                let matrix: Vec<f64> = dict
                    .get_item("matrix")?
                    .ok_or_else(|| PyIOError::new_err("affine missing 'matrix'"))?
                    .extract()?;
                TransformDescription::Affine { matrix }
            }
            "displacement_field" => {
                let dims: Vec<usize> = dict
                    .get_item("dims")?
                    .ok_or_else(|| PyIOError::new_err("displacement_field missing 'dims'"))?
                    .extract()?;
                let origin: Vec<f64> = dict
                    .get_item("origin")?
                    .ok_or_else(|| PyIOError::new_err("displacement_field missing 'origin'"))?
                    .extract()?;
                let spacing: Vec<f64> = dict
                    .get_item("spacing")?
                    .ok_or_else(|| PyIOError::new_err("displacement_field missing 'spacing'"))?
                    .extract()?;
                let components: Vec<Vec<f64>> = dict
                    .get_item("components")?
                    .ok_or_else(|| PyIOError::new_err("displacement_field missing 'components'"))?
                    .extract()?;
                TransformDescription::DisplacementField {
                    dims,
                    origin,
                    spacing,
                    components,
                }
            }
            "bspline" => {
                let grid_dims: Vec<usize> = dict
                    .get_item("grid_dims")?
                    .ok_or_else(|| PyIOError::new_err("bspline missing 'grid_dims'"))?
                    .extract()?;
                let grid_origin: Vec<f64> = dict
                    .get_item("grid_origin")?
                    .ok_or_else(|| PyIOError::new_err("bspline missing 'grid_origin'"))?
                    .extract()?;
                let grid_spacing: Vec<f64> = dict
                    .get_item("grid_spacing")?
                    .ok_or_else(|| PyIOError::new_err("bspline missing 'grid_spacing'"))?
                    .extract()?;
                let components: Vec<Vec<f64>> = dict
                    .get_item("components")?
                    .ok_or_else(|| PyIOError::new_err("bspline missing 'components'"))?
                    .extract()?;
                TransformDescription::BSpline {
                    grid_dims,
                    grid_origin,
                    grid_spacing,
                    components,
                }
            }
            other => {
                return Err(PyIOError::new_err(format!(
                    "unknown transform type '{other}'; expected one of: \
                     translation, rigid, affine, displacement_field, bspline"
                )));
            }
        };

        composite.push(td);
    }

    composite
        .save_json(&path)
        .map_err(|e| PyIOError::new_err(format!("Transform write error: {e}")))?;

    Ok(())
}

// ── Submodule registration ────────────────────────────────────────────────────

/// Register the `io` submodule with image and transform I/O functions.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "io")?;
    m.add_function(wrap_pyfunction!(read_image, &m)?)?;
    m.add_function(wrap_pyfunction!(write_image, &m)?)?;
    m.add_function(wrap_pyfunction!(read_transform, &m)?)?;
    m.add_function(wrap_pyfunction!(write_transform, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
