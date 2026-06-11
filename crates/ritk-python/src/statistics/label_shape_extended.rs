//! Python binding for extended label shape statistics (GAP-262-STA-03).
//!
//! Wraps `ritk_core::statistics::label_shape_extended` to expose per-label
//! shape attributes: perimeter, roundness, flatness, elongation,
//! principal moments, centroid, and Feret diameter.

use crate::errors::RitkResult;
use crate::image::PyImage;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_core::statistics::label_shape_extended::compute_label_shape_statistics_extended;
use std::sync::Arc;

/// Compute extended per-label shape statistics from a label image.
///
/// Extends `label_shape_statistics` with PCA-derived shape attributes:
/// perimeter, roundness, flatness, elongation, principal moments,
/// and Feret diameter.
///
/// Args:
///     label_image: Label image from `connected_components` (f32 labels in [0, K]).
///
/// Returns:
///     list of dicts, one per component, sorted by label ascending. Each dict has keys:
///       - label (int): component label index
///       - count (int): number of foreground voxels
///       - perimeter (int): 3-D surface area (voxel face count)
///       - roundness (float): sphericity (1.0 = perfect sphere)
///       - flatness (float): √(λ₀/λ₂) — smallest/largest moment ratio
///       - elongation (float): √(λ₁/λ₂) — middle/largest moment ratio
///       - principal_moments (list[float]): eigenvalues [λ₀, λ₁, λ₂] ascending
///       - centroid (list[float]): [z, y, x] in voxel index coordinates
///       - feret_diameter (float): approximate Feret diameter in physical units
///
/// Raises:
///     RuntimeError: if the underlying tensor data cannot be read as f32.
#[pyfunction]
pub fn extended_label_shape_statistics_py(
    py: Python<'_>,
    label_image: &PyImage,
) -> RitkResult<Py<PyList>> {
    let img = Arc::clone(&label_image.inner);
    let stats = py.allow_threads(|| compute_label_shape_statistics_extended(img.as_ref()));

    let list = PyList::empty_bound(py);
    for s in &stats {
        let dict = PyDict::new_bound(py);
        dict.set_item("label", s.label)?;
        dict.set_item("count", s.count)?;
        dict.set_item("perimeter", s.perimeter as i64)?;
        dict.set_item("roundness", s.roundness)?;
        dict.set_item("flatness", s.flatness)?;
        dict.set_item("elongation", s.elongation)?;
        let moments: Vec<f64> = s.principal_moments.to_vec();
        dict.set_item("principal_moments", moments)?;
        let centroid: Vec<f64> = s.centroid.to_array().to_vec();
        dict.set_item("centroid", centroid)?;
        dict.set_item("feret_diameter", s.feret_diameter)?;

        list.append(dict)?;
    }
    Ok(list.into())
}
