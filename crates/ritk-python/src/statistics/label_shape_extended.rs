//! Python binding for extended label shape statistics (GAP-262-STA-03).
//!
//! Wraps `ritk_statistics::label_shape_extended` to expose per-label
//! shape attributes: perimeter, roundness, flatness, elongation,
//! principal moments, centroid, and Feret diameter.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::PyImage;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_statistics::label_shape_extended::compute_label_shape_statistics_extended;

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
///       - perimeter (float): physical surface area (ITK `GetPerimeter`,
///         13-direction Crofton estimator)
///       - roundness (float): equivalent_spherical_perimeter / perimeter (ITK
///         `GetRoundness`); 1.0 ≈ sphere, not clamped
///       - flatness (float): √(λ₁/λ₀) — ITK convention, ≥ 1, 1.0 = isotropic
///       - elongation (float): √(λ₂/λ₁) — ITK convention, ≥ 1, 1.0 = isotropic
///       - principal_moments (list\[float\]): eigenvalues [λ₀, λ₁, λ₂] ascending
///       - centroid (list\[float\]): [z, y, x] in voxel index coordinates
///       - feret_diameter (float): max physical distance between surface voxels
///         (ITK `GetFeretDiameter`)
///       - equivalent_spherical_radius (float): (3·V/4π)^(1/3)
///       - equivalent_spherical_perimeter (float): 4π·r_eq² (ITK
///         `GetEquivalentSphericalPerimeter`)
///
/// Raises:
///     RuntimeError: if the underlying tensor data cannot be read as f32.
#[pyfunction]
pub fn extended_label_shape_statistics_py(
    py: Python<'_>,
    label_image: &PyImage,
) -> RitkResult<Py<PyList>> {
    let image = label_image.inner.clone();
    let stats = py
        .allow_threads(|| compute_label_shape_statistics_extended(image.as_ref()))
        .map_err(|error| RitkPyError::runtime(error.to_string()))?;

    let list = PyList::empty_bound(py);
    for s in &stats {
        let dict = PyDict::new_bound(py);
        dict.set_item("label", s.label)?;
        dict.set_item("count", s.count)?;
        dict.set_item("perimeter", s.perimeter)?;
        dict.set_item("roundness", s.roundness)?;
        dict.set_item("flatness", s.flatness)?;
        dict.set_item("elongation", s.elongation)?;
        let moments: Vec<f64> = s.principal_moments.to_vec();
        dict.set_item("principal_moments", moments)?;
        let centroid: Vec<f64> = s.centroid.to_array().to_vec();
        dict.set_item("centroid", centroid)?;
        dict.set_item("feret_diameter", s.feret_diameter)?;
        dict.set_item("equivalent_spherical_radius", s.equivalent_spherical_radius)?;
        dict.set_item(
            "equivalent_spherical_perimeter",
            s.equivalent_spherical_perimeter,
        )?;

        list.append(dict)?;
    }
    Ok(list.into())
}
