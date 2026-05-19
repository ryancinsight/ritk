//! Python-exposed Jacobian determinant functions.
//!
//! All functions delegate to `ritk_core::statistics::jacobian` (SSOT).
//!
//! # Input convention
//!
//! The `jacobian_determinant` function accepts three displacement-field
//! components (disp_z, disp_y, disp_x), each a `PyImage` of shape [D, H, W].
//! All three must have identical shape.
//!
//! # Output
//!
//! Returns a single PyImage of the same shape where each voxel holds the
//! Jacobian determinant det(J(φ)) of the deformation φ(x) = x + u(x).
//! det(J) > 0 indicates a topology-preserving deformation; det(J) ≤ 0
//! indicates folding (anatomically invalid).

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::statistics::jacobian;

/// Compute the Jacobian determinant of a 3-D displacement field.
///
/// Given three displacement components (disp_z, disp_y, disp_x) each of
/// shape [D, H, W], computes `det(I + ∇u)` at every voxel using central
/// finite differences (interior) and first-order one-sided differences
/// (boundary). Physical spacing from the first input (`disp_z`) is used
/// for derivative scaling.
///
/// Args:
///     disp_z: Displacement field component along Z (shape [D, H, W]).
///     disp_y: Displacement field component along Y (shape [D, H, W]).
///     disp_x: Displacement field component along X (shape [D, H, W]).
///
/// Returns:
///     Jacobian determinant image of shape [D, H, W].
///
/// Raises:
///     RuntimeError: on internal computation failure or shape mismatch.
#[pyfunction]
pub fn jacobian_determinant(
    py: Python<'_>,
    disp_z: &PyImage,
    disp_y: &PyImage,
    disp_x: &PyImage,
) -> RitkResult<PyImage> {
    let dz = std::sync::Arc::clone(&disp_z.inner);
    let dy = std::sync::Arc::clone(&disp_y.inner);
    let dx = std::sync::Arc::clone(&disp_x.inner);
    py.allow_threads(|| {
        jacobian::jacobian_determinant(dz.as_ref(), dy.as_ref(), dx.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Analyze a Jacobian determinant field and return summary statistics.
///
/// Voxels are classified into three disjoint categories:
///
/// | Category   | Condition    | Meaning                        |
/// |------------|--------------|--------------------------------|
/// | folded     | det ≤ 0      | topological singularity        |
/// | compressed | 0 < det < 1  | local volume shrinkage         |
/// | expanded   | det ≥ 1      | volume-preserving or expanding |
///
/// Args:
///     jac: Jacobian determinant image (output of `jacobian_determinant`).
///
/// Returns:
///     A dictionary with keys:
///     - ``min``: Minimum determinant value.
///     - ``max``: Maximum determinant value.
///     - ``mean``: Mean determinant value (float).
///     - ``num_folded``: Number of voxels with det ≤ 0.
///     - ``num_compressed``: Number of voxels with 0 < det < 1.
///     - ``num_expanded``: Number of voxels with det ≥ 1.
///     - ``num_valid``: Number of voxels with det > 0 (= compressed + expanded).
///     - ``total_voxels``: Total voxel count.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
pub fn analyze_jacobian<'a>(py: Python<'a>, jac: &PyImage) -> RitkResult<pyo3::Bound<'a, pyo3::types::PyDict>> {
    let jac_inner = std::sync::Arc::clone(&jac.inner);
    let stats = py.allow_threads(|| {
        jacobian::analyze_jacobian(jac_inner.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("min", stats.min).map_err(RitkPyError::from_py)?;
    dict.set_item("max", stats.max).map_err(RitkPyError::from_py)?;
    dict.set_item("mean", stats.mean).map_err(RitkPyError::from_py)?;
    dict.set_item("num_folded", stats.num_folded).map_err(RitkPyError::from_py)?;
    dict.set_item("num_compressed", stats.num_compressed).map_err(RitkPyError::from_py)?;
    dict.set_item("num_expanded", stats.num_expanded).map_err(RitkPyError::from_py)?;
    dict.set_item("num_valid", stats.num_valid).map_err(RitkPyError::from_py)?;
    dict.set_item("total_voxels", stats.total_voxels).map_err(RitkPyError::from_py)?;
    Ok(dict)
}
