use crate::errors::RitkPyError;
use crate::errors::RitkResult;
use crate::image::{image_from_py, into_py_image, PyImage};
use pyo3::prelude::*;

/// Warp a moving image through a dense displacement field.
///
/// `out(p) = moving(p + D(p))` with trilinear interpolation, matching
/// `SimpleITK.Warp` (linear interpolator). The displacement field is supplied as
/// three scalar component images `(disp_z, disp_y, disp_x)` on the output grid,
/// each `[D, H, W]` and in physical units; the output adopts the field geometry.
/// Samples whose mapped point leaves the moving buffer take 0 (edge padding).
#[pyfunction]
pub fn warp(
    py: Python<'_>,
    moving: &PyImage,
    disp_z: &PyImage,
    disp_y: &PyImage,
    disp_x: &PyImage,
) -> RitkResult<PyImage> {
    let mv = image_from_py(moving);
    let dz = image_from_py(disp_z);
    let dy = image_from_py(disp_y);
    let dx = image_from_py(disp_x);
    py.allow_threads(|| {
        ritk_filter::warp_image(&mv, &dz, &dy, &dx).map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Iteratively invert a dense displacement field, matching
/// `SimpleITK.InvertDisplacementField`.
///
/// The field is given as world components `(disp_z, disp_y, disp_x)` (each a
/// `[z,y,x]` scalar image, the order `filter.warp` consumes), and the inverse is
/// returned in the same `(disp_z, disp_y, disp_x)` order. Uses ITK's Chen et al.
/// fixed-point scheme with vector linear interpolation; internal arithmetic is
/// f64. Float-exact to sitk.
///
/// Args:
///     disp_z, disp_y, disp_x: forward displacement components (world frame).
///     max_iterations:         fixed-point iterations (default 10).
///     max_error_tolerance:    max scaled-residual stopping threshold (default 0.1).
///     mean_error_tolerance:   mean scaled-residual stopping threshold (default 0.001).
///     enforce_boundary:       pin the inverse to zero on the border (default True).
///
/// Returns:
///     (inv_disp_z, inv_disp_y, inv_disp_x): inverted components.
#[pyfunction]
#[pyo3(signature = (disp_z, disp_y, disp_x, max_iterations=10, max_error_tolerance=0.1,
                    mean_error_tolerance=0.001, enforce_boundary=true))]
#[allow(clippy::too_many_arguments)]
pub fn invert_displacement_field(
    py: Python<'_>,
    disp_z: &PyImage,
    disp_y: &PyImage,
    disp_x: &PyImage,
    max_iterations: usize,
    max_error_tolerance: f64,
    mean_error_tolerance: f64,
    enforce_boundary: bool,
) -> (PyImage, PyImage, PyImage) {
    let az = image_from_py(disp_z);
    let ay = image_from_py(disp_y);
    let ax = image_from_py(disp_x);
    let (vx, vy, vz) = py.allow_threads(|| {
        ritk_filter::InvertDisplacementField {
            max_iterations,
            max_error_tolerance,
            mean_error_tolerance,
            enforce_boundary,
        }
        .apply(&ax, &ay, &az)
    });
    // Return in (disp_z, disp_y, disp_x) order to match the input convention.
    (into_py_image(vz), into_py_image(vy), into_py_image(vx))
}

/// Invert a dense displacement field by thin-plate-spline fitting, matching
/// `SimpleITK.InverseDisplacementField`.
///
/// Components are `(disp_z, disp_y, disp_x)` in/out (world frame, the order
/// `filter.warp` uses). Subsamples the field every `subsampling_factor`-th voxel
/// into landmark pairs, fits ITK's `KernelTransform` (G(r)=r), and evaluates the
/// inverse at every voxel. A `z==1` field is inverted as a genuine 2-D field.
/// The TPS fit is unique and well-conditioned, so the result is float-exact to
/// sitk. Internal arithmetic is f64. Output grid matches the input grid.
///
/// Args:
///     disp_z, disp_y, disp_x: forward displacement components (world frame).
///     subsampling_factor:     landmark subsampling per axis (default 16).
///
/// Returns:
///     (inv_disp_z, inv_disp_y, inv_disp_x): inverted components.
#[pyfunction]
#[pyo3(signature = (disp_z, disp_y, disp_x, subsampling_factor=16))]
pub fn inverse_displacement_field(
    py: Python<'_>,
    disp_z: &PyImage,
    disp_y: &PyImage,
    disp_x: &PyImage,
    subsampling_factor: usize,
) -> (PyImage, PyImage, PyImage) {
    let az = image_from_py(disp_z);
    let ay = image_from_py(disp_y);
    let ax = image_from_py(disp_x);
    let (vx, vy, vz) = py.allow_threads(|| {
        ritk_filter::InverseDisplacementField { subsampling_factor }.apply(&ax, &ay, &az)
    });
    (into_py_image(vz), into_py_image(vy), into_py_image(vx))
}

/// Iteratively invert a dense displacement field by coordinate-descent line
/// search, matching `SimpleITK.IterativeInverseDisplacementField`.
///
/// Components are `(disp_z, disp_y, disp_x)` in/out (the order `filter.warp`
/// uses). A distinct algorithm from `invert_displacement_field` (Chen et al.):
/// a negated-field first guess refined per voxel. Internal f64; float-exact.
///
/// Args:
///     disp_z, disp_y, disp_x: forward displacement components (world frame).
///     number_of_iterations:   per-voxel line-search iterations (default 5).
///     stop_value:             per-voxel early-stop error threshold (default 0.0).
///
/// Returns:
///     (inv_disp_z, inv_disp_y, inv_disp_x): inverted components.
#[pyfunction]
#[pyo3(signature = (disp_z, disp_y, disp_x, number_of_iterations=5, stop_value=0.0))]
pub fn iterative_inverse_displacement_field(
    py: Python<'_>,
    disp_z: &PyImage,
    disp_y: &PyImage,
    disp_x: &PyImage,
    number_of_iterations: usize,
    stop_value: f64,
) -> (PyImage, PyImage, PyImage) {
    let az = image_from_py(disp_z);
    let ay = image_from_py(disp_y);
    let ax = image_from_py(disp_x);
    let (vx, vy, vz) = py.allow_threads(|| {
        ritk_filter::IterativeInverseDisplacementField {
            number_of_iterations,
            stop_value,
        }
        .apply(&ax, &ay, &az)
    });
    (into_py_image(vz), into_py_image(vy), into_py_image(vx))
}
