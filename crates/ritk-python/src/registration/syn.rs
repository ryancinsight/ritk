//! SyN-family and LDDMM registration: greedy SyN, BSpline FFD, multi-resolution SyN,
//! BSpline SyN, and LDDMM.

use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_registration::bspline_ffd::{BSplineFFDConfig, BSplineFFDRegistration};
use ritk_registration::diffeomorphic::bspline_syn::{BSplineSyNConfig, BSplineSyNRegistration};
use ritk_registration::diffeomorphic::multires_syn::{MultiResSyNConfig, MultiResSyNRegistration};
use ritk_registration::diffeomorphic::{SyNConfig, SyNRegistration};
use ritk_registration::lddmm::{LddmmConfig, LddmmRegistration};

/// Register a moving image to a fixed image using greedy SyN.
///
/// Symmetric Normalization (Avants et al. 2008, *Med. Image Anal.* 12(1):26–41).
/// Maintains forward (fixed→midpoint) and inverse (moving→midpoint) velocity
/// fields that are updated symmetrically at each iteration using the local
/// cross-correlation gradient.
///
/// Args:
///     fixed:                   Fixed (reference) image.
///     moving:                  Moving image.
///     max_iterations:          Maximum iterations (default 100).
///     sigma_smooth:            Velocity field Gaussian smoothing sigma in voxels
///                              (default 3.0).
///     cc_radius:               Local CC window radius in voxels (default 2).
///     gradient_step:           Max per-step voxel displacement for force normalization
///                              (default 0.25, matches ANTs gradientStep).
///     convergence_threshold:   Stop when CC variance over the last
///                              ``convergence_window`` iterations falls below this
///                              value.  Default 1e-8; set smaller to allow more
///                              iterations before declaring convergence.
///
/// Returns:
///     (warped_fixed, warped_moving):
///     - ``warped_fixed``:  fixed image warped to the symmetric midpoint.
///     - ``warped_moving``: moving image warped to the symmetric midpoint.
///       At convergence these two images should be nearly identical.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, max_iterations=100, sigma_smooth=3.0, cc_radius=2, gradient_step=0.25, convergence_threshold=1e-8))]
pub fn syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    sigma_smooth: f64,
    cc_radius: usize,
    gradient_step: f64,
    convergence_threshold: f64,
) -> PyResult<(PyImage, PyImage)> {
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref())?;
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref())?;

    if fixed_shape != moving_shape {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = fixed.inner.origin().clone();
    let fixed_spacing = fixed.inner.spacing().clone();
    let fixed_direction = fixed.inner.direction().clone();
    let moving_origin = moving.inner.origin().clone();
    let moving_spacing = moving.inner.spacing().clone();
    let moving_direction = moving.inner.direction().clone();

    let result = py
        .allow_threads(|| {
            let config = SyNConfig {
                max_iterations,
                sigma_smooth,
                cc_window_radius: cc_radius,
                gradient_step,
                convergence_threshold,
                ..Default::default()
            };
            let reg = SyNRegistration::new(config);
            reg.register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let warped_fixed_img = vec_to_image(
        result.warped_fixed,
        fixed_shape,
        fixed_origin,
        fixed_spacing,
        fixed_direction,
    );
    let warped_moving_img = vec_to_image(
        result.warped_moving,
        fixed_shape,
        moving_origin,
        moving_spacing,
        moving_direction,
    );

    Ok((
        into_py_image(warped_fixed_img),
        into_py_image(warped_moving_img),
    ))
}

/// Register a moving image to a fixed image using BSpline Free-Form Deformation.
///
/// Rueckert et al. (1999) multi-resolution BSpline control lattice with NCC
/// metric and bending energy regularization.
///
/// Args:
///     fixed:                  Fixed (reference) image.
///     moving:                 Moving image.
///     initial_control_spacing: Initial control-point spacing in voxels (default 8).
///     num_levels:             Number of multi-resolution levels (default 3).
///     max_iterations:         Max iterations per level (default 100).
///     learning_rate:          Gradient descent step size (default 0.01).
///     regularization_weight:  Bending energy weight (default 0.001).
///
/// Returns:
///     warped_moving: the moving image warped by the BSpline deformation.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, initial_control_spacing=8, num_levels=3, max_iterations=100, learning_rate=0.01, regularization_weight=0.001))]
pub fn bspline_ffd_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    initial_control_spacing: usize,
    num_levels: usize,
    max_iterations: usize,
    learning_rate: f64,
    regularization_weight: f64,
) -> PyResult<PyImage> {
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref())?;
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref())?;

    if fixed_shape != moving_shape {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = fixed.inner.origin().clone();
    let fixed_spacing = fixed.inner.spacing().clone();
    let fixed_direction = fixed.inner.direction().clone();

    let result = py
        .allow_threads(|| {
            let config = BSplineFFDConfig {
                initial_control_spacing: [
                    initial_control_spacing,
                    initial_control_spacing,
                    initial_control_spacing,
                ],
                num_levels,
                max_iterations_per_level: max_iterations,
                learning_rate,
                regularization_weight,
                ..Default::default()
            };
            BSplineFFDRegistration::register(
                &fixed_vals,
                &moving_vals,
                fixed_shape,
                [1.0, 1.0, 1.0],
                &config,
            )
            .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let warped_image = vec_to_image(
        result.warped_moving,
        fixed_shape,
        fixed_origin,
        fixed_spacing,
        fixed_direction,
    );

    Ok(into_py_image(warped_image))
}

/// Register a moving image to a fixed image using Multi-Resolution SyN.
///
/// Coarse-to-fine symmetric diffeomorphic registration with local
/// cross-correlation metric.  Extends greedy SyN (Avants et al. 2008) with a
/// multi-resolution pyramid for improved capture range and robustness.
///
/// Args:
///     fixed:                   Fixed (reference) image.
///     moving:                  Moving image.
///     num_levels:              Number of resolution levels (default 3).
///     iterations:              Max iterations per level, coarsest first
///                              (default [100, 70, 20] for num_levels=3).
///     sigma_smooth:            Velocity field Gaussian smoothing sigma (default 3.0).
///     cc_radius:               Local CC window radius in voxels (default 2).
///     inverse_consistency:     Enforce inverse consistency (default true).
///     gradient_step:           Max per-step voxel displacement for force normalization
///                              (default 0.25, matches ANTs gradientStep).
///     convergence_threshold:   Stop a level when CC variance over the last
///                              ``convergence_window`` iterations falls below this.
///                              Default 1e-8; set smaller for stricter per-level
///                              convergence checking.
///
/// Returns:
///     (warped_fixed, warped_moving):
///     - ``warped_fixed``:  fixed image warped to the symmetric midpoint.
///     - ``warped_moving``: moving image warped to the symmetric midpoint.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, num_levels=3, iterations=None, sigma_smooth=3.0, cc_radius=2, inverse_consistency=true, gradient_step=0.25, convergence_threshold=1e-8))]
pub fn multires_syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    num_levels: usize,
    iterations: Option<Vec<usize>>,
    sigma_smooth: f64,
    cc_radius: usize,
    inverse_consistency: bool,
    gradient_step: f64,
    convergence_threshold: f64,
) -> PyResult<(PyImage, PyImage)> {
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref())?;
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref())?;

    if fixed_shape != moving_shape {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = fixed.inner.origin().clone();
    let fixed_spacing = fixed.inner.spacing().clone();
    let fixed_direction = fixed.inner.direction().clone();
    let moving_origin = moving.inner.origin().clone();
    let moving_spacing = moving.inner.spacing().clone();
    let moving_direction = moving.inner.direction().clone();

    let result = py
        .allow_threads(|| {
            let iters = iterations.unwrap_or_else(|| vec![100, 70, 20]);
            let config = MultiResSyNConfig {
                num_levels,
                iterations_per_level: iters,
                sigma_smooth,
                convergence_threshold,
                convergence_window: 10,
                n_squarings: 6,
                cc_window_radius: cc_radius,
                enforce_inverse_consistency: inverse_consistency,
                gradient_step,
            };
            let reg = MultiResSyNRegistration::new(config);
            reg.register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let warped_fixed_img = vec_to_image(
        result.warped_fixed,
        fixed_shape,
        fixed_origin,
        fixed_spacing,
        fixed_direction,
    );
    let warped_moving_img = vec_to_image(
        result.warped_moving,
        fixed_shape,
        moving_origin,
        moving_spacing,
        moving_direction,
    );

    Ok((
        into_py_image(warped_fixed_img),
        into_py_image(warped_moving_img),
    ))
}

/// Register a moving image to a fixed image using BSpline SyN.
///
/// Symmetric diffeomorphic registration with BSpline-parameterized velocity
/// fields.  The BSpline representation provides intrinsic smoothness and
/// reduces the number of free parameters relative to dense SyN.
///
/// Args:
///     fixed:                  Fixed (reference) image.
///     moving:                 Moving image.
///     max_iterations:         Maximum iterations (default 100).
///     control_spacing_z:      Control-point spacing in Z (voxels, default 8).
///     control_spacing_y:      Control-point spacing in Y (voxels, default 8).
///     control_spacing_x:      Control-point spacing in X (voxels, default 8).
///     sigma_smooth:           Post-evaluation Gaussian smoothing sigma (default 1.0).
///     cc_radius:              Local CC window radius in voxels (default 2).
///     regularization_weight:  Bending energy weight (default 0.001).
///     gradient_step:  Max per-step voxel displacement for force normalization
///                     (default 0.25, matches ANTs gradientStep).
///
/// Returns:
///     (warped_fixed, warped_moving):
///     - ``warped_fixed``:  fixed image warped to the symmetric midpoint.
///     - ``warped_moving``: moving image warped to the symmetric midpoint.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, max_iterations=100, control_spacing_z=8, control_spacing_y=8, control_spacing_x=8, sigma_smooth=1.0, cc_radius=2, regularization_weight=0.001, gradient_step=0.25))]
pub fn bspline_syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    control_spacing_z: usize,
    control_spacing_y: usize,
    control_spacing_x: usize,
    sigma_smooth: f64,
    cc_radius: usize,
    regularization_weight: f64,
    gradient_step: f64,
) -> PyResult<(PyImage, PyImage)> {
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref())?;
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref())?;

    if fixed_shape != moving_shape {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = fixed.inner.origin().clone();
    let fixed_spacing = fixed.inner.spacing().clone();
    let fixed_direction = fixed.inner.direction().clone();
    let moving_origin = moving.inner.origin().clone();
    let moving_spacing = moving.inner.spacing().clone();
    let moving_direction = moving.inner.direction().clone();

    let result = py
        .allow_threads(|| {
            let config = BSplineSyNConfig {
                max_iterations,
                control_spacing: [control_spacing_z, control_spacing_y, control_spacing_x],
                sigma_smooth,
                convergence_threshold: 1e-6,
                convergence_window: 10,
                n_squarings: 6,
                cc_window_radius: cc_radius,
                regularization_weight,
                gradient_step,
            };
            let reg = BSplineSyNRegistration::new(config);
            reg.register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let warped_fixed_img = vec_to_image(
        result.warped_fixed,
        fixed_shape,
        fixed_origin,
        fixed_spacing,
        fixed_direction,
    );
    let warped_moving_img = vec_to_image(
        result.warped_moving,
        fixed_shape,
        moving_origin,
        moving_spacing,
        moving_direction,
    );

    Ok((
        into_py_image(warped_fixed_img),
        into_py_image(warped_moving_img),
    ))
}

/// Register a moving image to a fixed image using LDDMM.
///
/// Large Deformation Diffeomorphic Metric Mapping (Beg et al. 2005).
/// Generates geodesic paths in the space of diffeomorphisms via shooting
/// from an initial velocity field optimized by gradient descent.
///
/// Args:
///     fixed:                  Fixed (reference) image.
///     moving:                 Moving image.
///     max_iterations:         Maximum outer iterations (default 50).
///     num_time_steps:         Geodesic integration time steps (default 10).
///     kernel_sigma:           RKHS Gaussian kernel width in voxels (default 2.0).
///     learning_rate:          Gradient descent step size (default 0.1).
///     regularization_weight:  Velocity field energy weight λ (default 1.0).
///
/// Returns:
///     (warped_moving, displacement_field):
///     - ``warped_moving``: the moving image warped to the fixed image space.
///     - ``displacement_field``: PyImage with shape [3·Z, Y, X] where the
///       three Z-stacked planes represent (dz, dy, dx) displacement components.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, max_iterations=50, num_time_steps=10, kernel_sigma=2.0, learning_rate=0.1, regularization_weight=1.0))]
pub fn lddmm_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    num_time_steps: usize,
    kernel_sigma: f64,
    learning_rate: f64,
    regularization_weight: f64,
) -> PyResult<(PyImage, PyImage)> {
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref())?;
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref())?;

    if fixed_shape != moving_shape {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = fixed.inner.origin().clone();
    let fixed_spacing = fixed.inner.spacing().clone();
    let fixed_direction = fixed.inner.direction().clone();
    let [nz, ny, nx] = fixed_shape;

    let result = py
        .allow_threads(|| {
            let config = LddmmConfig {
                max_iterations,
                num_time_steps,
                kernel_sigma,
                learning_rate,
                regularization_weight,
                ..Default::default()
            };
            let reg = LddmmRegistration::new(config);
            reg.register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let warped_image = vec_to_image(
        result.warped_moving,
        fixed_shape,
        fixed_origin,
        fixed_spacing,
        fixed_direction,
    );

    let n = nz * ny * nx;
    let mut disp_packed = Vec::with_capacity(3 * n);
    disp_packed.extend_from_slice(&result.displacement_field.0);
    disp_packed.extend_from_slice(&result.displacement_field.1);
    disp_packed.extend_from_slice(&result.displacement_field.2);

    let disp_image = vec_to_image(
        disp_packed,
        [3 * nz, ny, nx],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    Ok((into_py_image(warped_image), into_py_image(disp_image)))
}
