//! SyN-family and LDDMM registration: greedy SyN, BSpline FFD, multi-resolution SyN,
//! BSpline SyN, and LDDMM.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_registration::bspline_ffd::{BSplineFFDConfig, BSplineFFDRegistration};
use ritk_registration::diffeomorphic::bspline_syn::{BSplineSyNConfig, BSplineSyNRegistration};
use ritk_registration::diffeomorphic::multires_syn::{MultiResSyNConfig, MultiResSyNRegistration};
use ritk_registration::diffeomorphic::{SyNConfig, SyNRegistration};
use ritk_registration::lddmm::{LddmmConfig, LddmmRegistration};

/// Configuration options for [`syn_register`].
///
/// Args:
///     max_iterations:        Maximum iterations (default 100).
///     sigma_smooth:          Velocity field Gaussian smoothing sigma in voxels (default 3.0).
///     cc_radius:             Local CC window radius in voxels (default 2).
///     gradient_step:         Max per-step voxel displacement (default 0.25).
///     convergence_threshold: Stop when CC variance over convergence window falls below
///                            this value (default 1e-8).
#[pyclass(name = "SynConfig")]
#[derive(Clone)]
pub struct PySynConfig {
    /// Maximum iterations.
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Velocity field Gaussian smoothing sigma (voxels).
    #[pyo3(get, set)]
    pub sigma_smooth: f64,
    /// Local CC window radius (voxels).
    #[pyo3(get, set)]
    pub cc_radius: usize,
    /// Max per-step voxel displacement for force normalization.
    #[pyo3(get, set)]
    pub gradient_step: f64,
    /// Convergence threshold on CC variance.
    #[pyo3(get, set)]
    pub convergence_threshold: f64,
}

impl Default for PySynConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            sigma_smooth: 3.0,
            cc_radius: 2,
            gradient_step: 0.25,
            convergence_threshold: 1e-8,
        }
    }
}

#[pymethods]
impl PySynConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Register a moving image to a fixed image using greedy SyN.
///
/// Symmetric Normalization (Avants et al. 2008, *Med. Image Anal.* 12(1):26–41).
/// Maintains forward (fixed→midpoint) and inverse (moving→midpoint) velocity
/// fields that are updated symmetrically at each iteration using the local
/// cross-correlation gradient.
///
/// Args:
///     fixed:  Fixed (reference) image.
///     moving: Moving image.
///     opts:   [`SynConfig`] controlling optimizer and convergence parameters.
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
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PySynConfig>,
) -> RitkResult<(PyImage, PyImage)> {
    let opts = opts.unwrap_or_default();
    let max_iterations = opts.max_iterations;
    let sigma_smooth = opts.sigma_smooth;
    let cc_radius = opts.cc_radius;
    let gradient_step = opts.gradient_step;
    let convergence_threshold = opts.convergence_threshold;
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref());
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref());

    if fixed_shape != moving_shape {
        return Err(RitkPyError::runtime(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = *fixed.inner.origin();
    let fixed_spacing = *fixed.inner.spacing();
    let fixed_direction = *fixed.inner.direction();
    let moving_origin = *moving.inner.origin();
    let moving_spacing = *moving.inner.spacing();
    let moving_direction = *moving.inner.direction();

    py
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
        .map_err(RitkPyError::runtime)
        .map(|result| {
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
            (into_py_image(warped_fixed_img), into_py_image(warped_moving_img))
        })
}

/// Configuration options for [`bspline_ffd_register`].
///
/// Args:
///     initial_control_spacing: Initial control-point spacing in voxels (default 8).
///     num_levels:              Number of multi-resolution levels (default 3).
///     max_iterations:          Max iterations per level (default 100).
///     learning_rate:           Gradient descent step size (default 0.01).
///     regularization_weight:   Bending energy weight (default 0.001).
#[pyclass(name = "BSplineFfdConfig")]
#[derive(Clone)]
pub struct PyBSplineFfdConfig {
    /// Initial control-point spacing in voxels.
    #[pyo3(get, set)]
    pub initial_control_spacing: usize,
    /// Number of multi-resolution levels.
    #[pyo3(get, set)]
    pub num_levels: usize,
    /// Max iterations per level.
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Gradient descent step size.
    #[pyo3(get, set)]
    pub learning_rate: f64,
    /// Bending energy regularization weight.
    #[pyo3(get, set)]
    pub regularization_weight: f64,
}

impl Default for PyBSplineFfdConfig {
    fn default() -> Self {
        Self {
            initial_control_spacing: 8,
            num_levels: 3,
            max_iterations: 100,
            learning_rate: 0.01,
            regularization_weight: 0.001,
        }
    }
}

#[pymethods]
impl PyBSplineFfdConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Register a moving image to a fixed image using BSpline Free-Form Deformation.
///
/// Rueckert et al. (1999) multi-resolution BSpline control lattice with NCC
/// metric and bending energy regularization.
///
/// Args:
///     fixed:  Fixed (reference) image.
///     moving: Moving image.
///     opts:   [`BSplineFfdConfig`] controlling control lattice and optimizer parameters.
///
/// Returns:
///     warped_moving: the moving image warped by the BSpline deformation.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn bspline_ffd_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyBSplineFfdConfig>,
) -> RitkResult<PyImage> {
    let opts = opts.unwrap_or_default();
    let initial_control_spacing = opts.initial_control_spacing;
    let num_levels = opts.num_levels;
    let max_iterations = opts.max_iterations;
    let learning_rate = opts.learning_rate;
    let regularization_weight = opts.regularization_weight;
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref());
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref());

    if fixed_shape != moving_shape {
        return Err(RitkPyError::runtime(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = *fixed.inner.origin();
    let fixed_spacing = *fixed.inner.spacing();
    let fixed_direction = *fixed.inner.direction();

    py
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
        .map_err(RitkPyError::runtime)
        .map(|result| {
            into_py_image(vec_to_image(
                result.warped_moving,
                fixed_shape,
                fixed_origin,
                fixed_spacing,
                fixed_direction,
            ))
        })
}

/// Configuration options for [`multires_syn_register`].
#[pyclass(name = "MultiResSynOptions")]
#[derive(Clone)]
pub struct PyMultiresSynOptions {
    /// Number of resolution levels.
    #[pyo3(get, set)]
    pub num_levels: usize,
    /// Max iterations per level, coarsest first; `None` → [100, 70, 20].
    #[pyo3(get, set)]
    pub iterations: Option<Vec<usize>>,
    /// Velocity field Gaussian smoothing sigma (voxels).
    #[pyo3(get, set)]
    pub sigma_smooth: f64,
    /// Local CC window radius (voxels).
    #[pyo3(get, set)]
    pub cc_radius: usize,
    /// Enforce inverse consistency.
    #[pyo3(get, set)]
    pub inverse_consistency: bool,
    /// Max per-step voxel displacement for force normalization.
    #[pyo3(get, set)]
    pub gradient_step: f64,
    /// Per-level convergence threshold on CC variance.
    #[pyo3(get, set)]
    pub convergence_threshold: f64,
}

#[pymethods]
impl PyMultiresSynOptions {
    #[new]
    #[pyo3(signature = (
        num_levels = 3,
        iterations = None,
        sigma_smooth = 3.0,
        cc_radius = 2,
        inverse_consistency = true,
        gradient_step = 0.25,
        convergence_threshold = 1e-8,
    ))]
    pub fn new(
        num_levels: usize,
        iterations: Option<Vec<usize>>,
        sigma_smooth: f64,
        cc_radius: usize,
        inverse_consistency: bool,
        gradient_step: f64,
        convergence_threshold: f64,
    ) -> Self {
        Self { num_levels, iterations, sigma_smooth, cc_radius, inverse_consistency, gradient_step, convergence_threshold }
    }
}

/// Register a moving image to a fixed image using Multi-Resolution SyN.
///
/// Coarse-to-fine symmetric diffeomorphic registration with local
/// cross-correlation metric.  Extends greedy SyN (Avants et al. 2008) with a
/// multi-resolution pyramid for improved capture range and robustness.
///
/// Args:
///     fixed:  Fixed (reference) image.
///     moving: Moving image.
///     opts:   [`MultiResSynOptions`] controlling pyramid configuration.
///
/// Returns:
///     (warped_fixed, warped_moving):
///     - ``warped_fixed``:  fixed image warped to the symmetric midpoint.
///     - ``warped_moving``: moving image warped to the symmetric midpoint.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn multires_syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyMultiresSynOptions>,
) -> RitkResult<(PyImage, PyImage)> {
    let opts = opts.unwrap_or_else(|| PyMultiresSynOptions::new(3, None, 3.0, 2, true, 0.25, 1e-8));
    let num_levels = opts.num_levels;
    let iterations = opts.iterations;
    let sigma_smooth = opts.sigma_smooth;
    let cc_radius = opts.cc_radius;
    let inverse_consistency = opts.inverse_consistency;
    let gradient_step = opts.gradient_step;
    let convergence_threshold = opts.convergence_threshold;
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref());
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref());

    if fixed_shape != moving_shape {
        return Err(RitkPyError::runtime(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = *fixed.inner.origin();
    let fixed_spacing = *fixed.inner.spacing();
    let fixed_direction = *fixed.inner.direction();
    let moving_origin = *moving.inner.origin();
    let moving_spacing = *moving.inner.spacing();
    let moving_direction = *moving.inner.direction();

    py
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
        .map_err(RitkPyError::runtime)
        .map(|result| {
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
            (into_py_image(warped_fixed_img), into_py_image(warped_moving_img))
        })
}

/// Configuration options for [`bspline_syn_register`].
#[pyclass(name = "BSplineSynOptions")]
#[derive(Clone)]
pub struct PyBSplineSynOptions {
    /// Maximum iterations.
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Control-point spacing in Z (voxels).
    #[pyo3(get, set)]
    pub control_spacing_z: usize,
    /// Control-point spacing in Y (voxels).
    #[pyo3(get, set)]
    pub control_spacing_y: usize,
    /// Control-point spacing in X (voxels).
    #[pyo3(get, set)]
    pub control_spacing_x: usize,
    /// Post-evaluation Gaussian smoothing sigma.
    #[pyo3(get, set)]
    pub sigma_smooth: f64,
    /// Local CC window radius (voxels).
    #[pyo3(get, set)]
    pub cc_radius: usize,
    /// Bending energy regularization weight.
    #[pyo3(get, set)]
    pub regularization_weight: f64,
    /// Max per-step voxel displacement for force normalization.
    #[pyo3(get, set)]
    pub gradient_step: f64,
}

impl Default for PyBSplineSynOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            control_spacing_z: 8,
            control_spacing_y: 8,
            control_spacing_x: 8,
            sigma_smooth: 1.0,
            cc_radius: 2,
            regularization_weight: 0.001,
            gradient_step: 0.25,
        }
    }
}

#[pymethods]
impl PyBSplineSynOptions {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Register a moving image to a fixed image using BSpline SyN.
///
/// Symmetric diffeomorphic registration with BSpline-parameterized velocity
/// fields.  The BSpline representation provides intrinsic smoothness and
/// reduces the number of free parameters relative to dense SyN.
///
/// Args:
///     fixed:  Fixed (reference) image.
///     moving: Moving image.
///     opts:   [`BSplineSynOptions`] controlling BSpline lattice and SyN configuration.
///
/// Returns:
///     (warped_fixed, warped_moving):
///     - ``warped_fixed``:  fixed image warped to the symmetric midpoint.
///     - ``warped_moving``: moving image warped to the symmetric midpoint.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn bspline_syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyBSplineSynOptions>,
) -> RitkResult<(PyImage, PyImage)> {
    let opts = opts.unwrap_or_default();
    let max_iterations = opts.max_iterations;
    let control_spacing_z = opts.control_spacing_z;
    let control_spacing_y = opts.control_spacing_y;
    let control_spacing_x = opts.control_spacing_x;
    let sigma_smooth = opts.sigma_smooth;
    let cc_radius = opts.cc_radius;
    let regularization_weight = opts.regularization_weight;
    let gradient_step = opts.gradient_step;
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref());
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref());

    if fixed_shape != moving_shape {
        return Err(RitkPyError::runtime(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = *fixed.inner.origin();
    let fixed_spacing = *fixed.inner.spacing();
    let fixed_direction = *fixed.inner.direction();
    let moving_origin = *moving.inner.origin();
    let moving_spacing = *moving.inner.spacing();
    let moving_direction = *moving.inner.direction();

    py
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
        .map_err(RitkPyError::runtime)
        .map(|result| {
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
            (into_py_image(warped_fixed_img), into_py_image(warped_moving_img))
        })
}

/// Configuration options for [`lddmm_register`].
///
/// Args:
///     max_iterations:        Maximum outer iterations (default 50).
///     num_time_steps:        Geodesic integration time steps (default 10).
///     kernel_sigma:          RKHS Gaussian kernel width in voxels (default 2.0).
///     learning_rate:         Gradient descent step size (default 0.1).
///     regularization_weight: Velocity field energy weight λ (default 1.0).
#[pyclass(name = "LddmmConfig")]
#[derive(Clone)]
pub struct PyLddmmConfig {
    /// Maximum outer iterations.
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Geodesic integration time steps.
    #[pyo3(get, set)]
    pub num_time_steps: usize,
    /// RKHS Gaussian kernel width in voxels.
    #[pyo3(get, set)]
    pub kernel_sigma: f64,
    /// Gradient descent step size.
    #[pyo3(get, set)]
    pub learning_rate: f64,
    /// Velocity field energy weight λ.
    #[pyo3(get, set)]
    pub regularization_weight: f64,
}

impl Default for PyLddmmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            num_time_steps: 10,
            kernel_sigma: 2.0,
            learning_rate: 0.1,
            regularization_weight: 1.0,
        }
    }
}

#[pymethods]
impl PyLddmmConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Register a moving image to a fixed image using LDDMM.
///
/// Large Deformation Diffeomorphic Metric Mapping (Beg et al. 2005).
/// Generates geodesic paths in the space of diffeomorphisms via shooting
/// from an initial velocity field optimized by gradient descent.
///
/// Args:
///     fixed:  Fixed (reference) image.
///     moving: Moving image.
///     opts:   [`LddmmConfig`] controlling geodesic integration and optimizer parameters.
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
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn lddmm_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyLddmmConfig>,
) -> RitkResult<(PyImage, PyImage)> {
    let opts = opts.unwrap_or_default();
    let max_iterations = opts.max_iterations;
    let num_time_steps = opts.num_time_steps;
    let kernel_sigma = opts.kernel_sigma;
    let learning_rate = opts.learning_rate;
    let regularization_weight = opts.regularization_weight;
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref());
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref());

    if fixed_shape != moving_shape {
        return Err(RitkPyError::runtime(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = *fixed.inner.origin();
    let fixed_spacing = *fixed.inner.spacing();
    let fixed_direction = *fixed.inner.direction();
    let [nz, ny, nx] = fixed_shape;

    py
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
        .map_err(RitkPyError::runtime)
        .map(|result| {
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
            (into_py_image(warped_image), into_py_image(disp_image))
        })
}
