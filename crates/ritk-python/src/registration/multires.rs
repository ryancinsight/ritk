//! Multi-resolution and inverse-consistent Demons registration algorithms.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::GaussianSigma;
use ritk_registration::demons::{
    DemonsConfig, DemonsVariant, InverseConsistentDemonsConfig,
    InverseConsistentDiffeomorphicDemonsRegistration, MultiResDemonsConfig,
    MultiResDemonsRegistration,
};
use ritk_spatial::{Direction, Point, Spacing};

/// Configuration options for [`multires_demons_register`].
#[pyclass(name = "MultiResDemonsOptions")]
#[derive(Clone)]
pub struct PyMultiresDemonsOptions {
    /// Base iteration count (scaled per pyramid level).
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Displacement field Gaussian smoothing sigma in voxels.
    #[pyo3(get, set)]
    pub sigma_diffusion: f64,
    /// Number of pyramid levels >= 1.
    #[pyo3(get, set)]
    pub levels: usize,
    /// Demons variant: "thirion" (classic) or "diffeomorphic".
    #[pyo3(get, set)]
    pub variant: String,
    /// Scaling-and-squaring steps when variant=diffeomorphic.
    #[pyo3(get, set)]
    pub n_squarings: usize,
}

#[pymethods]
impl PyMultiresDemonsOptions {
    #[new]
    #[pyo3(signature = (
        max_iterations = 50,
        sigma_diffusion = 1.0,
        levels = 3,
        variant = "thirion",
        n_squarings = 6,
    ))]
    pub fn new(
        max_iterations: usize,
        sigma_diffusion: f64,
        levels: usize,
        variant: &str,
        n_squarings: usize,
    ) -> PyResult<Self> {
        let normalized = variant.to_lowercase();
        match normalized.as_str() {
            "thirion" | "classic" | "diffeomorphic" => {}
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid Demons variant '{other}'. Expected 'thirion' or 'diffeomorphic'."
                )))
            }
        };
        Ok(Self {
            max_iterations,
            sigma_diffusion,
            levels,
            variant: normalized,
            n_squarings,
        })
    }
}

/// Register a moving image to a fixed image using multi-resolution Demons.
///
/// Coarse-to-fine pyramid with warm-started displacement injection.
/// Supports both Thirion (classic) and Diffeomorphic variants.
///
/// Args:
///     fixed: Fixed (reference) image.
///     moving: Moving image to register.
///     opts: `MultiResDemonsOptions` controlling pyramid and algorithm variant.
///
/// Returns:
///     (warped_moving, displacement_field) â€” same convention as demons_register.
///     displacement_field has shape [3Â·Z, Y, X].
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts = None))]
pub fn multires_demons_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: Option<PyMultiresDemonsOptions>,
) -> RitkResult<(PyImage, PyImage)> {
    let opts =
        opts.unwrap_or_else(|| PyMultiresDemonsOptions::new(50, 1.0, 3, "thirion", 6).unwrap());
    let max_iterations = opts.max_iterations;
    let sigma_diffusion = opts.sigma_diffusion;
    let levels = opts.levels;
    let n_squarings = opts.n_squarings;
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref());
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref());
    if fixed_shape != moving_shape {
        return Err(RitkPyError::runtime(format!(
            "fixed shape {:?} != moving shape {:?}; images must have identical shapes",
            fixed_shape, moving_shape
        )));
    }
    let fixed_origin = *fixed.inner.origin();
    let fixed_spacing = *fixed.inner.spacing();
    let fixed_direction = *fixed.inner.direction();
    let [nz, ny, nx] = fixed_shape;
    py.allow_threads(|| {
        let variant = match opts.variant.as_str() {
            "diffeomorphic" => DemonsVariant::Diffeomorphic,
            _ => DemonsVariant::Classic,
        };
        let config = MultiResDemonsConfig {
            base_config: DemonsConfig {
                max_iterations,
                sigma_diffusion: GaussianSigma::new(sigma_diffusion),
                sigma_fluid: None,
                max_step_length: 2.0,
            },
            levels,
            variant,
            n_squarings,
        };
        MultiResDemonsRegistration::new(config)
            .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
            .map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(|result| {
        let warped_image = vec_to_image(
            result.warped,
            fixed_shape,
            fixed_origin,
            fixed_spacing,
            fixed_direction,
        );
        let n = nz * ny * nx;
        let mut disp_packed = Vec::with_capacity(3 * n);
        disp_packed.extend_from_slice(&result.disp_z);
        disp_packed.extend_from_slice(&result.disp_y);
        disp_packed.extend_from_slice(&result.disp_x);
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

/// Register a moving image to a fixed image using inverse-consistent
/// diffeomorphic Demons.
///
/// Uses a stationary velocity field `v` and maintains the forward transform
/// `exp(v)` and exact inverse transform `exp(-v)` throughout optimization.
/// The bilateral objective is:
/// E(v) = (1-w) ||F - M o exp(v)||Â² + w ||M - F o exp(-v)||Â²
///
/// Args:
///     fixed: Fixed (reference) image.
///     moving: Moving image to register to the fixed image.
///     max_iterations: Number of iterations (default 50).
///     sigma_diffusion: Velocity field Gaussian smoothing sigma in
///         voxels (default 1.5).
///     inverse_consistency_weight: Backward-force weight w in [0, 1]
///         (default 0.5).
///     n_squarings: Scaling-and-squaring steps for exp(v)
///         and exp(-v) (default 6).
///
/// Returns:
///     (warped_moving, displacement_field) â€” same convention as
///     `demons_register`.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, max_iterations=50, sigma_diffusion=1.5, inverse_consistency_weight=0.5, n_squarings=6))]
pub fn inverse_consistent_demons_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    sigma_diffusion: f64,
    inverse_consistency_weight: f64,
    n_squarings: usize,
) -> RitkResult<(PyImage, PyImage)> {
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
    py.allow_threads(|| {
        let config = InverseConsistentDemonsConfig {
            demons: DemonsConfig {
                max_iterations,
                sigma_diffusion: GaussianSigma::new(sigma_diffusion),
                sigma_fluid: None,
                max_step_length: 2.0,
            },
            inverse_consistency_weight,
            n_squarings,
        };
        let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(config);
        reg.register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
            .map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(|result| {
        let warped_image = vec_to_image(
            result.warped,
            fixed_shape,
            fixed_origin,
            fixed_spacing,
            fixed_direction,
        );
        let n = nz * ny * nx;
        let mut disp_packed = Vec::with_capacity(3 * n);
        disp_packed.extend_from_slice(&result.disp_z);
        disp_packed.extend_from_slice(&result.disp_y);
        disp_packed.extend_from_slice(&result.disp_x);
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
