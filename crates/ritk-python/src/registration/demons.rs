//! Demons-family registration algorithms: Thirion, Diffeomorphic, and Symmetric Demons.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::GaussianSigma;
use ritk_registration::demons::{
    DemonsConfig, DiffeomorphicDemonsRegistration, SymmetricDemonsRegistration,
    ThirionDemonsRegistration,
};
use ritk_spatial::{Direction, Point, Spacing};

/// Register a moving image to a fixed image using Thirion's Demons algorithm.
///
/// Delegates to [`ThirionDemonsRegistration`] from `ritk-registration`.
///
/// Args:
///     fixed:            Fixed (reference) image.
///     moving:           Moving image to register to the fixed image.
///     max_iterations:   Number of Demons iterations (default 50).
///     sigma_diffusion:  Displacement field smoothing sigma in voxels (default 1.0).
///
/// Returns:
///     (warped_moving, displacement_field):
///     - `warped_moving`: the moving image warped by the final displacement field,
///       with the same shape and spatial metadata as `fixed`.
///     - `displacement_field`: PyImage with shape [3·Z, Y, X] where the three
///       Z-stacked planes represent (dz, dy, dx) displacement components.
///       The user can recover components with `.to_numpy().reshape(3, Z, Y, X)`.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, max_iterations=50, sigma_diffusion=1.0))]
pub fn demons_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    sigma_diffusion: f64,
) -> RitkResult<(PyImage, PyImage)> {
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
        let config = DemonsConfig {
            max_iterations,
            sigma_diffusion: GaussianSigma::new(sigma_diffusion),
            sigma_fluid: None,
            max_step_length: 2.0,
        };
        let reg = ThirionDemonsRegistration::new(config);
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

/// Register a moving image to a fixed image using Diffeomorphic Demons.
///
/// Uses a stationary velocity field with scaling-and-squaring to guarantee
/// invertibility of the displacement field (Vercauteren et al. 2009,
/// *NeuroImage* 45(S1):S61–S72).
///
/// Args:
///     fixed:            Fixed (reference) image.
///     moving:           Moving image to register to the fixed image.
///     max_iterations:   Number of iterations (default 50).
///     sigma_diffusion:  Velocity field Gaussian smoothing sigma in voxels
///                       (default 1.5).
///     n_squarings:      Scaling-and-squaring steps for exp(v) (default 6 = 64
///                       integration steps).
///
/// Returns:
///     (warped_moving, displacement_field) — same convention as demons_register.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, max_iterations=50, sigma_diffusion=1.5, n_squarings=6))]
pub fn diffeomorphic_demons_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    sigma_diffusion: f64,
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
        let config = DemonsConfig {
            max_iterations,
            sigma_diffusion: GaussianSigma::new(sigma_diffusion),
            sigma_fluid: None,
            max_step_length: 2.0,
        };
        let reg = DiffeomorphicDemonsRegistration {
            config,
            n_squarings,
        };
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

/// Register a moving image to a fixed image using Symmetric Demons.
///
/// Uses gradient information from both fixed and warped moving images, making
/// the algorithm approximately symmetric with respect to swapping fixed and
/// moving (Pennec et al. 1999, *MICCAI* LNCS 1679:597–605).
///
/// Args:
///     fixed:            Fixed (reference) image.
///     moving:           Moving image.
///     max_iterations:   Number of iterations (default 50).
///     sigma_diffusion:  Displacement field Gaussian smoothing sigma in voxels
///                       (default 1.5).
///
/// Returns:
///     (warped_moving, displacement_field) — same convention as demons_register.
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, max_iterations=50, sigma_diffusion=1.5))]
pub fn symmetric_demons_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    sigma_diffusion: f64,
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
        let config = DemonsConfig {
            max_iterations,
            sigma_diffusion: GaussianSigma::new(sigma_diffusion),
            sigma_fluid: None,
            max_step_length: 2.0,
        };
        let reg = SymmetricDemonsRegistration::new(config);
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
