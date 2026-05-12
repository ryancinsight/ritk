//! Demons-family registration algorithms: Thirion, Diffeomorphic, Symmetric, Multi-resolution,
//! and Inverse-consistent Diffeomorphic Demons.

use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_registration::demons::{
    DemonsConfig, DiffeomorphicDemonsRegistration, InverseConsistentDemonsConfig,
    InverseConsistentDiffeomorphicDemonsRegistration, MultiResDemonsConfig,
    MultiResDemonsRegistration, SymmetricDemonsRegistration, ThirionDemonsRegistration,
};

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
) -> PyResult<(PyImage, PyImage)> {
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref())?;
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref())?;

    if fixed_shape != moving_shape {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fixed shape {:?} != moving shape {:?}; images must have identical shapes",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = fixed.inner.origin().clone();
    let fixed_spacing = fixed.inner.spacing().clone();
    let fixed_direction = fixed.inner.direction().clone();
    let [nz, ny, nx] = fixed_shape;

    let result = py
        .allow_threads(|| {
            let config = DemonsConfig {
                max_iterations,
                sigma_diffusion,
                sigma_fluid: 0.0,
                max_step_length: 2.0,
            };
            let reg = ThirionDemonsRegistration::new(config);
            reg.register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

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

    Ok((into_py_image(warped_image), into_py_image(disp_image)))
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
            let config = DemonsConfig {
                max_iterations,
                sigma_diffusion,
                sigma_fluid: 0.0,
                max_step_length: 2.0,
            };
            let reg = DiffeomorphicDemonsRegistration {
                config,
                n_squarings,
            };
            reg.register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

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

    Ok((into_py_image(warped_image), into_py_image(disp_image)))
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
            let config = DemonsConfig {
                max_iterations,
                sigma_diffusion,
                sigma_fluid: 0.0,
                max_step_length: 2.0,
            };
            let reg = SymmetricDemonsRegistration::new(config);
            reg.register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

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

    Ok((into_py_image(warped_image), into_py_image(disp_image)))
}

/// Register a moving image to a fixed image using multi-resolution Demons.
///
/// Coarse-to-fine pyramid with warm-started displacement injection.
/// Supports both Thirion (classic) and Diffeomorphic variants.
///
/// Args:
///     fixed:              Fixed (reference) image.
///     moving:             Moving image to register.
///     max_iterations:     Base iteration count (scaled per pyramid level, default 50).
///     sigma_diffusion:    Displacement field Gaussian smoothing sigma in voxels (default 1.0).
///     levels:             Number of pyramid levels >= 1 (default 3). With 3 levels,
///                         downsampling factors are [4, 2, 1].
///     use_diffeomorphic:  When True, use DiffeomorphicDemons at each level (default False).
///     n_squarings:        Scaling-and-squaring steps when use_diffeomorphic=True (default 6).
///
/// Returns:
///     (warped_moving, displacement_field) — same convention as demons_register.
///     displacement_field has shape [3·Z, Y, X].
///
/// Raises:
///     RuntimeError: if image shapes do not match or registration fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, max_iterations=50, sigma_diffusion=1.0, levels=3, use_diffeomorphic=false, n_squarings=6))]
pub fn multires_demons_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    sigma_diffusion: f64,
    levels: usize,
    use_diffeomorphic: bool,
    n_squarings: usize,
) -> PyResult<(PyImage, PyImage)> {
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref())?;
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref())?;

    if fixed_shape != moving_shape {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fixed shape {:?} != moving shape {:?}; images must have identical shapes",
            fixed_shape, moving_shape
        )));
    }

    let fixed_origin = fixed.inner.origin().clone();
    let fixed_spacing = fixed.inner.spacing().clone();
    let fixed_direction = fixed.inner.direction().clone();
    let [nz, ny, nx] = fixed_shape;

    let result = py
        .allow_threads(|| {
            let config = MultiResDemonsConfig {
                base_config: DemonsConfig {
                    max_iterations,
                    sigma_diffusion,
                    sigma_fluid: 0.0,
                    max_step_length: 2.0,
                },
                levels,
                use_diffeomorphic,
                n_squarings,
            };
            MultiResDemonsRegistration::new(config)
                .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

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

    Ok((into_py_image(warped_image), into_py_image(disp_image)))
}

/// Register a moving image to a fixed image using inverse-consistent
/// diffeomorphic Demons.
///
/// Uses a stationary velocity field `v` and maintains the forward transform
/// `exp(v)` and exact inverse transform `exp(-v)` throughout optimization.
/// The bilateral objective is:
///   E(v) = (1-w) ||F - M o exp(v)||² + w ||M - F o exp(-v)||²
///
/// Args:
///     fixed:                       Fixed (reference) image.
///     moving:                      Moving image to register to the fixed image.
///     max_iterations:              Number of iterations (default 50).
///     sigma_diffusion:             Velocity field Gaussian smoothing sigma in
///                                  voxels (default 1.5).
///     inverse_consistency_weight:  Backward-force weight w in [0, 1]
///                                  (default 0.5).
///     n_squarings:                 Scaling-and-squaring steps for exp(v)
///                                  and exp(-v) (default 6).
///
/// Returns:
///     (warped_moving, displacement_field) — same convention as
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
            let config = InverseConsistentDemonsConfig {
                demons: DemonsConfig {
                    max_iterations,
                    sigma_diffusion,
                    sigma_fluid: 0.0,
                    max_step_length: 2.0,
                },
                inverse_consistency_weight,
                n_squarings,
            };
            let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(config);
            reg.register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

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

    Ok((into_py_image(warped_image), into_py_image(disp_image)))
}
