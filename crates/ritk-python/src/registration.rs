//! Python-exposed deformable image registration algorithms.
//!
//! All registration functions delegate to `ritk-registration` crate
//! implementations.  This module handles PyO3 boundary conversion
//! (PyImage ↔ flat `Vec<f32>`) and result packing only.
//!
//! # Supported algorithms
//! - **Thirion Demons** (`demons_register`) — `ThirionDemonsRegistration`
//! - **Diffeomorphic Demons** (`diffeomorphic_demons_register`) — `DiffeomorphicDemonsRegistration`
//! - **Symmetric Demons** (`symmetric_demons_register`) — `SymmetricDemonsRegistration`
//! - **Greedy SyN** (`syn_register`) — `SyNRegistration`

use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_registration::demons::{
    DemonsConfig, DiffeomorphicDemonsRegistration, SymmetricDemonsRegistration,
    ThirionDemonsRegistration,
};
use ritk_registration::diffeomorphic::{SyNConfig, SyNRegistration};

// ── demons_register ───────────────────────────────────────────────────────────

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

    let [nz, ny, nx] = fixed_shape;
    let config = DemonsConfig {
        max_iterations,
        sigma_diffusion,
        sigma_fluid: 0.0,
        max_step_length: 2.0,
    };
    let reg = ThirionDemonsRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let warped_image = vec_to_image(
        result.warped,
        fixed_shape,
        fixed.inner.origin().clone(),
        fixed.inner.spacing().clone(),
        fixed.inner.direction().clone(),
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

// ── diffeomorphic_demons_register ─────────────────────────────────────────────

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

    let [nz, ny, nx] = fixed_shape;
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
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let warped_image = vec_to_image(
        result.warped,
        fixed_shape,
        fixed.inner.origin().clone(),
        fixed.inner.spacing().clone(),
        fixed.inner.direction().clone(),
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

// ── symmetric_demons_register ─────────────────────────────────────────────────

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

    let [nz, ny, nx] = fixed_shape;
    let config = DemonsConfig {
        max_iterations,
        sigma_diffusion,
        sigma_fluid: 0.0,
        max_step_length: 2.0,
    };
    let reg = SymmetricDemonsRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let warped_image = vec_to_image(
        result.warped,
        fixed_shape,
        fixed.inner.origin().clone(),
        fixed.inner.spacing().clone(),
        fixed.inner.direction().clone(),
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

// ── syn_register ──────────────────────────────────────────────────────────────

/// Register a moving image to a fixed image using greedy SyN.
///
/// Symmetric Normalization (Avants et al. 2008, *Med. Image Anal.* 12(1):26–41).
/// Maintains forward (fixed→midpoint) and inverse (moving→midpoint) velocity
/// fields that are updated symmetrically at each iteration using the local
/// cross-correlation gradient.
///
/// Args:
///     fixed:          Fixed (reference) image.
///     moving:         Moving image.
///     max_iterations: Maximum iterations (default 100).
///     sigma_smooth:   Velocity field Gaussian smoothing sigma in voxels
///                     (default 3.0).
///     cc_radius:      Local CC window radius in voxels (default 2).
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
#[pyo3(signature = (fixed, moving, max_iterations=100, sigma_smooth=3.0, cc_radius=2))]
pub fn syn_register(
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    sigma_smooth: f64,
    cc_radius: usize,
) -> PyResult<(PyImage, PyImage)> {
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref())?;
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref())?;

    if fixed_shape != moving_shape {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    let config = SyNConfig {
        max_iterations,
        sigma_smooth,
        cc_window_radius: cc_radius,
        ..Default::default()
    };
    let reg = SyNRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let warped_fixed_img = vec_to_image(
        result.warped_fixed,
        fixed_shape,
        fixed.inner.origin().clone(),
        fixed.inner.spacing().clone(),
        fixed.inner.direction().clone(),
    );
    let warped_moving_img = vec_to_image(
        result.warped_moving,
        fixed_shape,
        moving.inner.origin().clone(),
        moving.inner.spacing().clone(),
        moving.inner.direction().clone(),
    );

    Ok((
        into_py_image(warped_fixed_img),
        into_py_image(warped_moving_img),
    ))
}

// ── Submodule registration ────────────────────────────────────────────────────

/// Register the `registration` submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "registration")?;
    m.add_function(wrap_pyfunction!(demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(diffeomorphic_demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(symmetric_demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(syn_register, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
