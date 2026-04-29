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
//! - **Multi-Resolution Demons** (`multires_demons_register`) — `MultiResDemonsRegistration`

use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_registration::atlas::label_fusion::{
    joint_label_fusion, majority_vote, LabelFusionConfig,
};
use ritk_registration::atlas::{AtlasConfig, AtlasRegistration};
use ritk_registration::bspline_ffd::{BSplineFFDConfig, BSplineFFDRegistration};
use ritk_registration::demons::{
    DemonsConfig, DiffeomorphicDemonsRegistration, InverseConsistentDemonsConfig,
    InverseConsistentDiffeomorphicDemonsRegistration, MultiResDemonsConfig,
    MultiResDemonsRegistration, SymmetricDemonsRegistration, ThirionDemonsRegistration,
};
use ritk_registration::diffeomorphic::bspline_syn::{BSplineSyNConfig, BSplineSyNRegistration};
use ritk_registration::diffeomorphic::multires_syn::{MultiResSyNConfig, MultiResSyNRegistration};
use ritk_registration::diffeomorphic::{SyNConfig, SyNRegistration};
use ritk_registration::lddmm::{LddmmConfig, LddmmRegistration};

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

// ── multires_demons_register ─────────────────────────────────────────────────

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
// ── inverse_consistent_demons_register ───────────────────────────────────────

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
///     gradient_step:  Max per-step voxel displacement for force normalization
///                     (default 0.25, matches ANTs gradientStep).
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
#[pyo3(signature = (fixed, moving, max_iterations=100, sigma_smooth=3.0, cc_radius=2, gradient_step=0.25))]
pub fn syn_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    sigma_smooth: f64,
    cc_radius: usize,
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
            let config = SyNConfig {
                max_iterations,
                sigma_smooth,
                cc_window_radius: cc_radius,
                gradient_step,
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

// ── bspline_ffd_register ──────────────────────────────────────────────────────

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

// ── multires_syn_register ─────────────────────────────────────────────────────

/// Register a moving image to a fixed image using Multi-Resolution SyN.
///
/// Coarse-to-fine symmetric diffeomorphic registration with local
/// cross-correlation metric.  Extends greedy SyN (Avants et al. 2008) with a
/// multi-resolution pyramid for improved capture range and robustness.
///
/// Args:
///     fixed:          Fixed (reference) image.
///     moving:         Moving image.
///     num_levels:     Number of resolution levels (default 3).
///     iterations:     Comma-separated or list of max iterations per level,
///                     coarsest first (default [100, 70, 20]).
///     sigma_smooth:   Velocity field Gaussian smoothing sigma (default 3.0).
///     cc_radius:      Local CC window radius in voxels (default 2).
///     inverse_consistency: Enforce inverse consistency (default true).
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
#[pyo3(signature = (fixed, moving, num_levels=3, iterations=None, sigma_smooth=3.0, cc_radius=2, inverse_consistency=true, gradient_step=0.25))]
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
                convergence_threshold: 1e-6,
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

// ── bspline_syn_register ──────────────────────────────────────────────────────

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

// ── lddmm_register ───────────────────────────────────────────────────────────

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

// ── build_atlas ───────────────────────────────────────────────────────────────

/// Build a population-specific atlas template from multiple subject images.
///
/// Iteratively registers all subjects to a mean template using Multi-Resolution
/// SyN, refines the template, and applies mean-drift sharpening until
/// convergence.
///
/// Args:
///     subjects:               List of subject images (all must share the same shape).
///     max_iterations:         Maximum outer template-building iterations (default 5).
///     convergence_threshold:  Per-voxel RMS change threshold for early stopping (default 0.01).
///     syn_iterations:         Per-level SyN iteration counts (default [100, 70, 20]).
///     sigma_smooth:           Gaussian smoothing sigma for velocity fields (default 3.0).
///     cc_radius:              Cross-correlation window radius (default 2).
///
/// Returns:
///     (template, convergence_history):
/// - `template`: the final atlas template image with spatial metadata from
/// the first subject.
/// - `convergence_history`: per-iteration RMS change values.
///
/// Args:
/// subjects: list of subject images (must share shape/dtype).
/// max_iterations: outer atlas convergence iterations (default 5).
/// convergence_threshold: RMS change threshold for atlas convergence (default 0.01).
/// syn_iterations: per-level SyN iterations; defaults to [100, 70, 20].
/// sigma_smooth: Gaussian regularisation sigma for SyN velocity fields (default 3.0).
/// cc_radius: local cross-correlation window radius (default 2).
/// gradient_step: maximum per-iteration displacement in voxels (inf-norm, default 0.25).
///
/// Raises:
/// RuntimeError: if subjects is empty, shapes mismatch, or registration fails.
#[pyfunction]
#[pyo3(signature = (subjects, max_iterations=5, convergence_threshold=0.01, syn_iterations=None, sigma_smooth=3.0, cc_radius=2, gradient_step=0.25))]
pub fn build_atlas(
    py: Python<'_>,
    subjects: Vec<Py<PyImage>>,
    max_iterations: usize,
    convergence_threshold: f64,
    syn_iterations: Option<Vec<usize>>,
    sigma_smooth: f64,
    cc_radius: usize,
    gradient_step: f64,
) -> PyResult<(PyImage, Vec<f64>)> {
    if subjects.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "subjects list is empty; at least one subject is required",
        ));
    }

    // Extract voxel data and validate uniform shape before releasing the GIL.
    let mut subject_vecs: Vec<Vec<f32>> = Vec::with_capacity(subjects.len());
    let (first_vals, first_shape) = image_to_vec(subjects[0].borrow(py).inner.as_ref())?;
    subject_vecs.push(first_vals);

    for (i, subj) in subjects.iter().enumerate().skip(1) {
        let (vals, shape) = image_to_vec(subj.borrow(py).inner.as_ref())?;
        if shape != first_shape {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "subject[{}] shape {:?} != subject[0] shape {:?}",
                i, shape, first_shape
            )));
        }
        subject_vecs.push(vals);
    }

    // Heavy atlas-building computation — release the GIL for the duration.
    // `subject_vecs` is moved into the closure; slices are constructed inside
    // to avoid cross-boundary lifetime issues.
    let result = py
        .allow_threads(|| {
            let subject_slices: Vec<&[f32]> = subject_vecs.iter().map(|v| v.as_slice()).collect();
            let syn_config = MultiResSyNConfig {
                num_levels: 3,
                iterations_per_level: syn_iterations.unwrap_or_else(|| vec![100, 70, 20]),
                sigma_smooth,
                convergence_threshold: 1e-6,
                convergence_window: 10,
                n_squarings: 6,
                cc_window_radius: cc_radius,
                enforce_inverse_consistency: true,
                gradient_step,
            };
            let config = AtlasConfig {
                max_iterations,
                convergence_threshold,
                syn_config,
            };
            let reg = AtlasRegistration::new(config);
            reg.build_atlas(&subject_slices, first_shape, [1.0, 1.0, 1.0])
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // GIL is re-acquired here; borrow(py) is valid again.
    let template_image = vec_to_image(
        result.template,
        first_shape,
        subjects[0].borrow(py).inner.origin().clone(),
        subjects[0].borrow(py).inner.spacing().clone(),
        subjects[0].borrow(py).inner.direction().clone(),
    );

    Ok((into_py_image(template_image), result.convergence_history))
}

// ── majority_vote_fusion ──────────────────────────────────────────────────────

/// Fuse multiple atlas label maps via majority voting.
///
/// For each voxel the output label is the mode across all atlas label maps.
/// Ties are broken by selecting the smallest label value.
///
/// Atlas label images are expected to contain integer labels stored as f32.
/// Values are rounded to the nearest u32 before voting.
///
/// Args:
///     atlas_labels: List of label map images (all must share the same shape).
///
/// Returns:
///     (labels, confidence):
///     - `labels`: fused label map (u32 labels stored as f32).
///     - `confidence`: per-voxel fraction of atlases voting for the winning label.
///
/// Raises:
///     RuntimeError: if atlas_labels is empty or shapes mismatch.
#[pyfunction]
pub fn majority_vote_fusion(
    py: Python<'_>,
    atlas_labels: Vec<Py<PyImage>>,
) -> PyResult<(PyImage, PyImage)> {
    if atlas_labels.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "atlas_labels list is empty; at least one atlas is required",
        ));
    }

    // Extract label data and validate uniform shape before releasing the GIL.
    let mut label_vecs: Vec<Vec<u32>> = Vec::with_capacity(atlas_labels.len());
    let (first_vals, first_shape) = image_to_vec(atlas_labels[0].borrow(py).inner.as_ref())?;
    label_vecs.push(first_vals.iter().map(|&v| v.round() as u32).collect());

    for (i, lbl) in atlas_labels.iter().enumerate().skip(1) {
        let (vals, shape) = image_to_vec(lbl.borrow(py).inner.as_ref())?;
        if shape != first_shape {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "atlas_labels[{}] shape {:?} != atlas_labels[0] shape {:?}",
                i, shape, first_shape
            )));
        }
        label_vecs.push(vals.iter().map(|&v| v.round() as u32).collect());
    }

    // Heavy voting computation — release the GIL for the duration.
    // `label_vecs` is moved into the closure; slices are constructed inside.
    let result = py
        .allow_threads(|| {
            let label_slices: Vec<&[u32]> = label_vecs.iter().map(|v| v.as_slice()).collect();
            majority_vote(&label_slices, first_shape).map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // GIL is re-acquired here; borrow(py) is valid again.
    let labels_f32: Vec<f32> = result.labels.iter().map(|&l| l as f32).collect();

    let labels_image = vec_to_image(
        labels_f32,
        first_shape,
        atlas_labels[0].borrow(py).inner.origin().clone(),
        atlas_labels[0].borrow(py).inner.spacing().clone(),
        atlas_labels[0].borrow(py).inner.direction().clone(),
    );
    let confidence_image = vec_to_image(
        result.confidence,
        first_shape,
        atlas_labels[0].borrow(py).inner.origin().clone(),
        atlas_labels[0].borrow(py).inner.spacing().clone(),
        atlas_labels[0].borrow(py).inner.direction().clone(),
    );

    Ok((into_py_image(labels_image), into_py_image(confidence_image)))
}

// ── joint_label_fusion_py ─────────────────────────────────────────────────────

/// Fuse atlas label maps using the Joint Label Fusion (JLF) algorithm.
///
/// Weighted voting where per-voxel weights are derived from local patch
/// intensity similarity between warped atlas images and the target image.
///
/// Args:
///     target:       Target intensity image (the image being segmented).
///     atlas_images: List of warped atlas intensity images registered to target space.
///     atlas_labels: List of atlas label maps registered to target space.
///     patch_radius: Local patch radius for similarity computation (default 2).
///     beta:         Regularization parameter for the pairwise similarity matrix (default 0.1).
///
/// Returns:
///     (labels, confidence):
///     - `labels`: fused label map (u32 labels stored as f32).
///     - `confidence`: per-voxel sum of JLF weights assigned to the winning label.
///
/// Raises:
///     RuntimeError: if atlas counts mismatch, shapes mismatch, or fusion fails.
#[pyfunction]
#[pyo3(signature = (target, atlas_images, atlas_labels, patch_radius=2, beta=0.1))]
pub fn joint_label_fusion_py(
    py: Python<'_>,
    target: &PyImage,
    atlas_images: Vec<Py<PyImage>>,
    atlas_labels: Vec<Py<PyImage>>,
    patch_radius: usize,
    beta: f64,
) -> PyResult<(PyImage, PyImage)> {
    let (target_vals, target_shape) = image_to_vec(target.inner.as_ref())?;

    if atlas_images.len() != atlas_labels.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "atlas_images length {} != atlas_labels length {}",
            atlas_images.len(),
            atlas_labels.len()
        )));
    }
    if atlas_images.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "atlas_images list is empty; at least one atlas is required",
        ));
    }

    // Extract all atlas data and validate shapes before releasing the GIL.
    let mut img_vecs: Vec<Vec<f32>> = Vec::with_capacity(atlas_images.len());
    let mut lbl_vecs: Vec<Vec<u32>> = Vec::with_capacity(atlas_labels.len());

    for (i, (img, lbl)) in atlas_images.iter().zip(atlas_labels.iter()).enumerate() {
        let (img_vals, img_shape) = image_to_vec(img.borrow(py).inner.as_ref())?;
        if img_shape != target_shape {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "atlas_images[{}] shape {:?} != target shape {:?}",
                i, img_shape, target_shape
            )));
        }
        let (lbl_vals, lbl_shape) = image_to_vec(lbl.borrow(py).inner.as_ref())?;
        if lbl_shape != target_shape {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "atlas_labels[{}] shape {:?} != target shape {:?}",
                i, lbl_shape, target_shape
            )));
        }
        img_vecs.push(img_vals);
        lbl_vecs.push(lbl_vals.iter().map(|&v| v.round() as u32).collect());
    }

    // Heavy JLF computation — release the GIL for the duration.
    // `target_vals`, `img_vecs`, and `lbl_vecs` are moved into the closure;
    // slices are constructed inside to avoid cross-boundary lifetime issues.
    let result = py
        .allow_threads(|| {
            let img_slices: Vec<&[f32]> = img_vecs.iter().map(|v| v.as_slice()).collect();
            let lbl_slices: Vec<&[u32]> = lbl_vecs.iter().map(|v| v.as_slice()).collect();
            let config = LabelFusionConfig { patch_radius, beta };
            joint_label_fusion(
                &target_vals,
                &img_slices,
                &lbl_slices,
                target_shape,
                &config,
            )
            .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // GIL is re-acquired here; target.inner is accessible again.
    let labels_f32: Vec<f32> = result.labels.iter().map(|&l| l as f32).collect();

    let labels_image = vec_to_image(
        labels_f32,
        target_shape,
        target.inner.origin().clone(),
        target.inner.spacing().clone(),
        target.inner.direction().clone(),
    );
    let confidence_image = vec_to_image(
        result.confidence,
        target_shape,
        target.inner.origin().clone(),
        target.inner.spacing().clone(),
        target.inner.direction().clone(),
    );

    Ok((into_py_image(labels_image), into_py_image(confidence_image)))
}

// ── Submodule registration ────────────────────────────────────────────────────

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "registration")?;
    m.add_function(wrap_pyfunction!(demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(diffeomorphic_demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(symmetric_demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(inverse_consistent_demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(multires_demons_register, &m)?)?;
    m.add_function(wrap_pyfunction!(syn_register, &m)?)?;
    m.add_function(wrap_pyfunction!(bspline_ffd_register, &m)?)?;
    m.add_function(wrap_pyfunction!(multires_syn_register, &m)?)?;
    m.add_function(wrap_pyfunction!(bspline_syn_register, &m)?)?;
    m.add_function(wrap_pyfunction!(lddmm_register, &m)?)?;
    m.add_function(wrap_pyfunction!(build_atlas, &m)?)?;
    m.add_function(wrap_pyfunction!(majority_vote_fusion, &m)?)?;
    m.add_function(wrap_pyfunction!(joint_label_fusion_py, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
