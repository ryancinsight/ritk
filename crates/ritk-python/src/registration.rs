//! Python-exposed deformable image registration using Thirion's Demons algorithm.
//!
//! # Mathematical Specification — Thirion Demons (1998)
//!
//! Given fixed image F and moving image M, both defined on ℤ³, the Demons
//! algorithm computes a displacement field D : ℤ³ → ℝ³ that warps M toward F.
//!
//! **Force at each voxel p:**
//!
//!   f(p) = (F(p) − M_w(p)) · ∇F(p) / (|∇F(p)|² + (F(p) − M_w(p))² + ε)
//!
//! where:
//! - M_w(p) = M(p + D(p)) is the current warp of M (trilinear interpolation)
//! - ∇F(p) = (∂F/∂z, ∂F/∂y, ∂F/∂x) estimated by central differences
//! - ε = 1e-5 prevents division by zero
//!
//! **Update rule:**
//!   D ← D + f
//!
//! **Regularisation:**
//!   D ← G_σ ∗ D  (separable 3-D Gaussian smoothing with σ = sigma_diffusion)
//!
//! This is iterated for `max_iterations` steps.
//!
//! # Reference
//! Thirion, J.-P. (1998). Image matching as a diffusion process: an analogy with
//! Maxwell's demons. Medical Image Analysis, 2(3), 243–260.
//!
//! # Implementation
//! Self-contained CPU implementation operating on `Vec<f32>` extracted from the
//! image tensors.  Does not depend on any ritk-registration module.

use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use pyo3::prelude::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_registration::demons::{
    DemonsConfig, DiffeomorphicDemonsRegistration, SymmetricDemonsRegistration,
};
use ritk_registration::diffeomorphic::{SyNConfig, SyNRegistration};

// ── Public pyfunction ─────────────────────────────────────────────────────────

/// Register a moving image to a fixed image using Thirion's Demons algorithm.
///
/// Performs iterative deformable registration.  Each iteration:
/// 1. Warps the moving image with the current displacement field.
/// 2. Computes Demons forces from image difference and fixed-image gradient.
/// 3. Adds forces to the displacement field.
/// 4. Smooths the displacement field with a Gaussian of sigma `sigma_diffusion`.
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
///     RuntimeError: if image shapes do not match or tensor extraction fails.
#[pyfunction]
#[pyo3(signature = (fixed, moving, max_iterations=50, sigma_diffusion=1.0))]
pub fn demons_register(
    fixed: &PyImage,
    moving: &PyImage,
    max_iterations: usize,
    sigma_diffusion: f64,
) -> PyResult<(PyImage, PyImage)> {
    // ── Extract and validate inputs ───────────────────────────────────────────
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref())?;
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref())?;

    if fixed_shape != moving_shape {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "fixed shape {:?} != moving shape {:?}; images must have identical shapes",
            fixed_shape, moving_shape
        )));
    }

    let [nz, ny, nx] = fixed_shape;
    let n = nz * ny * nx;

    // ── Initialise displacement field to zero ─────────────────────────────────
    // Three separate component vectors: dz, dy, dx.
    let mut disp_z = vec![0.0_f32; n];
    let mut disp_y = vec![0.0_f32; n];
    let mut disp_x = vec![0.0_f32; n];

    // ── Precompute fixed-image gradient via central differences ───────────────
    // ∂F/∂z, ∂F/∂y, ∂F/∂x — constant across iterations.
    let (grad_z, grad_y, grad_x) = compute_gradient(&fixed_vals, fixed_shape);

    // ── Iterations ────────────────────────────────────────────────────────────
    for _iter in 0..max_iterations {
        // 1. Warp moving image with current displacement.
        let moving_warped = warp_image(&moving_vals, fixed_shape, &disp_z, &disp_y, &disp_x);

        // 2. Compute Demons forces and accumulate into displacement field.
        for i in 0..n {
            let diff = fixed_vals[i] - moving_warped[i];

            let gz = grad_z[i];
            let gy = grad_y[i];
            let gx = grad_x[i];

            // Thirion denominator: |∇F|² + (F − M_w)² + ε.
            let grad_sq = gz * gz + gy * gy + gx * gx;
            let diff_sq = diff * diff;
            let denom = grad_sq + diff_sq + 1e-5_f32;

            let scale = diff / denom;
            disp_z[i] += scale * gz;
            disp_y[i] += scale * gy;
            disp_x[i] += scale * gx;
        }

        // 3. Smooth displacement field with separable Gaussian.
        if sigma_diffusion > 0.0 {
            gaussian_smooth_inplace(&mut disp_z, fixed_shape, sigma_diffusion);
            gaussian_smooth_inplace(&mut disp_y, fixed_shape, sigma_diffusion);
            gaussian_smooth_inplace(&mut disp_x, fixed_shape, sigma_diffusion);
        }
    }

    // ── Final warp of moving image ────────────────────────────────────────────
    let warped_vals = warp_image(&moving_vals, fixed_shape, &disp_z, &disp_y, &disp_x);

    // ── Pack displacement field as [3·Z, Y, X] image ──────────────────────────
    // Stacked: [0:nz, :, :] = dz component
    //         [nz:2*nz, :, :] = dy component
    //         [2*nz:3*nz, :, :] = dx component
    // The caller can recover (3, Z, Y, X) with .to_numpy().reshape(3, Z, Y, X).
    let mut disp_packed = Vec::with_capacity(3 * n);
    disp_packed.extend_from_slice(&disp_z);
    disp_packed.extend_from_slice(&disp_y);
    disp_packed.extend_from_slice(&disp_x);

    let disp_shape = [3 * nz, ny, nx];

    // Spatial metadata: warped image inherits fixed image metadata.
    let warped_image = vec_to_image(
        warped_vals,
        fixed_shape,
        fixed.inner.origin().clone(),
        fixed.inner.spacing().clone(),
        fixed.inner.direction().clone(),
    );

    // Displacement field: unit spacing (components are in voxel units).
    let disp_image = vec_to_image(
        disp_packed,
        disp_shape,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    Ok((into_py_image(warped_image), into_py_image(disp_image)))
}

// ── Gradient computation ──────────────────────────────────────────────────────

/// Compute the gradient of a 3-D image using central differences.
///
/// At interior voxels: ∂F/∂z ≈ (F[iz+1, iy, ix] − F[iz−1, iy, ix]) / 2.
/// At boundary voxels: one-sided (forward or backward) first-order differences.
///
/// Returns three flat Vec<f32>: (grad_z, grad_y, grad_x).
fn compute_gradient(data: &[f32], dims: [usize; 3]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let mut gz = vec![0.0_f32; n];
    let mut gy = vec![0.0_f32; n];
    let mut gx = vec![0.0_f32; n];

    let idx = |z: usize, y: usize, x: usize| -> usize { z * ny * nx + y * nx + x };

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);

                // ∂F/∂z
                gz[flat] = if nz == 1 {
                    0.0
                } else if iz == 0 {
                    data[idx(1, iy, ix)] - data[flat]
                } else if iz == nz - 1 {
                    data[flat] - data[idx(nz - 2, iy, ix)]
                } else {
                    (data[idx(iz + 1, iy, ix)] - data[idx(iz - 1, iy, ix)]) * 0.5
                };

                // ∂F/∂y
                gy[flat] = if ny == 1 {
                    0.0
                } else if iy == 0 {
                    data[idx(iz, 1, ix)] - data[flat]
                } else if iy == ny - 1 {
                    data[flat] - data[idx(iz, ny - 2, ix)]
                } else {
                    (data[idx(iz, iy + 1, ix)] - data[idx(iz, iy - 1, ix)]) * 0.5
                };

                // ∂F/∂x
                gx[flat] = if nx == 1 {
                    0.0
                } else if ix == 0 {
                    data[idx(iz, iy, 1)] - data[flat]
                } else if ix == nx - 1 {
                    data[flat] - data[idx(iz, iy, nx - 2)]
                } else {
                    (data[idx(iz, iy, ix + 1)] - data[idx(iz, iy, ix - 1)]) * 0.5
                };
            }
        }
    }

    (gz, gy, gx)
}

// ── Trilinear interpolation ───────────────────────────────────────────────────

/// Sample a 3-D volume at a continuous (z, y, x) position using trilinear
/// interpolation with clamp-to-border boundary condition.
///
/// # Invariants
/// - z, y, x coordinates outside [0, dim-1] are clamped to the border value.
/// - The eight-neighbour trilinear formula is exact for integer positions.
#[inline(always)]
fn trilinear_interpolate(data: &[f32], dims: [usize; 3], z: f32, y: f32, x: f32) -> f32 {
    let [nz, ny, nx] = dims;

    // Clamp to valid range.
    let z = z.max(0.0).min((nz as f32) - 1.0);
    let y = y.max(0.0).min((ny as f32) - 1.0);
    let x = x.max(0.0).min((nx as f32) - 1.0);

    let iz0 = z.floor() as usize;
    let iy0 = y.floor() as usize;
    let ix0 = x.floor() as usize;

    let iz1 = (iz0 + 1).min(nz - 1);
    let iy1 = (iy0 + 1).min(ny - 1);
    let ix1 = (ix0 + 1).min(nx - 1);

    let dz = z - iz0 as f32;
    let dy = y - iy0 as f32;
    let dx = x - ix0 as f32;

    let g = |z: usize, y: usize, x: usize| data[z * ny * nx + y * nx + x];

    // Trilinear blend in x, then y, then z.
    let c00 = g(iz0, iy0, ix0) * (1.0 - dx) + g(iz0, iy0, ix1) * dx;
    let c01 = g(iz0, iy1, ix0) * (1.0 - dx) + g(iz0, iy1, ix1) * dx;
    let c10 = g(iz1, iy0, ix0) * (1.0 - dx) + g(iz1, iy0, ix1) * dx;
    let c11 = g(iz1, iy1, ix0) * (1.0 - dx) + g(iz1, iy1, ix1) * dx;

    let c0 = c00 * (1.0 - dy) + c01 * dy;
    let c1 = c10 * (1.0 - dy) + c11 * dy;

    c0 * (1.0 - dz) + c1 * dz
}

// ── Image warping ─────────────────────────────────────────────────────────────

/// Warp a 3-D volume using a displacement field.
///
/// For each voxel p = (iz, iy, ix):
///   warped(p) = moving(p + D(p)) = moving(iz + dz[p], iy + dy[p], ix + dx[p])
/// sampled with trilinear interpolation.
fn warp_image(
    moving: &[f32],
    dims: [usize; 3],
    disp_z: &[f32],
    disp_y: &[f32],
    disp_x: &[f32],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut warped = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = iz * ny * nx + iy * nx + ix;
                let wz = iz as f32 + disp_z[flat];
                let wy = iy as f32 + disp_y[flat];
                let wx = ix as f32 + disp_x[flat];
                warped[flat] = trilinear_interpolate(moving, dims, wz, wy, wx);
            }
        }
    }

    warped
}

// ── Separable Gaussian smoothing ──────────────────────────────────────────────

/// Build a normalised 1-D Gaussian kernel of radius = ⌈3σ⌉.
///
/// Kernel values: k[i] = exp(−((i − r)² / (2σ²))), then normalised to sum = 1.
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let width = 2 * radius + 1;
    let inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);

    let mut kernel: Vec<f64> = (0..width)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x * inv_two_sigma2).exp()
        })
        .collect();

    let sum: f64 = kernel.iter().sum();
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Convolve `data` along `axis` with `kernel` using replicate (clamp) padding.
///
/// Writes the result into `output` (which must have the same length as `data`).
fn convolve_axis(data: &[f32], dims: [usize; 3], kernel: &[f64], axis: usize, output: &mut [f32]) {
    let [nz, ny, nx] = dims;
    let r = kernel.len() / 2;
    let axis_len = [nz, ny, nx][axis];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let pos = [iz, iy, ix][axis];
                let flat = iz * ny * nx + iy * nx + ix;
                let mut acc = 0.0_f64;

                for (ki, &kv) in kernel.iter().enumerate() {
                    let src_pos = (pos as isize + ki as isize - r as isize)
                        .max(0)
                        .min(axis_len as isize - 1) as usize;

                    let src_flat = match axis {
                        0 => src_pos * ny * nx + iy * nx + ix,
                        1 => iz * ny * nx + src_pos * nx + ix,
                        _ => iz * ny * nx + iy * nx + src_pos,
                    };
                    acc += kv * data[src_flat] as f64;
                }
                output[flat] = acc as f32;
            }
        }
    }
}

/// Apply separable 3-D Gaussian smoothing in-place.
///
/// Convolves along Z, then Y, then X axes sequentially.
/// Uses a temporary buffer to avoid read-after-write aliasing.
fn gaussian_smooth_inplace(data: &mut Vec<f32>, dims: [usize; 3], sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let kernel = gaussian_kernel(sigma);
    let n = data.len();
    let mut tmp = vec![0.0_f32; n];

    // Z axis
    convolve_axis(data, dims, &kernel, 0, &mut tmp);
    std::mem::swap(data, &mut tmp);

    // Y axis
    convolve_axis(data, dims, &kernel, 1, &mut tmp);
    std::mem::swap(data, &mut tmp);

    // X axis
    convolve_axis(data, dims, &kernel, 2, &mut tmp);
    std::mem::swap(data, &mut tmp);
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
