//! End-to-end differentiable MSE-over-a-translation registration metric on
//! Coeus autograd `Var`s.
//!
//! Composes the three verified primitives of the autodiff migration path into
//! the first genuinely usable Coeus-native registration metric:
//!
//! ```text
//! loss = mse( sample_trilinear( moving, translate(grid, t) ), fixed )
//! ```
//!
//! - [`super::transform::translate_axis_coeus`] maps the fixed grid into moving
//!   space as a function of the translation parameters `t`;
//! - [`super::sampling::sample_trilinear_coeus`] reads the moving image at the
//!   transformed coordinates (gradient flows back to the coordinates);
//! - [`super::mse::mean_squared_error_coeus`] reduces to the scalar loss.
//!
//! The whole chain stays on the autograd tape, so `.backward()` on the returned
//! loss accumulates the gradient into the three translation parameters — the
//! signal a gradient-descent optimizer consumes to drive alignment. Parameters
//! are supplied per axis (three `[1]` `Var`s) for the same reason coordinates
//! are (avoids an unverified differentiable slice on a bundled `[3]`); the
//! ADR-gated Coeus-native `Transform` surface will present the bundled form.

use coeus_autograd::{reshape, slice, Var};
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_ops::BackendOps;

use super::mse::mean_squared_error_coeus;
use super::sampling::sample_trilinear_coeus;
use super::transform::{affine_transform_coeus, translate_axis_coeus};

/// Differentiable MSE between `fixed` and the `moving` image sampled at a
/// translated copy of the fixed sampling grid.
///
/// - `moving_flat`: flattened moving image, shape `[Z·Y·X]` with
///   `dims = [Z, Y, X]` (row-major).
/// - `fixed`: fixed-image intensities at the `N` grid points, shape `[N]`.
/// - `grid_z`/`grid_y`/`grid_x`: fixed-grid voxel coordinates, each `[N]`.
/// - `tz`/`ty`/`tx`: per-axis translation parameters, each `[1]`; mark these
///   `requires_grad` to receive the alignment gradient.
///
/// Returns the scalar (`[1]`) loss `Var`. Calling `.backward()` on it
/// accumulates `∂loss/∂t` into `tz`/`ty`/`tx`.
///
/// # Panics
///
/// Panics on the shape-invariant violations of the composed primitives
/// (non-flat `moving_flat`, mismatched coordinate lengths, non-`[1]`
/// parameters) — caller invariants.
#[allow(clippy::too_many_arguments)]
pub fn translation_mse_coeus<T, B>(
    moving_flat: &Var<T, B>,
    dims: [usize; 3],
    fixed: &Var<T, B>,
    grid_z: &Var<T, B>,
    grid_y: &Var<T, B>,
    grid_x: &Var<T, B>,
    tz: &Var<T, B>,
    ty: &Var<T, B>,
    tx: &Var<T, B>,
) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let cz = translate_axis_coeus(grid_z, tz);
    let cy = translate_axis_coeus(grid_y, ty);
    let cx = translate_axis_coeus(grid_x, tx);
    let sampled = sample_trilinear_coeus(moving_flat, dims, &cz, &cy, &cx);
    mean_squared_error_coeus(&sampled, fixed)
}

/// Differentiable MSE between `fixed` and the `moving` image sampled at an
/// affine-transformed copy of the fixed sampling grid.
///
/// - `moving_flat`: flattened moving image, shape `[Z·Y·X]` with
///   `dims = [Z, Y, X]` (row-major).
/// - `fixed`: fixed-image intensities at the `N` grid points, shape `[N]`.
/// - `grid`: fixed-grid voxel coordinates as `[N, 3]` rows ordered `(z, y, x)`.
/// - `r`: `[3, 3]` linear map, `t`: `[3]` translation — the affine parameters;
///   mark `requires_grad` to receive the alignment gradient.
///
/// Returns the scalar (`[1]`) loss `Var`. Internally the affine's `[N, 3]`
/// output is split into the three per-axis coordinate `Var`s the trilinear
/// sampler consumes, via the differentiable `slice` (its scatter backward keeps
/// the tape intact through the split), so `.backward()` accumulates `∂loss/∂R`
/// and `∂loss/∂t`.
///
/// # Panics
///
/// Panics on the shape-invariant violations of the composed primitives
/// (non-flat `moving_flat`, `grid` not `[N, 3]`, `r` not `[3, 3]`, `t` not
/// `[3]`) — caller invariants.
pub fn affine_mse_coeus<T, B>(
    moving_flat: &Var<T, B>,
    dims: [usize; 3],
    fixed: &Var<T, B>,
    grid: &Var<T, B>,
    r: &Var<T, B>,
    t: &Var<T, B>,
) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let n = grid.tensor.shape().first().copied().unwrap_or(0);
    let transformed = affine_transform_coeus(grid, r, t); // [N, 3], columns (z, y, x)
    let cz = reshape(&slice(&transformed, &[(0, n), (0, 1)]), [n]);
    let cy = reshape(&slice(&transformed, &[(0, n), (1, 2)]), [n]);
    let cx = reshape(&slice(&transformed, &[(0, n), (2, 3)]), [n]);
    let sampled = sample_trilinear_coeus(moving_flat, dims, &cz, &cy, &cx);
    mean_squared_error_coeus(&sampled, fixed)
}

#[cfg(test)]
#[path = "tests_metric.rs"]
mod tests;
