//! Generic differentiable MSE registration metric on Coeus autograd `Var`s
//! (ADR 0001).
//!
//! [`mse_metric`] is the single composition SSOT:
//!
//! ```text
//! loss = mse( sample_trilinear( moving, split( transform(grid) ) ), fixed )
//! ```
//!
//! - a [`Transform`] maps the `[N, 3]` fixed grid into moving space as a
//!   function of its trainable parameters;
//! - [`super::sampling::sample_trilinear`] reads the moving image at the
//!   transformed coordinates (gradient flows back to them);
//! - [`super::mse::mean_squared_error`] reduces to the scalar loss.
//!
//! The whole chain stays on the autograd tape, so `.backward()` on the returned
//! loss accumulates the gradient into the transform's parameters — the signal a
//! gradient-descent optimizer consumes to drive alignment. [`affine_mse`]
//! is a thin convenience wrapper constructing an [`Affine`] and dispatching.

use coeus_autograd::{reshape, slice, Var};
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_ops::BackendOps;

use super::mse::Mse;
use super::sampling::sample_trilinear;
use super::traits::{Metric, Transform};
use super::transform::Affine;

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
pub fn affine_mse<T, B>(
    moving_flat: &Var<T, B>,
    dims: [usize; 3],
    fixed: &Var<T, B>,
    grid: &Var<T, B>,
    r: &Var<T, B>,
    t: &Var<T, B>,
) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    // Thin wrapper: the split → sample → reduce composition lives once in
    // `evaluate`; this bundles the affine parameters and picks the MSE metric.
    let transform = Affine {
        r: r.clone(),
        t: t.clone(),
    };
    mse_metric(moving_flat, dims, fixed, grid, &transform)
}

/// Convenience MSE metric over any [`Transform`]: `evaluate` with [`Mse`].
///
/// See [`evaluate`] for the composition and parameter semantics.
pub fn mse_metric<T, B, Tf>(
    moving_flat: &Var<T, B>,
    dims: [usize; 3],
    fixed: &Var<T, B>,
    grid: &Var<T, B>,
    transform: &Tf,
) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    Tf: Transform<T, B>,
{
    evaluate(moving_flat, dims, fixed, grid, &Mse, transform)
}

/// Generic differentiable registration metric over any [`Metric`] and
/// [`Transform`] (ADR 0001). This is the single composition SSOT:
/// `metric.reduce(sample_trilinear(moving, split(transform(grid))), fixed)`.
///
/// - `moving_flat`: flattened moving image `[Z·Y·X]` with `dims = [Z, Y, X]`.
/// - `fixed`: fixed-image intensities at the `N` grid points, `[N]`.
/// - `grid`: fixed-grid voxel coordinates as `[N, 3]` rows `(z, y, x)`.
/// - `metric`: the intensity-loss reduction ([`Mse`], [`super::ncc::Ncc`], …).
/// - `transform`: the differentiable coordinate transform; its parameters
///   receive the alignment gradient via `.backward()` on the returned loss.
///
/// The transformed `[N, 3]` coordinates are split into the three per-axis `[N]`
/// `Var`s the trilinear sampler consumes via the differentiable `slice` +
/// `reshape` (tape-transparent, MIG-477).
///
/// # Panics
///
/// Panics if `grid` is not `[N, 3]` or on the composed primitives' shape
/// invariants — caller invariants.
pub fn evaluate<T, B, M, Tf>(
    moving_flat: &Var<T, B>,
    dims: [usize; 3],
    fixed: &Var<T, B>,
    grid: &Var<T, B>,
    metric: &M,
    transform: &Tf,
) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    M: Metric<T, B>,
    Tf: Transform<T, B>,
{
    let grid_shape = grid.tensor.shape();
    assert_eq!(grid_shape.len(), 2, "evaluate: grid must be [N, 3]");
    assert_eq!(grid_shape[1], 3, "evaluate: grid must have 3 columns");
    let n = grid_shape[0];

    let transformed = transform.transform_points(grid); // [N, 3], columns (z, y, x)
    let cz = reshape(&slice(&transformed, &[(0, n), (0, 1)]), [n]);
    let cy = reshape(&slice(&transformed, &[(0, n), (1, 2)]), [n]);
    let cx = reshape(&slice(&transformed, &[(0, n), (2, 3)]), [n]);
    let sampled = sample_trilinear(moving_flat, dims, &cz, &cy, &cx);
    metric.reduce(&sampled, fixed)
}

#[cfg(test)]
#[path = "tests_metric.rs"]
mod tests;
