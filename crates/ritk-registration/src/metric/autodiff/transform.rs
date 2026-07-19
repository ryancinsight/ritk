//! Differentiable coordinate transforms on Coeus autograd `Var`s.
//!
//! The transform maps a fixed-image sampling grid into moving-image coordinate
//! space as a function of trainable parameters; its output feeds
//! [`super::sampling`]. Because the parameter is on the autograd tape, the loss
//! gradient reaches it through the sampled intensities.
//!
//! This module provides the point-transform function [`affine_transform`]
//! and the [`Transform`]-implementing parameter bundles ([`Translation`],
//! [`Affine`]) that the generic metric ([`super::metric::mse_metric`])
//! dispatches over (ADR 0001).

use coeus_autograd::{add, broadcast_to, matmul, reshape, transpose_2d, Var};
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_ops::BackendOps;

use super::traits::Transform;

/// Differentiable affine transform of a batch of points: `out = coords·Ráµ€ + t`,
/// i.e. `out[n, :] = R · coords[n, :] + t` per point.
///
/// - `coords`: `[N, 3]` batch of points (row per point).
/// - `r`: `[3, 3]` linear map (rotation/scale/shear); `out = matmul(coords, Ráµ€)`.
/// - `t`: `[3]` translation, broadcast across all `N` points.
///
/// Returns `[N, 3]`. The autograd graph links back to `r` (through
/// `matmul`/`transpose_2d`) and `t` (through `broadcast_to`, whose summing
/// backward gives `∂loss/∂t_j = Σ_n ∂loss/∂out[n,j]`) — and to `coords` if it
/// requires grad. This is the general affine that a rigid/similarity/affine
/// registration optimizer parameterizes; `R` is the natural `[3, 3]` parameter
/// tensor (contrast the per-axis scalar form, which would need 9 separate
/// scalars). Uses Coeus `matmul` — the Atlas replacement for the Burn/nalgebra
/// matrix path.
///
/// # Panics
///
/// Panics if `coords` is not `[N, 3]`, `r` is not `[3, 3]`, or `t` is not `[3]`
/// — caller invariants.
pub fn affine_transform<T, B>(coords: &Var<T, B>, r: &Var<T, B>, t: &Var<T, B>) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let coords_shape = coords.tensor.shape();
    assert_eq!(
        coords_shape.len(),
        2,
        "affine_transform: coords must be [N, 3]"
    );
    assert_eq!(
        coords_shape[1], 3,
        "affine_transform: coords must have 3 columns"
    );
    assert_eq!(
        r.tensor.shape(),
        [3, 3],
        "affine_transform: R must be [3, 3]"
    );
    assert_eq!(t.tensor.shape(), [3], "affine_transform: t must be [3]");
    let n = coords_shape[0];

    // out = coords[N,3] · Ráµ€[3,3]  ⇒  out[n,j] = Σ_k coords[n,k]·R[j,k] = (R·coords[n])[j].
    let linear = matmul(coords, &transpose_2d(r));
    // Broadcast t[3] → [1,3] → [N,3] and add.
    let t_row = reshape(t, [1usize, 3]);
    let t_broadcast = broadcast_to(&t_row, vec![n, 3]);
    add(&linear, &t_broadcast)
}

/// Translation transform parameter bundle: a single `[3]` offset applied to
/// every point. Implements [`Transform`] (`out = points + t`).
#[derive(Clone)]
pub struct Translation<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
{
    /// Translation vector, shape `[3]`; mark `requires_grad` to optimize it.
    pub t: Var<T, B>,
}

impl<T, B> Transform<T, B> for Translation<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    #[inline]
    fn transform_points(&self, points: &Var<T, B>) -> Var<T, B> {
        let shape = points.tensor.shape();
        assert_eq!(shape.len(), 2, "Translation: points must be [N, 3]");
        assert_eq!(shape[1], 3, "Translation: points must have 3 columns");
        assert_eq!(self.t.tensor.shape(), [3], "Translation: t must be [3]");
        let n = shape[0];
        let t_row = reshape(&self.t, [1usize, 3]);
        add(points, &broadcast_to(&t_row, vec![n, 3]))
    }
}

/// Affine transform parameter bundle: a `[3, 3]` linear map plus a `[3]`
/// translation. Implements [`Transform`] (`out = points·Ráµ€ + t`).
#[derive(Clone)]
pub struct Affine<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
{
    /// Linear map, shape `[3, 3]`.
    pub r: Var<T, B>,
    /// Translation vector, shape `[3]`.
    pub t: Var<T, B>,
}

impl<T, B> Transform<T, B> for Affine<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    #[inline]
    fn transform_points(&self, points: &Var<T, B>) -> Var<T, B> {
        affine_transform(points, &self.r, &self.t)
    }
}

#[cfg(test)]
#[path = "tests_transform.rs"]
mod tests;
