//! Differentiable coordinate transforms on Coeus autograd `Var`s.
//!
//! The transform maps a fixed-image sampling grid into moving-image coordinate
//! space as a function of trainable parameters; its output feeds
//! [`super::sampling`]. Because the parameter is on the autograd tape, the loss
//! gradient reaches it through the sampled intensities.
//!
//! This module currently provides translation — the simplest transform, and
//! the one that needs no new Coeus op (a broadcast add). Rotation/affine
//! (matmul-based) transforms are later increments; the eventual Coeus-native
//! `Transform` trait surface that bundles per-axis parameters into a single
//! `[D]`/matrix parameter is ADR-gated (`[arch]`) and will wrap these primitives.

use coeus_autograd::{add, broadcast_to, matmul, reshape, transpose_2d, Var};
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_ops::BackendOps;

/// Differentiable per-axis translation: `out = coords + t` (broadcast).
///
/// `coords` is a `[N]` coordinate vector for one axis; `t` is a scalar (`[1]`)
/// translation parameter. The result is `[N]`. The scalar broadcasts across all
/// `N` points, and `broadcast_to`'s summing backward means the gradient
/// accumulated into `t` is `Σ_k ∂loss/∂out_k` — the correct gradient for a
/// single translation parameter shared by every point.
///
/// # Panics
///
/// Panics if `coords` is not 1-D or `t` is not a single-element (`[1]`) tensor
/// — caller invariants.
pub fn translate_axis_coeus<T, B>(coords: &Var<T, B>, t: &Var<T, B>) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let coords_shape = coords.tensor.shape();
    assert_eq!(coords_shape.len(), 1, "translate_axis_coeus: coords must be 1-D");
    assert_eq!(
        t.tensor.shape(),
        [1],
        "translate_axis_coeus: translation must be a single [1] scalar parameter"
    );
    let n = coords_shape[0];
    let t_broadcast = broadcast_to(t, vec![n]);
    add(coords, &t_broadcast)
}

/// Differentiable affine transform of a batch of points: `out = coords·Rᵀ + t`,
/// i.e. `out[n, :] = R · coords[n, :] + t` per point.
///
/// - `coords`: `[N, 3]` batch of points (row per point).
/// - `r`: `[3, 3]` linear map (rotation/scale/shear); `out = matmul(coords, Rᵀ)`.
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
pub fn affine_transform_coeus<T, B>(coords: &Var<T, B>, r: &Var<T, B>, t: &Var<T, B>) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let coords_shape = coords.tensor.shape();
    assert_eq!(coords_shape.len(), 2, "affine_transform_coeus: coords must be [N, 3]");
    assert_eq!(coords_shape[1], 3, "affine_transform_coeus: coords must have 3 columns");
    assert_eq!(r.tensor.shape(), [3, 3], "affine_transform_coeus: R must be [3, 3]");
    assert_eq!(t.tensor.shape(), [3], "affine_transform_coeus: t must be [3]");
    let n = coords_shape[0];

    // out = coords[N,3] · Rᵀ[3,3]  ⇒  out[n,j] = Σ_k coords[n,k]·R[j,k] = (R·coords[n])[j].
    let linear = matmul(coords, &transpose_2d(r));
    // Broadcast t[3] → [1,3] → [N,3] and add.
    let t_row = reshape(t, [1usize, 3]);
    let t_broadcast = broadcast_to(&t_row, vec![n, 3]);
    add(&linear, &t_broadcast)
}

#[cfg(test)]
#[path = "tests_transform.rs"]
mod tests;
