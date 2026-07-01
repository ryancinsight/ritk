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

use coeus_autograd::{add, broadcast_to, Var};
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

#[cfg(test)]
#[path = "tests_transform.rs"]
mod tests;
