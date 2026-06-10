//! 1-D linear interpolation with const-generic autodiff dispatch (Sprint 355).
//!
//! This module provides the 1-D linear interpolation kernel with an inline
//! `B::ad_enabled()` const-generic branch to select between the autodiff-safe
//! path (preserves the Burn graph) and the non-autodiff fast path (saves
//! clones per call).
//!
//! # Dispatch mechanism
//!
//! The dispatch is expressed in a **single function** (`interpolate_1d`)
//! using `if B::ad_enabled() { ... } else { ... }`, mirroring the
//! `dim3.rs` design from Sprint 355. The branch is monomorphized per
//! backend and dead-code-eliminated at compile time, so there is no
//! runtime cost. This is the "#[cfg(...)]-free specialization" pattern
//! requested in audit §4.2.
//!
//! # Gather helpers
//!
//! - [`gather_1d`] — autodiff-safe, borrows the index tensor (preserves graph)
//! - [`gather_1d_owned`] — non-autodiff, takes the owned index tensor (saves a clone)
//!
//! # Clone counts
//!
//! | Path | Gather helper | Inside-gather clones | Graph preserved? |
//! |------|---------------|---------------------:|------------------|
//! | Autodiff (`B::ad_enabled() == true`) | `gather_1d` (borrowed) | 2 (one per call) | yes |
//! | Non-autodiff (`B::ad_enabled() == false`) | `gather_1d_owned` (owned) | 1 (one per call) | n/a |

use burn::tensor::{backend::Backend, Tensor};

use crate::interpolation::shared::{in_bounds_mask, OutOfBoundsMode};

/// 1-D gather with a borrowed index — used by the autodiff path where
/// the caller must retain the index tensor for further use.
#[inline]
fn gather_1d<B: Backend>(
    flat_data: &Tensor<B, 1>,
    idx: &Tensor<B, 1, burn::tensor::Int>,
) -> Tensor<B, 1> {
    flat_data.clone().gather(0, idx.clone())
}

/// 1-D gather consuming the index tensor — used by the non-autodiff
/// fast path. Eliminates the `idx.clone()` per call (the caller
/// transfers ownership).
#[inline]
fn gather_1d_owned<B: Backend>(
    flat_data: &Tensor<B, 1>,
    idx: Tensor<B, 1, burn::tensor::Int>,
) -> Tensor<B, 1> {
    flat_data.clone().gather(0, idx)
}

/// Linear interpolation with const-generic autodiff dispatch (Sprint 355).
///
/// Single-function implementation: the `if B::ad_enabled()` branch is
/// monomorphized per backend and dead-code-eliminated at compile time,
/// so there is no runtime cost. This unifies the previous flat design
/// into one body, matching the `dim3.rs` shape and unblocking a future
/// macro-template unification of all 4 D-arms (D=1, 2, 3, 4).
pub(crate) fn interpolate_1d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // X
    let batch_size = indices.dims()[0];
    let device = indices.device();

    // indices: [N, 1] -> x (consumes indices, so we can't avoid the move)
    let x = indices.narrow(1, 0, 1).squeeze_dims(&[1]);

    // Floor coordinate and weight (identical in both paths).
    let x0 = x.clone().floor();
    let wx = x - x0.clone();

    // Ceil coord.
    let x1 = x0.clone() + 1.0;

    // Clamped int indices. x0 is consumed by clamp+int; the leftover
    // floor value is needed for `in_bounds_mask` at the end.
    let x0_i = x0.clone().clamp(0.0, (d0 - 1) as f64).int();
    let x1_i = x1.clamp(0.0, (d0 - 1) as f64).int();

    // Pre-flatten data — reshape consumes self, but data is &Tensor so
    // clone once.
    let flat_data = data.clone().reshape([d0]);

    // ── Const-generic dispatch (Sprint 355) ────────────────────────────
    // The `if B::ad_enabled()` branch is monomorphized per backend and
    // dead-code-eliminated at compile time. Only one arm is live in the
    // monomorphized function — the other is folded away by the compiler.
    //
    // - Autodiff arm: borrows `&x0_i` / `&x1_i` (preserves Burn graph)
    // - Non-autodiff arm: transfers ownership of `x0_i` / `x1_i` (saves clones)
    let (v0, v1) = if B::ad_enabled() {
        (
            gather_1d(&flat_data, &x0_i),
            gather_1d(&flat_data, &x1_i),
        )
    } else {
        (
            gather_1d_owned(&flat_data, x0_i),
            gather_1d_owned(&flat_data, x1_i),
        )
    };

    // Linear interpolation (identical in both paths).
    let one = Tensor::<B, 1>::ones([batch_size], &device);
    let one_minus_wx = one - wx.clone();

    let result = v0 * one_minus_wx + v1 * wx;

    if let Some(mask) = in_bounds_mask(x0, (d0 - 1) as f64, mode) {
        result * mask
    } else {
        result
    }
}
