//! 2-D linear interpolation with const-generic autodiff dispatch (Sprint 355).
//!
//! This module provides the 2-D bilinear interpolation kernel with an inline
//! `B::ad_enabled()` const-generic branch to select between the autodiff-safe
//! path (preserves the Burn graph) and the non-autodiff fast path (saves
//! clones per call).
//!
//! # Dispatch mechanism
//!
//! The dispatch is expressed in a **single function** (`interpolate_2d`)
//! using `if B::ad_enabled() { ... } else { ... }`, mirroring the
//! `dim3.rs` design from Sprint 355. The branch is monomorphized per
//! backend and dead-code-eliminated at compile time, so there is no
//! runtime cost. This is the "#[cfg(...)]-free specialization" pattern
//! requested in audit §4.2.
//!
//! # Gather helpers
//!
//! - [`gather_2d`] — autodiff-safe, borrows coordinate tensors (preserves graph)
//! - [`gather_2d_owned`] — non-autodiff, takes owned coordinate tensors (saves clones)
//!
//! # Clone counts
//!
//! | Path | Gather helper | Inside-gather clones | Graph preserved? |
//! |------|---------------|---------------------:|------------------|
//! | Autodiff (`B::ad_enabled() == true`) | `gather_2d` (borrowed) | 8 (two per call × 4) | yes |
//! | Non-autodiff (`B::ad_enabled() == false`) | `gather_2d_owned` (owned) | 4 (one per call × 4) | n/a |

use burn::tensor::{backend::Backend, Int, Tensor};

use crate::interpolation::shared::{in_bounds_mask, OutOfBoundsMode};

/// 2-D gather with borrowed coordinates — used by the autodiff path
/// where `xi`/`yi` must remain usable after the call. For non-autodiff
/// backends, prefer [`gather_2d_owned`] which avoids the two
/// coordinate-tensor clones per call.
#[inline]
fn gather_2d<B: Backend>(
    flat_data: &Tensor<B, 1>,
    xi: &Tensor<B, 1, Int>,
    yi: &Tensor<B, 1, Int>,
    stride_y: i32,
) -> Tensor<B, 1> {
    let idx = yi.clone() * stride_y + xi.clone();
    flat_data.clone().gather(0, idx)
}

/// 2-D gather consuming the coordinate tensors — used by the non-autodiff
/// fast path. Eliminates the two `xi`/`yi` clones per call (the caller
/// transfers ownership). Still clones `flat_data` once per call
/// (unavoidable: Burn's `Tensor::gather` consumes `self`).
#[inline]
fn gather_2d_owned<B: Backend>(
    flat_data: &Tensor<B, 1>,
    xi: Tensor<B, 1, Int>,
    yi: Tensor<B, 1, Int>,
    stride_y: i32,
) -> Tensor<B, 1> {
    let idx = yi * stride_y + xi;
    flat_data.clone().gather(0, idx)
}

/// Bilinear interpolation with const-generic autodiff dispatch (Sprint 355).
///
/// Single-function implementation: the `if B::ad_enabled()` branch is
/// monomorphized per backend and dead-code-eliminated at compile time,
/// so there is no runtime cost. This unifies the previous flat design
/// into one body, matching the `dim3.rs` shape and unblocking a future
/// macro-template unification of all 4 D-arms (D=1, 2, 3, 4).
pub(crate) fn interpolate_2d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // Y
    let d1 = shape.dims[1]; // X
    let batch_size = indices.dims()[0];
    let device = indices.device();

    // indices: [N, 2] -> (x, y). narrow consumes self, so we can't
    // avoid the 1 clone of indices_local.
    let indices_local = indices;
    let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
    let y = indices_local.narrow(1, 1, 1).squeeze_dims(&[1]);

    // Floor coordinates and weights (identical in both paths).
    let x0 = x.clone().floor();
    let wx = x - x0.clone();
    let y0 = y.clone().floor();
    let wy = y - y0.clone();

    // Ceil coords.
    let x1 = x0.clone() + 1.0;
    let y1 = y0.clone() + 1.0;

    // Clamped int indices. x0/y0 and x1/y1 are consumed by clamp+int;
    // x0/y0 are still owned after weight derivation and used by
    // `in_bounds_mask` at the end.
    let x0_i = x0.clone().clamp(0.0, (d1 - 1) as f64).int();
    let y0_i = y0.clone().clamp(0.0, (d0 - 1) as f64).int();
    let x1_i = x1.clamp(0.0, (d1 - 1) as f64).int();
    let y1_i = y1.clamp(0.0, (d0 - 1) as f64).int();

    // Stride for [Y, X] layout (d0, d1).
    let stride_y = d1 as i32;

    // Pre-flatten data — reshape consumes self, but data is &Tensor so
    // clone once.
    let flat_data = data.clone().reshape([d0 * d1]);

    // ── Const-generic dispatch (Sprint 355) ────────────────────────────
    // The `if B::ad_enabled()` branch is monomorphized per backend and
    // dead-code-eliminated at compile time. Only one arm is live in the
    // monomorphized function — the other is folded away by the compiler.
    //
    // - Autodiff arm: borrows `&x0_i` / `&y0_i` etc. (preserves Burn graph)
    // - Non-autodiff arm: transfers ownership of `x0_i` / `y0_i` etc. (saves clones)
    let (v00, v01, v10, v11) = if B::ad_enabled() {
        // Autodiff path: borrow coords (4 borrowed `gather_2d` calls).
        (
            gather_2d(&flat_data, &x0_i, &y0_i, stride_y),
            gather_2d(&flat_data, &x0_i, &y1_i, stride_y),
            gather_2d(&flat_data, &x1_i, &y0_i, stride_y),
            gather_2d(&flat_data, &x1_i, &y1_i, stride_y),
        )
    } else {
        // Non-autodiff path: own coords (4 owned `gather_2d_owned` calls).
        // Each index tensor is cloned 3 times (for its first 3 uses) and
        // consumed on its 4th use. Total caller-side clones: 4 × 3 = 12.
        (
            gather_2d_owned(&flat_data, x0_i.clone(), y0_i.clone(), stride_y),
            gather_2d_owned(&flat_data, x0_i, y1_i.clone(), stride_y),
            gather_2d_owned(&flat_data, x1_i.clone(), y0_i, stride_y),
            gather_2d_owned(&flat_data, x1_i, y1_i, stride_y),
        )
    };

    // Bilinear lerp cascade (identical in both paths).
    let one = Tensor::<B, 1>::ones([batch_size], &device);
    let one_minus_wx = one.clone() - wx.clone();
    let one_minus_wy = one - wy.clone();

    let c0 = v00 * one_minus_wx.clone() + v10 * wx.clone();
    let c1 = v01 * one_minus_wx + v11 * wx;

    let result = c0 * one_minus_wy + c1 * wy;

    // In-bounds mask (identical in both paths).
    let x_mask = in_bounds_mask(x0, (d1 - 1) as f64, mode);
    let y_mask = in_bounds_mask(y0, (d0 - 1) as f64, mode);

    match (x_mask, y_mask) {
        (Some(xm), Some(ym)) => result * xm * ym,
        _ => result,
    }
}
