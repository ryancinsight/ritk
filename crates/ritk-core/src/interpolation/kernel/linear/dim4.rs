//! 4-D linear interpolation with const-generic autodiff dispatch (Sprint 355).
//!
//! This module provides the 4-D quadrilinear interpolation kernel with an inline
//! `B::ad_enabled()` const-generic branch to select between the autodiff-safe
//! path (preserves the Burn graph) and the non-autodiff fast path (saves
//! clones per call).
//!
//! # Dispatch mechanism
//!
//! The dispatch is expressed in a **single function** (`interpolate_4d`)
//! using `if B::ad_enabled() { ... } else { ... }`, mirroring the
//! `dim3.rs` design from Sprint 355. The branch is monomorphized per
//! backend and dead-code-eliminated at compile time, so there is no
//! runtime cost. This is the "#[cfg(...)]-free specialization" pattern
//! requested in audit §4.2.
//!
//! # Gather helpers
//!
//! - [`gather_4d`] — autodiff-safe, borrows coordinate tensors (preserves graph)
//! - [`gather_4d_owned`] — non-autodiff, takes owned coordinate tensors (saves clones)
//!
//! # Clone counts
//!
//! | Path | Gather helper | Inside-gather clones | Graph preserved? |
//! |------|---------------|---------------------:|------------------|
//! | Autodiff (`B::ad_enabled() == true`) | `gather_4d` (borrowed) | 64 (four per call × 16) | yes |
//! | Non-autodiff (`B::ad_enabled() == false`) | `gather_4d_owned` (owned) | 16 (one per call × 16) | n/a |

use burn::tensor::{backend::Backend, Int, Tensor};

use crate::interpolation::shared::{in_bounds_mask, OutOfBoundsMode};

/// 4-D gather with borrowed coordinates — used by the autodiff path
/// where `xi`/`yi`/`zi`/`wi` must remain usable after the call. For
/// non-autodiff backends, prefer [`gather_4d_owned`] which avoids the
/// four coordinate-tensor clones per call.
#[inline]
fn gather_4d<B: Backend>(
    flat_data: &Tensor<B, 1>,
    xi: &Tensor<B, 1, Int>,
    yi: &Tensor<B, 1, Int>,
    zi: &Tensor<B, 1, Int>,
    wi: &Tensor<B, 1, Int>,
    strides: [i32; 3],
) -> Tensor<B, 1> {
    let [stride_y, stride_z, stride_w] = strides;
    let idx = wi.clone() * stride_w + zi.clone() * stride_z + yi.clone() * stride_y + xi.clone();
    flat_data.clone().gather(0, idx)
}

/// 4-D gather consuming the coordinate tensors — used by the non-autodiff
/// fast path. Eliminates the four `xi`/`yi`/`zi`/`wi` clones per call
/// (the caller transfers ownership). Still clones `flat_data` once per
/// call (unavoidable: Burn's `Tensor::gather` consumes `self`).
#[inline]
fn gather_4d_owned<B: Backend>(
    flat_data: &Tensor<B, 1>,
    xi: Tensor<B, 1, Int>,
    yi: Tensor<B, 1, Int>,
    zi: Tensor<B, 1, Int>,
    wi: Tensor<B, 1, Int>,
    strides: [i32; 3],
) -> Tensor<B, 1> {
    let [stride_y, stride_z, stride_w] = strides;
    let idx = wi * stride_w + zi * stride_z + yi * stride_y + xi;
    flat_data.clone().gather(0, idx)
}

/// Quadrilinear interpolation with const-generic autodiff dispatch (Sprint 355).
///
/// Single-function implementation: the `if B::ad_enabled()` branch is
/// monomorphized per backend and dead-code-eliminated at compile time,
/// so there is no runtime cost. This unifies the previous flat design
/// into one body, matching the `dim3.rs` shape and unblocking a future
/// macro-template unification of all 4 D-arms (D=1, 2, 3, 4).
pub(crate) fn interpolate_4d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // W (time/4th dim)
    let d1 = shape.dims[1]; // Z
    let d2 = shape.dims[2]; // Y
    let d3 = shape.dims[3]; // X
    let batch_size = indices.dims()[0];
    let device = indices.device();

    // indices: [N, 4] -> (x, y, z, w). narrow consumes self, so we
    // can't avoid the 3 clones of indices_local.
    let indices_local = indices;
    let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
    let y = indices_local.clone().narrow(1, 1, 1).squeeze_dims(&[1]);
    let z = indices_local.clone().narrow(1, 2, 1).squeeze_dims(&[1]);
    let w = indices_local.narrow(1, 3, 1).squeeze_dims(&[1]);

    // Floor coordinates and weights (identical in both paths).
    let x0 = x.clone().floor();
    let wx = x - x0.clone();
    let y0 = y.clone().floor();
    let wy = y - y0.clone();
    let z0 = z.clone().floor();
    let wz = z - z0.clone();
    let w0 = w.clone().floor();
    let ww = w - w0.clone();

    // Ceil coords.
    let x1 = x0.clone() + 1.0;
    let y1 = y0.clone() + 1.0;
    let z1 = z0.clone() + 1.0;
    let w1 = w0.clone() + 1.0;

    // Clamped int indices. Each is consumed by clamp+int; x0/y0/z0/w0
    // are still owned after weight derivation and used by
    // `in_bounds_mask` at the end.
    let x0_i = x0.clone().clamp(0.0, (d3 - 1) as f64).int();
    let y0_i = y0.clone().clamp(0.0, (d2 - 1) as f64).int();
    let z0_i = z0.clone().clamp(0.0, (d1 - 1) as f64).int();
    let w0_i = w0.clone().clamp(0.0, (d0 - 1) as f64).int();
    let x1_i = x1.clamp(0.0, (d3 - 1) as f64).int();
    let y1_i = y1.clamp(0.0, (d2 - 1) as f64).int();
    let z1_i = z1.clamp(0.0, (d1 - 1) as f64).int();
    let w1_i = w1.clamp(0.0, (d0 - 1) as f64).int();

    // Strides for [W, Z, Y, X] layout (d0, d1, d2, d3).
    let stride_w = (d1 * d2 * d3) as i32;
    let stride_z = (d2 * d3) as i32;
    let stride_y = d3 as i32;
    let strides = [stride_y, stride_z, stride_w];

    // Pre-flatten data — reshape consumes self, but data is &Tensor so
    // clone once.
    let flat_data = data.clone().reshape([d0 * d1 * d2 * d3]);

    // ── Const-generic dispatch (Sprint 355) ────────────────────────────
    // The `if B::ad_enabled()` branch is monomorphized per backend and
    // dead-code-eliminated at compile time. Only one arm is live in the
    // monomorphized function — the other is folded away by the compiler.
    //
    // - Autodiff arm: borrows `&x0_i` / `&y0_i` etc. (preserves Burn graph)
    // - Non-autodiff arm: transfers ownership of `x0_i` / `y0_i` etc. (saves clones)
    let (
        v0000,
        v0001,
        v0010,
        v0011,
        v0100,
        v0101,
        v0110,
        v0111,
        v1000,
        v1001,
        v1010,
        v1011,
        v1100,
        v1101,
        v1110,
        v1111,
    ) = if B::ad_enabled() {
        // Autodiff path: borrow coords (16 borrowed `gather_4d` calls).
        (
            gather_4d(&flat_data, &x0_i, &y0_i, &z0_i, &w0_i, strides),
            gather_4d(&flat_data, &x0_i, &y0_i, &z0_i, &w1_i, strides),
            gather_4d(&flat_data, &x0_i, &y0_i, &z1_i, &w0_i, strides),
            gather_4d(&flat_data, &x0_i, &y0_i, &z1_i, &w1_i, strides),
            gather_4d(&flat_data, &x0_i, &y1_i, &z0_i, &w0_i, strides),
            gather_4d(&flat_data, &x0_i, &y1_i, &z0_i, &w1_i, strides),
            gather_4d(&flat_data, &x0_i, &y1_i, &z1_i, &w0_i, strides),
            gather_4d(&flat_data, &x0_i, &y1_i, &z1_i, &w1_i, strides),
            gather_4d(&flat_data, &x1_i, &y0_i, &z0_i, &w0_i, strides),
            gather_4d(&flat_data, &x1_i, &y0_i, &z0_i, &w1_i, strides),
            gather_4d(&flat_data, &x1_i, &y0_i, &z1_i, &w0_i, strides),
            gather_4d(&flat_data, &x1_i, &y0_i, &z1_i, &w1_i, strides),
            gather_4d(&flat_data, &x1_i, &y1_i, &z0_i, &w0_i, strides),
            gather_4d(&flat_data, &x1_i, &y1_i, &z0_i, &w1_i, strides),
            gather_4d(&flat_data, &x1_i, &y1_i, &z1_i, &w0_i, strides),
            gather_4d(&flat_data, &x1_i, &y1_i, &z1_i, &w1_i, strides),
        )
    } else {
        // Non-autodiff path: own coords (16 owned `gather_4d_owned` calls).
        // Each index tensor is cloned 15 times (for its first 15 uses) and
        // consumed on its 16th use. Total caller-side clones: 4 × 15 = 60.
        (
            gather_4d_owned(
                &flat_data,
                x0_i.clone(),
                y0_i.clone(),
                z0_i.clone(),
                w0_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x0_i.clone(),
                y0_i.clone(),
                z0_i.clone(),
                w1_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x0_i.clone(),
                y0_i.clone(),
                z1_i.clone(),
                w0_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x0_i.clone(),
                y0_i.clone(),
                z1_i.clone(),
                w1_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x0_i.clone(),
                y1_i.clone(),
                z0_i.clone(),
                w0_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x0_i.clone(),
                y1_i.clone(),
                z0_i.clone(),
                w1_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x0_i.clone(),
                y1_i.clone(),
                z1_i.clone(),
                w0_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x0_i,
                y1_i.clone(),
                z1_i.clone(),
                w1_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x1_i.clone(),
                y0_i.clone(),
                z0_i.clone(),
                w0_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x1_i.clone(),
                y0_i.clone(),
                z0_i.clone(),
                w1_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x1_i.clone(),
                y0_i.clone(),
                z1_i.clone(),
                w0_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x1_i.clone(),
                y0_i.clone(),
                z1_i.clone(),
                w1_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x1_i.clone(),
                y1_i.clone(),
                z0_i.clone(),
                w0_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x1_i.clone(),
                y1_i.clone(),
                z0_i.clone(),
                w1_i.clone(),
                strides,
            ),
            gather_4d_owned(
                &flat_data,
                x1_i.clone(),
                y1_i.clone(),
                z1_i.clone(),
                w0_i.clone(),
                strides,
            ),
            gather_4d_owned(&flat_data, x1_i, y1_i, z1_i, w1_i, strides),
        )
    };

    // Quadrilinear lerp cascade (identical in both paths).
    let one = Tensor::<B, 1>::ones([batch_size], &device);
    let one_minus_wx = one.clone() - wx.clone();
    let one_minus_wy = one.clone() - wy.clone();
    let one_minus_wz = one.clone() - wz.clone();
    let one_minus_ww = one - ww.clone();

    // Interpolate along X
    let c000 = v0000 * one_minus_wx.clone() + v1000 * wx.clone();
    let c001 = v0001 * one_minus_wx.clone() + v1001 * wx.clone();
    let c010 = v0010 * one_minus_wx.clone() + v1010 * wx.clone();
    let c011 = v0011 * one_minus_wx.clone() + v1011 * wx.clone();
    let c100 = v0100 * one_minus_wx.clone() + v1100 * wx.clone();
    let c101 = v0101 * one_minus_wx.clone() + v1101 * wx.clone();
    let c110 = v0110 * one_minus_wx.clone() + v1110 * wx.clone();
    let c111 = v0111 * one_minus_wx + v1111 * wx;

    // Interpolate along Y
    let c00 = c000 * one_minus_wy.clone() + c100 * wy.clone();
    let c01 = c001 * one_minus_wy.clone() + c101 * wy.clone();
    let c10 = c010 * one_minus_wy.clone() + c110 * wy.clone();
    let c11 = c011 * one_minus_wy.clone() + c111 * wy.clone();

    // Interpolate along Z
    let c0 = c00 * one_minus_wz.clone() + c10 * wz.clone();
    let c1 = c01 * one_minus_wz.clone() + c11 * wz.clone();

    // Interpolate along W
    let result = c0 * one_minus_ww + c1 * ww;

    // In-bounds mask (identical in both paths).
    let x_mask = in_bounds_mask(x0, (d3 - 1) as f64, mode);
    let y_mask = in_bounds_mask(y0, (d2 - 1) as f64, mode);
    let z_mask = in_bounds_mask(z0, (d1 - 1) as f64, mode);
    let w_mask = in_bounds_mask(w0, (d0 - 1) as f64, mode);

    match (x_mask, y_mask, z_mask, w_mask) {
        (Some(xm), Some(ym), Some(zm), Some(wm)) => result * xm * ym * zm * wm,
        _ => result,
    }
}
