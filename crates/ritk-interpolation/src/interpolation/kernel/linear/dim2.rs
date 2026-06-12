//! 2-D linear interpolation kernel (Sprint 356).
//!
//! Generated via the [`ritk_macros::interp_dim_template!`] proc-macro
//! which resolves the `macro_rules!` hygiene barrier (see
//! `docs/audit_optimization_sprint_350.md` §4.2.2 / §7.7).
//!
//! The proc-macro generates the prelude and mask application; this
//! file supplies only the gather + bilinear lerp cascade body.

use burn::tensor::{backend::Backend, Int, Tensor};

/// 2-D gather with borrowed coordinates — used by the autodiff path.
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
/// fast path.
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

ritk_macros::interp_dim_template!(2, interpolate_2d, x, y, wx, wy, d1 - 1, d0 - 1, {
    let flat_data = data.clone().reshape([d0 * d1]);

    // ── Const-generic dispatch (Sprint 355) ────────────────────────
    let (v00, v01, v10, v11) = if B::ad_enabled() {
        (
            gather_2d(&flat_data, &x0_i, &y0_i, stride_y),
            gather_2d(&flat_data, &x0_i, &y1_i, stride_y),
            gather_2d(&flat_data, &x1_i, &y0_i, stride_y),
            gather_2d(&flat_data, &x1_i, &y1_i, stride_y),
        )
    } else {
        (
            gather_2d_owned(&flat_data, x0_i.clone(), y0_i.clone(), stride_y),
            gather_2d_owned(&flat_data, x0_i, y1_i.clone(), stride_y),
            gather_2d_owned(&flat_data, x1_i.clone(), y0_i, stride_y),
            gather_2d_owned(&flat_data, x1_i, y1_i, stride_y),
        )
    };

    // Bilinear lerp cascade.
    let one = Tensor::<B, 1>::ones([batch_size], &_device);
    let one_minus_wx = one.clone() - wx.clone();
    let one_minus_wy = one - wy.clone();

    let c0 = v00 * one_minus_wx.clone() + v10 * wx.clone();
    let c1 = v01 * one_minus_wx + v11 * wx;

    c0 * one_minus_wy + c1 * wy
});

// ════════════════════════════════════════════════════════════════════════
//  Const-generic shape specialization (audit §8 351-01)
// ════════════════════════════════════════════════════════════════════════
//
// Parallel to `interpolate_2d` above, but takes the volume shape as
// `const D0: usize, const D1: usize`. This enables compile-time bounds,
// mask inlining, and per-shape monomorphization.
ritk_macros::interp_dim_template_typed!(
    2,
    interpolate_2d_typed,
    x,
    y,
    wx,
    wy,
    D1 - 1,
    D0 - 1,
    D0,
    D1,
    {
        let flat_data = data.clone().reshape([d0 * d1]);

        // ── Const-generic dispatch (Sprint 355) ────────────────────────
        let (v00, v01, v10, v11) = if B::ad_enabled() {
            (
                gather_2d(&flat_data, &x0_i, &y0_i, stride_y),
                gather_2d(&flat_data, &x0_i, &y1_i, stride_y),
                gather_2d(&flat_data, &x1_i, &y0_i, stride_y),
                gather_2d(&flat_data, &x1_i, &y1_i, stride_y),
            )
        } else {
            (
                gather_2d_owned(&flat_data, x0_i.clone(), y0_i.clone(), stride_y),
                gather_2d_owned(&flat_data, x0_i, y1_i.clone(), stride_y),
                gather_2d_owned(&flat_data, x1_i.clone(), y0_i, stride_y),
                gather_2d_owned(&flat_data, x1_i, y1_i, stride_y),
            )
        };

        // Bilinear lerp cascade.
        let one = Tensor::<B, 1>::ones([batch_size], &_device);
        let one_minus_wx = one.clone() - wx.clone();
        let one_minus_wy = one - wy.clone();

        let c0 = v00 * one_minus_wx.clone() + v10 * wx.clone();
        let c1 = v01 * one_minus_wx + v11 * wx;

        c0 * one_minus_wy + c1 * wy
    }
);
