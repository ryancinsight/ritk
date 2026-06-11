//! 4-D linear interpolation kernel (Sprint 356).
//!
//! Generated via the [`ritk_macros::interp_dim_template!`] proc-macro
//! which resolves the `macro_rules!` hygiene barrier (see
//! `docs/audit_optimization_sprint_350.md` §4.2.2 / §7.7).
//!
//! The proc-macro generates the prelude and mask application; this
//! file supplies only the gather + quadrilinear lerp cascade body
//! with the inline `B::ad_enabled()` const-generic dispatch.

use burn::tensor::{backend::Backend, Int, Tensor};


/// 4-D gather with borrowed coordinates — used by the autodiff path.
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
/// fast path.
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

ritk_macros::interp_dim_template!(
    4,
    interpolate_4d,
    x, y, z, w,
    wx, wy, wz, ww,
    d3 - 1, d2 - 1, d1 - 1, d0 - 1,
    {
        let flat_data = data.clone().reshape([d0 * d1 * d2 * d3]);

        // ── Const-generic dispatch (Sprint 355) ────────────────────────
        let (
            v0000, v0001, v0010, v0011,
            v0100, v0101, v0110, v0111,
            v1000, v1001, v1010, v1011,
            v1100, v1101, v1110, v1111,
        ) = if B::ad_enabled() {
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
            (
                gather_4d_owned(&flat_data, x0_i.clone(), y0_i.clone(), z0_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y0_i.clone(), z0_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y0_i.clone(), z1_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y0_i.clone(), z1_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y1_i.clone(), z0_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y1_i.clone(), z0_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y1_i.clone(), z1_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i, y1_i.clone(), z1_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y0_i.clone(), z0_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y0_i.clone(), z0_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y0_i.clone(), z1_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y0_i.clone(), z1_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y1_i.clone(), z0_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y1_i.clone(), z0_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y1_i.clone(), z1_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i, y1_i, z1_i, w1_i, strides),
            )
        };

        // Quadrilinear lerp cascade.
        let one = Tensor::<B, 1>::ones([batch_size], &_device);
        let one_minus_wx = one.clone() - wx.clone();
        let one_minus_wy = one.clone() - wy.clone();
        let one_minus_wz = one.clone() - wz.clone();
        let one_minus_ww = one - ww.clone();

        let c000 = v0000 * one_minus_wx.clone() + v1000 * wx.clone();
        let c001 = v0001 * one_minus_wx.clone() + v1001 * wx.clone();
        let c010 = v0010 * one_minus_wx.clone() + v1010 * wx.clone();
        let c011 = v0011 * one_minus_wx.clone() + v1011 * wx.clone();
        let c100 = v0100 * one_minus_wx.clone() + v1100 * wx.clone();
        let c101 = v0101 * one_minus_wx.clone() + v1101 * wx.clone();
        let c110 = v0110 * one_minus_wx.clone() + v1110 * wx.clone();
        let c111 = v0111 * one_minus_wx + v1111 * wx;

        let c00 = c000 * one_minus_wy.clone() + c100 * wy.clone();
        let c01 = c001 * one_minus_wy.clone() + c101 * wy.clone();
        let c10 = c010 * one_minus_wy.clone() + c110 * wy.clone();
        let c11 = c011 * one_minus_wy.clone() + c111 * wy.clone();

        let c0 = c00 * one_minus_wz.clone() + c10 * wz.clone();
        let c1 = c01 * one_minus_wz.clone() + c11 * wz.clone();

        c0 * one_minus_ww + c1 * ww
    }
);

// ════════════════════════════════════════════════════════════════════════
//  Const-generic shape specialization (audit §8 351-01)
// ════════════════════════════════════════════════════════════════════════
//
// Parallel to `interpolate_4d` above, but takes the volume shape as
// `const D0: usize, const D1: usize, const D2: usize, const D3: usize`.
// This enables compile-time bounds, mask inlining, and per-shape
// monomorphization.
ritk_macros::interp_dim_template_typed!(
    4,
    interpolate_4d_typed,
    x, y, z, w,
    wx, wy, wz, ww,
    D3 - 1, D2 - 1, D1 - 1, D0 - 1,
    D0, D1, D2, D3,
    {
        let flat_data = data.clone().reshape([d0 * d1 * d2 * d3]);

        // ── Const-generic dispatch (Sprint 355) ────────────────────────
        let (
            v0000, v0001, v0010, v0011,
            v0100, v0101, v0110, v0111,
            v1000, v1001, v1010, v1011,
            v1100, v1101, v1110, v1111,
        ) = if B::ad_enabled() {
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
            (
                gather_4d_owned(&flat_data, x0_i.clone(), y0_i.clone(), z0_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y0_i.clone(), z0_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y0_i.clone(), z1_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y0_i.clone(), z1_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y1_i.clone(), z0_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y1_i.clone(), z0_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i.clone(), y1_i.clone(), z1_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x0_i, y1_i.clone(), z1_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y0_i.clone(), z0_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y0_i.clone(), z0_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y0_i.clone(), z1_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y0_i.clone(), z1_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y1_i.clone(), z0_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y1_i.clone(), z0_i.clone(), w1_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i.clone(), y1_i.clone(), z1_i.clone(), w0_i.clone(), strides),
                gather_4d_owned(&flat_data, x1_i, y1_i, z1_i, w1_i, strides),
            )
        };

        // Quadrilinear lerp cascade.
        let one = Tensor::<B, 1>::ones([batch_size], &_device);
        let one_minus_wx = one.clone() - wx.clone();
        let one_minus_wy = one.clone() - wy.clone();
        let one_minus_wz = one.clone() - wz.clone();
        let one_minus_ww = one - ww.clone();

        let c000 = v0000 * one_minus_wx.clone() + v1000 * wx.clone();
        let c001 = v0001 * one_minus_wx.clone() + v1001 * wx.clone();
        let c010 = v0010 * one_minus_wx.clone() + v1010 * wx.clone();
        let c011 = v0011 * one_minus_wx.clone() + v1011 * wx.clone();
        let c100 = v0100 * one_minus_wx.clone() + v1100 * wx.clone();
        let c101 = v0101 * one_minus_wx.clone() + v1101 * wx.clone();
        let c110 = v0110 * one_minus_wx.clone() + v1110 * wx.clone();
        let c111 = v0111 * one_minus_wx + v1111 * wx;

        let c00 = c000 * one_minus_wy.clone() + c100 * wy.clone();
        let c01 = c001 * one_minus_wy.clone() + c101 * wy.clone();
        let c10 = c010 * one_minus_wy.clone() + c110 * wy.clone();
        let c11 = c011 * one_minus_wy.clone() + c111 * wy.clone();

        let c0 = c00 * one_minus_wz.clone() + c10 * wz.clone();
        let c1 = c01 * one_minus_wz.clone() + c11 * wz.clone();

        c0 * one_minus_ww + c1 * ww
    }
);
