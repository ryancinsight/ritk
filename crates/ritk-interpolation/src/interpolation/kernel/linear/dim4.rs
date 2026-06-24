//! 4-D linear interpolation kernel (Sprint 356).
//!
//! Generated via the [`ritk_macros::interp_dim_template!`] proc-macro
//! which resolves the `macro_rules!` hygiene barrier (see
//! `docs/audit_optimization_sprint_350.md` §4.2.2 / §7.7).
//!
//! The proc-macro generates the prelude and mask application; this
//! file supplies only the gather + quadrilinear lerp cascade body
//! with the inline `B::ad_enabled()` const-generic dispatch.

use super::slice_batch;
use burn::tensor::Tensor;

ritk_macros::interp_dim_template!(
    4,
    interpolate_4d,
    x,
    y,
    z,
    w,
    wx,
    wy,
    wz,
    ww,
    d3 - 1,
    d2 - 1,
    d1 - 1,
    d0 - 1,
    {
        let flat_data = data.clone().reshape([d0 * d1 * d2 * d3]);

        // ── Const-generic dispatch (Sprint 355) ────────────────────────
        let [stride_y, stride_z, stride_w] = strides;
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
            let idx0000 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0001 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0010 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0011 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0100 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0101 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0110 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0111 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx1000 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1001 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1010 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1011 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1100 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();
            let idx1101 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();
            let idx1110 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();
            let idx1111 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();

            let all_indices = Tensor::cat(
                vec![
                    idx0000, idx0001, idx0010, idx0011, idx0100, idx0101, idx0110, idx0111,
                    idx1000, idx1001, idx1010, idx1011, idx1100, idx1101, idx1110, idx1111,
                ],
                0,
            );
            let all_values = flat_data.clone().gather(0, all_indices);

            let v0000 = slice_batch(all_values.clone(), 0, batch_size);
            let v0001 = slice_batch(all_values.clone(), batch_size, 2 * batch_size);
            let v0010 = slice_batch(all_values.clone(), 2 * batch_size, 3 * batch_size);
            let v0011 = slice_batch(all_values.clone(), 3 * batch_size, 4 * batch_size);
            let v0100 = slice_batch(all_values.clone(), 4 * batch_size, 5 * batch_size);
            let v0101 = slice_batch(all_values.clone(), 5 * batch_size, 6 * batch_size);
            let v0110 = slice_batch(all_values.clone(), 6 * batch_size, 7 * batch_size);
            let v0111 = slice_batch(all_values.clone(), 7 * batch_size, 8 * batch_size);
            let v1000 = slice_batch(all_values.clone(), 8 * batch_size, 9 * batch_size);
            let v1001 = slice_batch(all_values.clone(), 9 * batch_size, 10 * batch_size);
            let v1010 = slice_batch(all_values.clone(), 10 * batch_size, 11 * batch_size);
            let v1011 = slice_batch(all_values.clone(), 11 * batch_size, 12 * batch_size);
            let v1100 = slice_batch(all_values.clone(), 12 * batch_size, 13 * batch_size);
            let v1101 = slice_batch(all_values.clone(), 13 * batch_size, 14 * batch_size);
            let v1110 = slice_batch(all_values.clone(), 14 * batch_size, 15 * batch_size);
            let v1111 = slice_batch(all_values, 15 * batch_size, 16 * batch_size);

            (
                v0000, v0001, v0010, v0011, v0100, v0101, v0110, v0111, v1000, v1001, v1010, v1011,
                v1100, v1101, v1110, v1111,
            )
        } else {
            let idx0000 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0001 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0010 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0011 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0100 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0101 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0110 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0111 = w1_i.clone() * stride_w + z1_i.clone() * stride_y + x0_i;
            let idx1000 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1001 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1010 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1011 =
                w1_i.clone() * stride_w + z1_i.clone() * stride_z + y0_i * stride_y + x1_i.clone();
            let idx1100 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();
            let idx1101 =
                w1_i.clone() * stride_w + z0_i * stride_z + y1_i.clone() * stride_y + x1_i.clone();
            let idx1110 =
                w0_i * stride_w + z1_i.clone() * stride_z + y1_i.clone() * stride_y + x1_i.clone();
            let idx1111 = w1_i * stride_w + z1_i * stride_z + y1_i * stride_y + x1_i;

            let all_indices = Tensor::cat(
                vec![
                    idx0000, idx0001, idx0010, idx0011, idx0100, idx0101, idx0110, idx0111,
                    idx1000, idx1001, idx1010, idx1011, idx1100, idx1101, idx1110, idx1111,
                ],
                0,
            );
            let all_values = flat_data.clone().gather(0, all_indices);

            let v0000 = slice_batch(all_values.clone(), 0, batch_size);
            let v0001 = slice_batch(all_values.clone(), batch_size, 2 * batch_size);
            let v0010 = slice_batch(all_values.clone(), 2 * batch_size, 3 * batch_size);
            let v0011 = slice_batch(all_values.clone(), 3 * batch_size, 4 * batch_size);
            let v0100 = slice_batch(all_values.clone(), 4 * batch_size, 5 * batch_size);
            let v0101 = slice_batch(all_values.clone(), 5 * batch_size, 6 * batch_size);
            let v0110 = slice_batch(all_values.clone(), 6 * batch_size, 7 * batch_size);
            let v0111 = slice_batch(all_values.clone(), 7 * batch_size, 8 * batch_size);
            let v1000 = slice_batch(all_values.clone(), 8 * batch_size, 9 * batch_size);
            let v1001 = slice_batch(all_values.clone(), 9 * batch_size, 10 * batch_size);
            let v1010 = slice_batch(all_values.clone(), 10 * batch_size, 11 * batch_size);
            let v1011 = slice_batch(all_values.clone(), 11 * batch_size, 12 * batch_size);
            let v1100 = slice_batch(all_values.clone(), 12 * batch_size, 13 * batch_size);
            let v1101 = slice_batch(all_values.clone(), 13 * batch_size, 14 * batch_size);
            let v1110 = slice_batch(all_values.clone(), 14 * batch_size, 15 * batch_size);
            let v1111 = slice_batch(all_values, 15 * batch_size, 16 * batch_size);

            (
                v0000, v0001, v0010, v0011, v0100, v0101, v0110, v0111, v1000, v1001, v1010, v1011,
                v1100, v1101, v1110, v1111,
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
    x,
    y,
    z,
    w,
    wx,
    wy,
    wz,
    ww,
    D3 - 1,
    D2 - 1,
    D1 - 1,
    D0 - 1,
    D0,
    D1,
    D2,
    D3,
    {
        let flat_data = data.clone().reshape([d0 * d1 * d2 * d3]);

        // ── Const-generic dispatch (Sprint 355) ────────────────────────
        let [stride_y, stride_z, stride_w] = strides;
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
            let idx0000 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0001 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0010 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0011 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0100 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0101 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0110 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0111 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx1000 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1001 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1010 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1011 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1100 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();
            let idx1101 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();
            let idx1110 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();
            let idx1111 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();

            let all_indices = Tensor::cat(
                vec![
                    idx0000, idx0001, idx0010, idx0011, idx0100, idx0101, idx0110, idx0111,
                    idx1000, idx1001, idx1010, idx1011, idx1100, idx1101, idx1110, idx1111,
                ],
                0,
            );
            let all_values = flat_data.clone().gather(0, all_indices);

            let v0000 = slice_batch(all_values.clone(), 0, batch_size);
            let v0001 = slice_batch(all_values.clone(), batch_size, 2 * batch_size);
            let v0010 = slice_batch(all_values.clone(), 2 * batch_size, 3 * batch_size);
            let v0011 = slice_batch(all_values.clone(), 3 * batch_size, 4 * batch_size);
            let v0100 = slice_batch(all_values.clone(), 4 * batch_size, 5 * batch_size);
            let v0101 = slice_batch(all_values.clone(), 5 * batch_size, 6 * batch_size);
            let v0110 = slice_batch(all_values.clone(), 6 * batch_size, 7 * batch_size);
            let v0111 = slice_batch(all_values.clone(), 7 * batch_size, 8 * batch_size);
            let v1000 = slice_batch(all_values.clone(), 8 * batch_size, 9 * batch_size);
            let v1001 = slice_batch(all_values.clone(), 9 * batch_size, 10 * batch_size);
            let v1010 = slice_batch(all_values.clone(), 10 * batch_size, 11 * batch_size);
            let v1011 = slice_batch(all_values.clone(), 11 * batch_size, 12 * batch_size);
            let v1100 = slice_batch(all_values.clone(), 12 * batch_size, 13 * batch_size);
            let v1101 = slice_batch(all_values.clone(), 13 * batch_size, 14 * batch_size);
            let v1110 = slice_batch(all_values.clone(), 14 * batch_size, 15 * batch_size);
            let v1111 = slice_batch(all_values, 15 * batch_size, 16 * batch_size);

            (
                v0000, v0001, v0010, v0011, v0100, v0101, v0110, v0111, v1000, v1001, v1010, v1011,
                v1100, v1101, v1110, v1111,
            )
        } else {
            let idx0000 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0001 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0010 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0011 = w1_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x0_i.clone();
            let idx0100 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0101 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0110 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x0_i.clone();
            let idx0111 = w1_i.clone() * stride_w + z1_i.clone() * stride_y + x0_i;
            let idx1000 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1001 = w1_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1010 = w0_i.clone() * stride_w
                + z1_i.clone() * stride_z
                + y0_i.clone() * stride_y
                + x1_i.clone();
            let idx1011 =
                w1_i.clone() * stride_w + z1_i.clone() * stride_z + y0_i * stride_y + x1_i.clone();
            let idx1100 = w0_i.clone() * stride_w
                + z0_i.clone() * stride_z
                + y1_i.clone() * stride_y
                + x1_i.clone();
            let idx1101 =
                w1_i.clone() * stride_w + z0_i * stride_z + y1_i.clone() * stride_y + x1_i.clone();
            let idx1110 =
                w0_i * stride_w + z1_i.clone() * stride_z + y1_i.clone() * stride_y + x1_i.clone();
            let idx1111 = w1_i * stride_w + z1_i * stride_z + y1_i * stride_y + x1_i;

            let all_indices = Tensor::cat(
                vec![
                    idx0000, idx0001, idx0010, idx0011, idx0100, idx0101, idx0110, idx0111,
                    idx1000, idx1001, idx1010, idx1011, idx1100, idx1101, idx1110, idx1111,
                ],
                0,
            );
            let all_values = flat_data.clone().gather(0, all_indices);

            let v0000 = slice_batch(all_values.clone(), 0, batch_size);
            let v0001 = slice_batch(all_values.clone(), batch_size, 2 * batch_size);
            let v0010 = slice_batch(all_values.clone(), 2 * batch_size, 3 * batch_size);
            let v0011 = slice_batch(all_values.clone(), 3 * batch_size, 4 * batch_size);
            let v0100 = slice_batch(all_values.clone(), 4 * batch_size, 5 * batch_size);
            let v0101 = slice_batch(all_values.clone(), 5 * batch_size, 6 * batch_size);
            let v0110 = slice_batch(all_values.clone(), 6 * batch_size, 7 * batch_size);
            let v0111 = slice_batch(all_values.clone(), 7 * batch_size, 8 * batch_size);
            let v1000 = slice_batch(all_values.clone(), 8 * batch_size, 9 * batch_size);
            let v1001 = slice_batch(all_values.clone(), 9 * batch_size, 10 * batch_size);
            let v1010 = slice_batch(all_values.clone(), 10 * batch_size, 11 * batch_size);
            let v1011 = slice_batch(all_values.clone(), 11 * batch_size, 12 * batch_size);
            let v1100 = slice_batch(all_values.clone(), 12 * batch_size, 13 * batch_size);
            let v1101 = slice_batch(all_values.clone(), 13 * batch_size, 14 * batch_size);
            let v1110 = slice_batch(all_values.clone(), 14 * batch_size, 15 * batch_size);
            let v1111 = slice_batch(all_values, 15 * batch_size, 16 * batch_size);

            (
                v0000, v0001, v0010, v0011, v0100, v0101, v0110, v0111, v1000, v1001, v1010, v1011,
                v1100, v1101, v1110, v1111,
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
