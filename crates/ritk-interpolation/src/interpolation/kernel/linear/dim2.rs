//! 2-D linear interpolation kernel (Sprint 356).
//!
//! Generated via the [`ritk_macros::interp_dim_template!`] proc-macro
//! which resolves the `macro_rules!` hygiene barrier (see
//! `docs/audit_optimization_sprint_350.md` §4.2.2 / §7.7).
//!
//! The proc-macro generates the prelude and mask application; this
//! file supplies only the gather + bilinear lerp cascade body.

use burn::tensor::Tensor;


ritk_macros::interp_dim_template!(2, interpolate_2d, x, y, wx, wy, d1 - 1, d0 - 1, {
    let flat_data = data.clone().reshape([d0 * d1]);

    // ── Const-generic dispatch (Sprint 355) ────────────────────────
    let (v00, v01, v10, v11) = if B::ad_enabled() {
        let idx00 = y0_i.clone() * stride_y + x0_i.clone();
        let idx01 = y1_i.clone() * stride_y + x0_i.clone();
        let idx10 = y0_i.clone() * stride_y + x1_i.clone();
        let idx11 = y1_i.clone() * stride_y + x1_i.clone();

        let all_indices = Tensor::cat(vec![idx00, idx01, idx10, idx11], 0);
        let all_values = flat_data.clone().gather(0, all_indices);

        let v00 = all_values.clone().slice([0..batch_size]);
        let v01 = all_values.clone().slice([batch_size..2 * batch_size]);
        let v10 = all_values.clone().slice([2 * batch_size..3 * batch_size]);
        let v11 = all_values.slice([3 * batch_size..4 * batch_size]);

        (v00, v01, v10, v11)
    } else {
        let idx00 = y0_i.clone() * stride_y + x0_i.clone();
        let idx01 = y1_i.clone() * stride_y + x0_i.clone();
        let idx10 = y0_i.clone() * stride_y + x1_i.clone();
        let idx11 = y1_i * stride_y + x1_i;

        let all_indices = Tensor::cat(vec![idx00, idx01, idx10, idx11], 0);
        let all_values = flat_data.clone().gather(0, all_indices);

        let v00 = all_values.clone().slice([0..batch_size]);
        let v01 = all_values.clone().slice([batch_size..2 * batch_size]);
        let v10 = all_values.clone().slice([2 * batch_size..3 * batch_size]);
        let v11 = all_values.slice([3 * batch_size..4 * batch_size]);

        (v00, v01, v10, v11)
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
            let idx00 = y0_i.clone() * stride_y + x0_i.clone();
            let idx01 = y1_i.clone() * stride_y + x0_i.clone();
            let idx10 = y0_i.clone() * stride_y + x1_i.clone();
            let idx11 = y1_i.clone() * stride_y + x1_i.clone();

            let all_indices = Tensor::cat(vec![idx00, idx01, idx10, idx11], 0);
            let all_values = flat_data.clone().gather(0, all_indices);

            let v00 = all_values.clone().slice([0..batch_size]);
            let v01 = all_values.clone().slice([batch_size..2 * batch_size]);
            let v10 = all_values.clone().slice([2 * batch_size..3 * batch_size]);
            let v11 = all_values.slice([3 * batch_size..4 * batch_size]);

            (v00, v01, v10, v11)
        } else {
            let idx00 = y0_i.clone() * stride_y + x0_i.clone();
            let idx01 = y1_i.clone() * stride_y + x0_i.clone();
            let idx10 = y0_i.clone() * stride_y + x1_i.clone();
            let idx11 = y1_i * stride_y + x1_i;

            let all_indices = Tensor::cat(vec![idx00, idx01, idx10, idx11], 0);
            let all_values = flat_data.clone().gather(0, all_indices);

            let v00 = all_values.clone().slice([0..batch_size]);
            let v01 = all_values.clone().slice([batch_size..2 * batch_size]);
            let v10 = all_values.clone().slice([2 * batch_size..3 * batch_size]);
            let v11 = all_values.slice([3 * batch_size..4 * batch_size]);

            (v00, v01, v10, v11)
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
