//! 3-D linear interpolation kernel (Sprint 356).
//!
//! Generated via the [`ritk_macros::interp_dim_template!`] proc-macro
//! which resolves the `macro_rules!` hygiene barrier (see
//! `docs/audit_optimization_sprint_350.md` §4.2.2 / §7.7).
//!
//! The proc-macro generates the prelude and mask application; this
//! file supplies only the gather + trilinear lerp cascade body with
//! the inline `B::ad_enabled()` const-generic dispatch.

use super::slice_batch;
use coeus_tensor::Tensor;

ritk_macros::interp_dim_template!(
    3,
    interpolate_3d,
    x,
    y,
    z,
    wx,
    wy,
    wz,
    d2 - 1,
    d1 - 1,
    d0 - 1,
    {
        let flat_data = data.clone().reshape([d0 * d1 * d2]);

        // ── Const-generic dispatch (Sprint 355) ────────────────────────
        let (v000, v001, v010, v011, v100, v101, v110, v111) = if B::ad_enabled() {
            let idx000 = z0_i.clone() * stride_z + y0_i.clone() * stride_y + x0_i.clone();
            let idx001 = z1_i.clone() * stride_z + y0_i.clone() * stride_y + x0_i.clone();
            let idx010 = z0_i.clone() * stride_z + y1_i.clone() * stride_y + x0_i.clone();
            let idx011 = z1_i.clone() * stride_z + y1_i.clone() * stride_y + x0_i.clone();
            let idx100 = z0_i.clone() * stride_z + y0_i.clone() * stride_y + x1_i.clone();
            let idx101 = z1_i.clone() * stride_z + y0_i.clone() * stride_y + x1_i.clone();
            let idx110 = z0_i.clone() * stride_z + y1_i.clone() * stride_y + x1_i.clone();
            let idx111 = z1_i.clone() * stride_z + y1_i.clone() * stride_y + x1_i.clone();

            let all_indices = Tensor::cat(
                vec![
                    idx000, idx001, idx010, idx011, idx100, idx101, idx110, idx111,
                ],
                0,
            );
            let all_values = flat_data.clone().gather(0, all_indices);

            let v000 = slice_batch(all_values.clone(), 0, batch_size);
            let v001 = slice_batch(all_values.clone(), batch_size, 2 * batch_size);
            let v010 = slice_batch(all_values.clone(), 2 * batch_size, 3 * batch_size);
            let v011 = slice_batch(all_values.clone(), 3 * batch_size, 4 * batch_size);
            let v100 = slice_batch(all_values.clone(), 4 * batch_size, 5 * batch_size);
            let v101 = slice_batch(all_values.clone(), 5 * batch_size, 6 * batch_size);
            let v110 = slice_batch(all_values.clone(), 6 * batch_size, 7 * batch_size);
            let v111 = slice_batch(all_values, 7 * batch_size, 8 * batch_size);

            (v000, v001, v010, v011, v100, v101, v110, v111)
        } else {
            let idx000 = z0_i.clone() * stride_z + y0_i.clone() * stride_y + x0_i.clone();
            let idx001 = z1_i.clone() * stride_z + y0_i.clone() * stride_y + x0_i.clone();
            let idx010 = z0_i.clone() * stride_z + y1_i.clone() * stride_y + x0_i.clone();
            let idx011 = z1_i.clone() * stride_z + y1_i.clone() * stride_y + x0_i.clone();
            let idx100 = z0_i.clone() * stride_z + y0_i.clone() * stride_y + x1_i.clone();
            let idx101 = z1_i.clone() * stride_z + y0_i.clone() * stride_y + x1_i.clone();
            let idx110 = z0_i.clone() * stride_z + y1_i.clone() * stride_y + x1_i.clone();
            let idx111 = z1_i * stride_z + y1_i * stride_y + x1_i;

            let all_indices = Tensor::cat(
                vec![
                    idx000, idx001, idx010, idx011, idx100, idx101, idx110, idx111,
                ],
                0,
            );
            let all_values = flat_data.clone().gather(0, all_indices);

            let v000 = slice_batch(all_values.clone(), 0, batch_size);
            let v001 = slice_batch(all_values.clone(), batch_size, 2 * batch_size);
            let v010 = slice_batch(all_values.clone(), 2 * batch_size, 3 * batch_size);
            let v011 = slice_batch(all_values.clone(), 3 * batch_size, 4 * batch_size);
            let v100 = slice_batch(all_values.clone(), 4 * batch_size, 5 * batch_size);
            let v101 = slice_batch(all_values.clone(), 5 * batch_size, 6 * batch_size);
            let v110 = slice_batch(all_values.clone(), 6 * batch_size, 7 * batch_size);
            let v111 = slice_batch(all_values, 7 * batch_size, 8 * batch_size);

            (v000, v001, v010, v011, v100, v101, v110, v111)
        };

        // Trilinear lerp cascade.
        let one = Tensor::<f32, B>::ones([batch_size], &_device);
        let one_minus_wx = one.clone() - wx.clone();
        let one_minus_wy = one.clone() - wy.clone();
        let one_minus_wz = one - wz.clone();

        let c00 = v000 * one_minus_wx.clone() + v100 * wx.clone();
        let c01 = v001 * one_minus_wx.clone() + v101 * wx.clone();
        let c10 = v010 * one_minus_wx.clone() + v110 * wx.clone();
        let c11 = v011 * one_minus_wx + v111 * wx;

        let c0 = c00 * one_minus_wy.clone() + c10 * wy.clone();
        let c1 = c01 * one_minus_wy.clone() + c11 * wy.clone();

        c0 * one_minus_wz + c1 * wz
    }
);

// ════════════════════════════════════════════════════════════════════════
//  Const-generic shape specialization (audit §8 351-01)
// ════════════════════════════════════════════════════════════════════════
//
// The runtime `interpolate_3d` above reads the volume shape from
// `data.shape()` on every call. `interpolate_3d_typed` is a parallel
// const-generic variant that takes the shape as `const D0: usize, const
// D1: usize, const D2: usize` parameters, enabling:
//   1. **Compile-time bounds**: the `clamp(0.0, (D2 - 1) as f64)` calls
//      become constant expressions — no runtime arithmetic.
//   2. **Mask inlining**: `in_bounds_mask(x0, (D2 - 1) as f64, mode)`
//      gets a compile-time-known max — the compiler can inline and
//      eliminate the function call for `OutOfBoundsMode::Extend`.
//   3. **No `data.shape()` read**: saves 3 memory loads per call.
//   4. **Monomorphization**: each `(D0, D1, D2)` triple is a separate
//      monomorphized function.
//
// Use this from hot paths where the volume size is known at compile
// time (e.g. a fixed-size benchmark, or a registration pipeline that
// loads a fixed-size volume). For dynamic-shape callers, the runtime
// `interpolate_3d` is the right choice.
ritk_macros::interp_dim_template_typed!(
    3,
    interpolate_3d_typed,
    x,
    y,
    z,
    wx,
    wy,
    wz,
    D0 - 1,
    D1 - 1,
    D2 - 1,
    D0,
    D1,
    D2,
    {
        let flat_data = data.clone().reshape([d0 * d1 * d2]);

        // ── Const-generic dispatch (Sprint 355) ────────────────────────
        let (v000, v001, v010, v011, v100, v101, v110, v111) = if B::ad_enabled() {
            let idx000 = z0_i.clone() * stride_z + y0_i.clone() * stride_y + x0_i.clone();
            let idx001 = z1_i.clone() * stride_z + y0_i.clone() * stride_y + x0_i.clone();
            let idx010 = z0_i.clone() * stride_z + y1_i.clone() * stride_y + x0_i.clone();
            let idx011 = z1_i.clone() * stride_z + y1_i.clone() * stride_y + x0_i.clone();
            let idx100 = z0_i.clone() * stride_z + y0_i.clone() * stride_y + x1_i.clone();
            let idx101 = z1_i.clone() * stride_z + y0_i.clone() * stride_y + x1_i.clone();
            let idx110 = z0_i.clone() * stride_z + y1_i.clone() * stride_y + x1_i.clone();
            let idx111 = z1_i.clone() * stride_z + y1_i.clone() * stride_y + x1_i.clone();

            let all_indices = Tensor::cat(
                vec![
                    idx000, idx001, idx010, idx011, idx100, idx101, idx110, idx111,
                ],
                0,
            );
            let all_values = flat_data.clone().gather(0, all_indices);

            let v000 = slice_batch(all_values.clone(), 0, batch_size);
            let v001 = slice_batch(all_values.clone(), batch_size, 2 * batch_size);
            let v010 = slice_batch(all_values.clone(), 2 * batch_size, 3 * batch_size);
            let v011 = slice_batch(all_values.clone(), 3 * batch_size, 4 * batch_size);
            let v100 = slice_batch(all_values.clone(), 4 * batch_size, 5 * batch_size);
            let v101 = slice_batch(all_values.clone(), 5 * batch_size, 6 * batch_size);
            let v110 = slice_batch(all_values.clone(), 6 * batch_size, 7 * batch_size);
            let v111 = slice_batch(all_values, 7 * batch_size, 8 * batch_size);

            (v000, v001, v010, v011, v100, v101, v110, v111)
        } else {
            let idx000 = z0_i.clone() * stride_z + y0_i.clone() * stride_y + x0_i.clone();
            let idx001 = z1_i.clone() * stride_z + y0_i.clone() * stride_y + x0_i.clone();
            let idx010 = z0_i.clone() * stride_z + y1_i.clone() * stride_y + x0_i.clone();
            let idx011 = z1_i.clone() * stride_z + y1_i.clone() * stride_y + x0_i.clone();
            let idx100 = z0_i.clone() * stride_z + y0_i.clone() * stride_y + x1_i.clone();
            let idx101 = z1_i.clone() * stride_z + y0_i.clone() * stride_y + x1_i.clone();
            let idx110 = z0_i.clone() * stride_z + y1_i.clone() * stride_y + x1_i.clone();
            let idx111 = z1_i * stride_z + y1_i * stride_y + x1_i;

            let all_indices = Tensor::cat(
                vec![
                    idx000, idx001, idx010, idx011, idx100, idx101, idx110, idx111,
                ],
                0,
            );
            let all_values = flat_data.clone().gather(0, all_indices);

            let v000 = slice_batch(all_values.clone(), 0, batch_size);
            let v001 = slice_batch(all_values.clone(), batch_size, 2 * batch_size);
            let v010 = slice_batch(all_values.clone(), 2 * batch_size, 3 * batch_size);
            let v011 = slice_batch(all_values.clone(), 3 * batch_size, 4 * batch_size);
            let v100 = slice_batch(all_values.clone(), 4 * batch_size, 5 * batch_size);
            let v101 = slice_batch(all_values.clone(), 5 * batch_size, 6 * batch_size);
            let v110 = slice_batch(all_values.clone(), 6 * batch_size, 7 * batch_size);
            let v111 = slice_batch(all_values, 7 * batch_size, 8 * batch_size);

            (v000, v001, v010, v011, v100, v101, v110, v111)
        };

        // Trilinear lerp cascade.
        let one = Tensor::<f32, B>::ones([batch_size], &_device);
        let one_minus_wx = one.clone() - wx.clone();
        let one_minus_wy = one.clone() - wy.clone();
        let one_minus_wz = one - wz.clone();

        let c00 = v000 * one_minus_wx.clone() + v100 * wx.clone();
        let c01 = v001 * one_minus_wx.clone() + v101 * wx.clone();
        let c10 = v010 * one_minus_wx.clone() + v110 * wx.clone();
        let c11 = v011 * one_minus_wx + v111 * wx;

        let c0 = c00 * one_minus_wy.clone() + c10 * wy.clone();
        let c1 = c01 * one_minus_wy.clone() + c11 * wy.clone();

        c0 * one_minus_wz + c1 * wz
    }
);
