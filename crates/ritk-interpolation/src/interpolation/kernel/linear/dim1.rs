//! 1-D linear interpolation kernel (Sprint 356).
//!
//! Generated via the [`ritk_macros::interp_dim_template!`] proc-macro
//! which resolves the `macro_rules!` hygiene barrier that blocked the
//! original `interp_dim_template!` template (see
//! `crates/ritk-core/src/interpolation/kernel/macros.rs` and
//! `docs/audit_optimization_sprint_350.md` §4.2.2 / §7.7).
//!
//! The proc-macro generates the prelude (coord extraction, floor/ceil,
//! weights, clamped int indices) and the in-bounds mask application;
//! this file supplies only the gather + lerp cascade body.

use burn::tensor::{backend::Backend, Int, Tensor};

/// 1-D gather with a borrowed index — used by the autodiff path where
/// the caller must retain the index tensor for further use.
#[inline]
fn gather_1d<B: Backend>(flat_data: &Tensor<B, 1>, idx: &Tensor<B, 1, Int>) -> Tensor<B, 1> {
    flat_data.clone().gather(0, idx.clone())
}

/// 1-D gather consuming the index tensor — used by the non-autodiff
/// fast path. Eliminates the `idx.clone()` per call.
#[inline]
fn gather_1d_owned<B: Backend>(flat_data: &Tensor<B, 1>, idx: Tensor<B, 1, Int>) -> Tensor<B, 1> {
    flat_data.clone().gather(0, idx)
}

ritk_macros::interp_dim_template!(1, interpolate_1d, x, wx, d0 - 1, {
    // ── Const-generic dispatch (Sprint 355) ────────────────────────
    // The `if B::ad_enabled()` branch is monomorphized per backend
    // and dead-code-eliminated at compile time.
    let (v0, v1) = if B::ad_enabled() {
        (
            gather_1d(&data.clone().reshape([d0]), &x0_i),
            gather_1d(&data.clone().reshape([d0]), &x1_i),
        )
    } else {
        (
            gather_1d_owned(&data.clone().reshape([d0]), x0_i),
            gather_1d_owned(&data.clone().reshape([d0]), x1_i),
        )
    };

    // Linear interpolation.
    let one = Tensor::<B, 1>::ones([batch_size], &_device);
    let one_minus_wx = one - wx.clone();

    v0 * one_minus_wx + v1 * wx
});

// ════════════════════════════════════════════════════════════════════════
//  Const-generic shape specialization (audit §8 351-01)
// ════════════════════════════════════════════════════════════════════════
//
// Parallel to `interpolate_1d` above, but takes the volume size as a
// const generic `const D0: usize`. This enables:
//   1. **Compile-time bounds**: `clamp(0.0, (D0 - 1) as f64)` is a
//      constant expression — no runtime arithmetic.
//   2. **Mask inlining**: `in_bounds_mask(x0, (D0 - 1) as f64, mode)`
//      gets a compile-time-known max — the compiler can inline and
//      eliminate the function call for `OutOfBoundsMode::Clamp`.
//   3. **No `data.shape()` read**: saves 1 memory load per call.
//   4. **Monomorphization**: each `D0` value is a separate monomorphized
//      function.
ritk_macros::interp_dim_template_typed!(1, interpolate_1d_typed, x, wx, D0 - 1, D0, {
    // ── Const-generic dispatch (Sprint 355) ────────────────────────
    let (v0, v1) = if B::ad_enabled() {
        (
            gather_1d(&data.clone().reshape([d0]), &x0_i),
            gather_1d(&data.clone().reshape([d0]), &x1_i),
        )
    } else {
        (
            gather_1d_owned(&data.clone().reshape([d0]), x0_i),
            gather_1d_owned(&data.clone().reshape([d0]), x1_i),
        )
    };

    // Linear interpolation.
    let one = Tensor::<B, 1>::ones([batch_size], &_device);
    let one_minus_wx = one - wx.clone();

    v0 * one_minus_wx + v1 * wx
});
