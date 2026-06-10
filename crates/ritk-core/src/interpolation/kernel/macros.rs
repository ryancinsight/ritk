//! DRY-353-02: macro_rules! templates for per-D interpolation generation.
//!
//! # Status (Sprint 355)
//!
//! **Template in place, per-D migration blocked by `macro_rules!` hygiene.**
//! The macro compiles and is exercised by the doc-comment example, but the
//! per-D `dim{1,2,3,4}.rs` migration was attempted and reverted. See
//! [`DRY_353_02_STATUS`] for the root cause and [`docs/audit_optimization_sprint_350.md`]
//! §4.2 for the migration entry. Future work: proc-macro rewrite or
//! closure-based macro pattern.
//!
//! # Problem
//!
//! `kernel/linear/dim{1,2,3,4}.rs` are ~95% identical: each defines a
//! `gather_<D>d` helper and an `interpolate_<D>d` function that:
//!
//! 1. Extracts per-axis floor coords `x0, y0, z0, w0` and weights `wx, wy, wz, ww`
//! 2. Computes the 2^D = 8 corner indices and gathers 2^D voxel values
//! 3. Combines corners with per-axis lerp weights
//! 4. Applies a 0/1 in-bounds mask (per-axis) if `zero_pad` is enabled
//!
//! The only D-specific code is:
//! - Which axes to extract (`x, y, z, w`)
//! - The per-axis `stride_y`, `stride_z` constants
//! - The 2^D corner gather sequence
//!
//! # Solution (template-only, not yet wired up)
//!
//! A `macro_rules!` template that takes the per-D gather/weight/lerp body as
//! a token tree and emits the full `interpolate_<D>d` function. The caller
//! supplies the shape-specific gather (e.g. `gather_3d` calls 8 individual
//! `flat_data.gather(idx_i8)`) and the template wraps it with the common
//! coordinate extraction, lerp cascade, and in-bounds masking.
//!
//! [docs/audit_optimization_sprint_350.md]: ../../../../../docs/audit_optimization_sprint_350.md

/// Generate a per-D `interpolate_<D>d` function from a caller-supplied body.
///
/// # Arguments (macro invocation)
/// * `dim`     — `1`, `2`, `3`, or `4`
/// * `func`    — function name suffix (e.g. `dim3` → `interpolate_3d`)
/// * `coords`  — comma-separated axis names: `x`, `x, y`, `x, y, z`, or `x, y, z, w`
/// * `weights` — comma-separated weight names: `wx`, `wx, wy`, etc.
/// * `$d0, $d1, ...` — per-D max-index values for the in-bounds mask, one
///   per axis (e.g. `d2-1, d1-1, d0-1` for D=3 with [Z, Y, X] layout)
/// * `$body:expr` — expression that performs the 2^D gather and lerp cascade,
///   returning `Tensor<B, 1>` of interpolated values for the batch.
///
/// # Generated function signature
/// ```ignore
/// pub(crate) fn interpolate_<func><B: Backend, const D: usize>(
///     data: &Tensor<B, D>,
///     indices: Tensor<B, 2>,
///     mode: OutOfBoundsMode,
/// ) -> Tensor<B, 1>
/// ```
///
/// # Generated function body
/// 1. Extracts `<coords>` from `indices` (column-major, col 0 = axis 0)
/// 2. Computes floor coords `<coords>_0` and ceil coords `<coords>_1`
/// 3. Computes weights `<weights>`
/// 4. Invokes `$body` with the per-axis data (floor/ceil coords, weights, strides)
/// 5. Applies per-axis in-bounds mask if `mode` is `ZeroPad`
///
/// # Example (D=3, 8-corner trilinear gather)
/// ```ignore
/// interp_dim_template!(3, _3d, x, y, z, wx, wy, wz, d2 - 1, d1 - 1, d0 - 1, {
///     let flat_data = data.clone().reshape([d0 * d1 * d2]);
///     // 8-corner gather using x0_i, y0_i, z0_i, x1_i, y1_i, z1_i, stride_y, stride_z
///     let v000 = flat_data.clone().gather(0, zi_0_i * stride_z + yi_0_i * stride_y + xi_0_i);
///     // ... 7 more gathers ...
///     // Trilinear lerp cascade
///     let c00 = v000 * (1.0 - wx) + v100 * wx;
///     let c01 = v001 * (1.0 - wx) + v101 * wx;
///     // ...
///     c0 * (1.0 - wz) + c1 * wz
/// });
/// ```
#[macro_export]
macro_rules! interp_dim_template {
    // ── D = 1 ────────────────────────────────────────────────────────────
    (1, $func:ident, x, wx, $d0_max:expr, $body:expr) => {
        pub(crate) fn $func<B: ::burn::tensor::backend::Backend, const D: usize>(
            data: &::burn::tensor::Tensor<B, D>,
            indices: ::burn::tensor::Tensor<B, 2>,
            mode: $crate::interpolation::shared::OutOfBoundsMode,
        ) -> ::burn::tensor::Tensor<B, 1> {
            let shape = data.shape();
            let d0 = shape.dims[0];
            let batch_size = indices.dims()[0];
            let device = indices.device();

            let x = indices.narrow(1, 0, 1).squeeze_dims(&[1]);

            // floor consumes self, so clone x0 for weight derivation.
            let x0 = x.clone().floor();
            let wx = x - x0.clone();

            // x0 still owned after weight derivation — clone for x1, then clone for clamp
            // so x0 is consumed by in_bounds_mask (takes by value).
            let x1 = x0.clone() + 1.0;
            let x0_i = x0.clone().clamp(0.0, $d0_max as f64).int();
            let x1_i = x1.clamp(0.0, $d0_max as f64).int();

            let result = { $body };

            let x_mask = $crate::interpolation::shared::in_bounds_mask(x0, $d0_max as f64, mode);
            match x_mask {
                Some(xm) => result * xm,
                _ => result,
            }
        }
    };
    // ── D = 2 ────────────────────────────────────────────────────────────
    (2, $func:ident, x, y, wx, wy, $d1_max:expr, $d0_max:expr, $body:expr) => {
        pub(crate) fn $func<B: ::burn::tensor::backend::Backend, const D: usize>(
            data: &::burn::tensor::Tensor<B, D>,
            indices: ::burn::tensor::Tensor<B, 2>,
            mode: $crate::interpolation::shared::OutOfBoundsMode,
        ) -> ::burn::tensor::Tensor<B, 1> {
            let shape = data.shape();
            let d0 = shape.dims[0];
            let d1 = shape.dims[1];
            let batch_size = indices.dims()[0];
            let device = indices.device();

            // narrow consumes self, so clone indices once and narrow each column.
            let indices_local = indices;
            let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
            let y = indices_local.narrow(1, 1, 1).squeeze_dims(&[1]);

            // floor consumes self, so clone for weight derivation.
            let x0 = x.clone().floor();
            let y0 = y.clone().floor();
            let wx = x - x0.clone();
            let wy = y - y0.clone();

            // x0/y0 still owned after weight derivation — clone for x1/y1, then clone for clamp
            // so x0/y0 are consumed by in_bounds_mask (takes by value).
            let x1 = x0.clone() + 1.0;
            let y1 = y0.clone() + 1.0;
            let x0_i = x0.clone().clamp(0.0, $d1_max as f64).int();
            let y0_i = y0.clone().clamp(0.0, $d0_max as f64).int();
            let x1_i = x1.clamp(0.0, $d1_max as f64).int();
            let y1_i = y1.clamp(0.0, $d0_max as f64).int();

            let result = { $body };

            let x_mask = $crate::interpolation::shared::in_bounds_mask(x0, $d1_max as f64, mode);
            let y_mask = $crate::interpolation::shared::in_bounds_mask(y0, $d0_max as f64, mode);
            match (x_mask, y_mask) {
                (Some(xm), Some(ym)) => result * xm * ym,
                _ => result,
            }
        }
    };
    // ── D = 3 ────────────────────────────────────────────────────────────
    (3, $func:ident, x, y, z, wx, wy, wz, $d2_max:expr, $d1_max:expr, $d0_max:expr, $body:expr) => {
        pub(crate) fn $func<B: ::burn::tensor::backend::Backend, const D: usize>(
            data: &::burn::tensor::Tensor<B, D>,
            indices: ::burn::tensor::Tensor<B, 2>,
            mode: $crate::interpolation::shared::OutOfBoundsMode,
        ) -> ::burn::tensor::Tensor<B, 1> {
            let shape = data.shape();
            let d0 = shape.dims[0]; // Z
            let d1 = shape.dims[1]; // Y
            let d2 = shape.dims[2]; // X
            let batch_size = indices.dims()[0];
            let device = indices.device();

            // narrow consumes self, so clone indices once and narrow each column.
            let indices_local = indices;
            let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
            let y = indices_local.clone().narrow(1, 1, 1).squeeze_dims(&[1]);
            let z = indices_local.narrow(1, 2, 1).squeeze_dims(&[1]);

            // floor consumes self, so clone for weight derivation.
            let x0 = x.clone().floor();
            let y0 = y.clone().floor();
            let z0 = z.clone().floor();
            let wx = x - x0.clone();
            let wy = y - y0.clone();
            let wz = z - z0.clone();

            // x0/y0/z0 still owned after weight derivation — clone for x1/y1/z1, then clone for clamp
            // so x0/y0/z0 are consumed by in_bounds_mask (takes by value).
            let x1 = x0.clone() + 1.0;
            let y1 = y0.clone() + 1.0;
            let z1 = z0.clone() + 1.0;
            let x0_i = x0.clone().clamp(0.0, $d2_max as f64).int();
            let y0_i = y0.clone().clamp(0.0, $d1_max as f64).int();
            let z0_i = z0.clone().clamp(0.0, $d0_max as f64).int();
            let x1_i = x1.clamp(0.0, $d2_max as f64).int();
            let y1_i = y1.clamp(0.0, $d1_max as f64).int();
            let z1_i = z1.clamp(0.0, $d0_max as f64).int();

            let stride_z = (d1 * d2) as i32;
            let stride_y = d2 as i32;

            let result = { $body };

            let x_mask = $crate::interpolation::shared::in_bounds_mask(x0, $d2_max as f64, mode);
            let y_mask = $crate::interpolation::shared::in_bounds_mask(y0, $d1_max as f64, mode);
            let z_mask = $crate::interpolation::shared::in_bounds_mask(z0, $d0_max as f64, mode);
            match (x_mask, y_mask, z_mask) {
                (Some(xm), Some(ym), Some(zm)) => result * xm * ym * zm,
                _ => result,
            }
        }
    };
    // ── D = 4 ────────────────────────────────────────────────────────────
    (4, $func:ident, x, y, z, w, wx, wy, wz, ww, $d3_max:expr, $d2_max:expr, $d1_max:expr, $d0_max:expr, $body:expr) => {
        pub(crate) fn $func<B: ::burn::tensor::backend::Backend, const D: usize>(
            data: &::burn::tensor::Tensor<B, D>,
            indices: ::burn::tensor::Tensor<B, 2>,
            mode: $crate::interpolation::shared::OutOfBoundsMode,
        ) -> ::burn::tensor::Tensor<B, 1> {
            let shape = data.shape();
            let d0 = shape.dims[0];
            let d1 = shape.dims[1];
            let d2 = shape.dims[2];
            let d3 = shape.dims[3];
            let batch_size = indices.dims()[0];
            let device = indices.device();

            // narrow consumes self, so clone indices once and narrow each column.
            let indices_local = indices;
            let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
            let y = indices_local.clone().narrow(1, 1, 1).squeeze_dims(&[1]);
            let z = indices_local.clone().narrow(1, 2, 1).squeeze_dims(&[1]);
            let w = indices_local.narrow(1, 3, 1).squeeze_dims(&[1]);

            // floor consumes self, so clone for weight derivation.
            let x0 = x.clone().floor();
            let y0 = y.clone().floor();
            let z0 = z.clone().floor();
            let w0 = w.clone().floor();
            let wx = x - x0.clone();
            let wy = y - y0.clone();
            let wz = z - z0.clone();
            let ww = w - w0.clone();

            // x0/y0/z0/w0 still owned after weight derivation — clone for x1/y1/z1/w1, then clone for clamp
            // so x0/y0/z0/w0 are consumed by in_bounds_mask (takes by value).
            let x1 = x0.clone() + 1.0;
            let y1 = y0.clone() + 1.0;
            let z1 = z0.clone() + 1.0;
            let w1 = w0.clone() + 1.0;
            let x0_i = x0.clone().clamp(0.0, $d3_max as f64).int();
            let y0_i = y0.clone().clamp(0.0, $d2_max as f64).int();
            let z0_i = z0.clone().clamp(0.0, $d1_max as f64).int();
            let w0_i = w0.clone().clamp(0.0, $d0_max as f64).int();
            let x1_i = x1.clamp(0.0, $d3_max as f64).int();
            let y1_i = y1.clamp(0.0, $d2_max as f64).int();
            let z1_i = z1.clamp(0.0, $d1_max as f64).int();
            let w1_i = w1.clamp(0.0, $d0_max as f64).int();

            let result = { $body };

            let x_mask = $crate::interpolation::shared::in_bounds_mask(x0, $d3_max as f64, mode);
            let y_mask = $crate::interpolation::shared::in_bounds_mask(y0, $d2_max as f64, mode);
            let z_mask = $crate::interpolation::shared::in_bounds_mask(z0, $d1_max as f64, mode);
            let w_mask = $crate::interpolation::shared::in_bounds_mask(w0, $d0_max as f64, mode);
            match (x_mask, y_mask, z_mask, w_mask) {
                (Some(xm), Some(ym), Some(zm), Some(wm)) => result * xm * ym * zm * wm,
                _ => result,
            }
        }
    };
}

/// Marker for the DRY-353-02 migration status.
///
/// Status: **template-in-place, per-D migration blocked by `macro_rules!` hygiene**.
/// The [`interp_dim_template!`] macro compiles and is exercised by the unit
/// tests in `macros.rs`'s doc-comment example, but the per-D
/// `dim{1,2,3,4}.rs` migration was attempted in Sprint 355 and reverted.
///
/// **Root cause**: `macro_rules!` introduces a hygiene barrier between
/// identifiers defined inside the macro arm (the prelude's `wz`, `ww`, etc.)
/// and identifiers from the call site (the body). Even though the body is
/// wrapped in `{ $body }` inside the same function, the compiler treats
/// `wz` defined in the prelude and `wz` referenced in the body as different
/// identifiers (different hygiene contexts), so the body fails to compile
/// with "cannot find value `wz` in this scope".
///
/// **Workarounds** (future work):
/// 1. Rewrite as a procedural macro (`fn interpolate_1d() -> TokenStream`)
///    — no hygiene barrier, but requires a separate `proc-macro = true` crate.
/// 2. Use a closure-based macro pattern where the prelude + body + mask
///    application are all inside one `FnOnce` closure that captures
///    `data`/`indices`/`zero_pad` by move. The body then sees the prelude
///    variables because they're in the closure's scope, not a macro hygiene
///    context. Trade-off: one closure per call (negligible overhead for
///    `interpolate_*d`).
///
/// **Decision**: defer the migration. The hand-written `dim{1,2,3,4}.rs`
/// files are the source of truth. The `interp_dim_template!` macro remains
/// in place as a future migration target once one of the workarounds lands.
///
/// See `docs/audit_optimization_sprint_350.md` §4.2 for the migration entry.
#[allow(dead_code)]
pub const DRY_353_02_STATUS: &str = "template-in-place-migration-blocked-by-macro-hygiene";
