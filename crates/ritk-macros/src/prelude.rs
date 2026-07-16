//! Prelude generators: emit coordinate extraction, floor/ceil, weight, and
//! stride bindings for each dimensionality variant.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, Ident};

/// D=1 prelude: extract `x`, floor/ceil, weight, clamped int index.
pub(crate) fn generate_d1_prelude(
    _coords: &[Ident],
    _weights: &[Ident],
    d_max: &[Expr],
) -> TokenStream {
    let d0_max = &d_max[0];
    quote! {
        let shape = data.shape();
        let d0 = shape.dims[0];
        let batch_size = indices.dims()[0];
        let _device = indices.device();

        // Extract coordinate: [N, 1] -> [N]. narrow consumes self.
        let x = indices.narrow(1, 0, 1).squeeze_dims(&[1]);

        // floor consumes self, so clone x0 for weight derivation.
        let x0 = x.clone().floor();
        let wx = x - x0.clone();

        // x0 still owned after weight derivation — clone for x1, then clone for clamp
        // so x0 is consumed by in_bounds_mask (takes by value).
        let x1 = x0.clone() + 1.0;
        let x0_i = x0.clone().clamp(0.0, (#d0_max) as f64).int();
        let x1_i = x1.clamp(0.0, (#d0_max) as f64).int();

        let _ = (&batch_size, &_device);
    }
}

/// D=2 prelude: extract `x, y`, floor/ceil, weights, clamped int indices.
pub(crate) fn generate_d2_prelude(
    _coords: &[Ident],
    _weights: &[Ident],
    d_max: &[Expr],
) -> TokenStream {
    let d1_max = &d_max[0];
    let d0_max = &d_max[1];
    quote! {
        let shape = data.shape();
        let d0 = shape.dims[0]; // Y
        let d1 = shape.dims[1]; // X
        let batch_size = indices.dims()[0];
        let _device = indices.device();

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
        let x0_i = x0.clone().clamp(0.0, (#d1_max) as f64).int();
        let y0_i = y0.clone().clamp(0.0, (#d0_max) as f64).int();
        let x1_i = x1.clamp(0.0, (#d1_max) as f64).int();
        let y1_i = y1.clamp(0.0, (#d0_max) as f64).int();

        let stride_y = d1 as i32;

        let _ = (&batch_size, &_device, &stride_y);
    }
}

/// D=3 prelude: extract `x, y, z`, floor/ceil, weights, clamped int indices, strides.
pub(crate) fn generate_d3_prelude(
    _coords: &[Ident],
    _weights: &[Ident],
    d_max: &[Expr],
) -> TokenStream {
    let d2_max = &d_max[0];
    let d1_max = &d_max[1];
    let d0_max = &d_max[2];
    quote! {
        let shape = data.shape();
        let d0 = shape.dims[0]; // Z
        let d1 = shape.dims[1]; // Y
        let d2 = shape.dims[2]; // X
        let batch_size = indices.dims()[0];
        let _device = indices.device();

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
        let x0_i = x0.clone().clamp(0.0, (#d2_max) as f64).int();
        let y0_i = y0.clone().clamp(0.0, (#d1_max) as f64).int();
        let z0_i = z0.clone().clamp(0.0, (#d0_max) as f64).int();
        let x1_i = x1.clamp(0.0, (#d2_max) as f64).int();
        let y1_i = y1.clamp(0.0, (#d1_max) as f64).int();
        let z1_i = z1.clamp(0.0, (#d0_max) as f64).int();

        let stride_z = (d1 * d2) as i32;
        let stride_y = d2 as i32;

        let _ = (&batch_size, &_device);
    }
}

/// D=4 prelude: extract `x, y, z, w`, floor/ceil, weights, clamped int indices, strides.
pub(crate) fn generate_d4_prelude(
    _coords: &[Ident],
    _weights: &[Ident],
    d_max: &[Expr],
) -> TokenStream {
    let d3_max = &d_max[0];
    let d2_max = &d_max[1];
    let d1_max = &d_max[2];
    let d0_max = &d_max[3];
    quote! {
        let shape = data.shape();
        let d0 = shape.dims[0]; // W
        let d1 = shape.dims[1]; // Z
        let d2 = shape.dims[2]; // Y
        let d3 = shape.dims[3]; // X
        let batch_size = indices.dims()[0];
        let _device = indices.device();

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
        let x0_i = x0.clone().clamp(0.0, (#d3_max) as f64).int();
        let y0_i = y0.clone().clamp(0.0, (#d2_max) as f64).int();
        let z0_i = z0.clone().clamp(0.0, (#d1_max) as f64).int();
        let w0_i = w0.clone().clamp(0.0, (#d0_max) as f64).int();
        let x1_i = x1.clamp(0.0, (#d3_max) as f64).int();
        let y1_i = y1.clamp(0.0, (#d2_max) as f64).int();
        let z1_i = z1.clamp(0.0, (#d1_max) as f64).int();
        let w1_i = w1.clamp(0.0, (#d0_max) as f64).int();

        let stride_w = (d1 * d2 * d3) as i32;
        let stride_z = (d2 * d3) as i32;
        let stride_y = d3 as i32;
        let strides = [stride_y, stride_z, stride_w];

        let _ = (&batch_size, &_device, &strides);
    }
}

// ── Typed linear prelude generators (audit §8 351-01) ─────────────────
//
// The typed prelude is the same as the runtime prelude EXCEPT:
// - `let d0 = shape.dims[0];` is replaced with `let d0: usize = D0;`
// (the const generic becomes a local usize alias)
// - The `let shape = data.shape();` line is omitted
// - The clamp/stride expressions use the const generic identifiers
// directly (e.g. `D2 - 1` instead of `d2 - 1`)
//
// The body can then refer to `d0/d1/d2` (or `d0..d{D-1}`) as before;
// they're just compile-time constants instead of runtime variables.

/// D=1 typed linear prelude: aliases `D0` to `d0` and skips the shape read.
pub(crate) fn generate_typed_d1_prelude(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0];
    quote! {
        let d0: usize = #d0_dim;
        let batch_size = indices.dims()[0];
        let _device = indices.device();

        // Extract coordinate: [N, 1] -> [N]. narrow consumes self.
        let x = indices.narrow(1, 0, 1).squeeze_dims(&[1]);

        // floor consumes self, so clone x0 for weight derivation.
        let x0 = x.clone().floor();
        let wx = x - x0.clone();

        // x0 still owned after weight derivation — clone for x1, then clone for clamp
        // so x0 is consumed by in_bounds_mask (takes by value).
        let x1 = x0.clone() + 1.0;
        let x0_i = x0.clone().clamp(0.0, (#d0_dim - 1) as f64).int();
        let x1_i = x1.clamp(0.0, (#d0_dim - 1) as f64).int();

        let _ = (&batch_size, &_device);
    }
}

/// D=2 typed linear prelude: aliases `D0, D1` to `d0, d1` and skips the shape read.
pub(crate) fn generate_typed_d2_prelude(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0]; // Y
    let d1_dim = &dims[1]; // X
    quote! {
        let d0: usize = #d0_dim; // Y
        let d1: usize = #d1_dim; // X
        let batch_size = indices.dims()[0];
        let _device = indices.device();

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
        let x0_i = x0.clone().clamp(0.0, (#d1_dim - 1) as f64).int();
        let y0_i = y0.clone().clamp(0.0, (#d0_dim - 1) as f64).int();
        let x1_i = x1.clamp(0.0, (#d1_dim - 1) as f64).int();
        let y1_i = y1.clamp(0.0, (#d0_dim - 1) as f64).int();

        let stride_y = d1 as i32;

        let _ = (&batch_size, &_device, &stride_y);
    }
}

/// D=3 typed linear prelude: aliases `D0, D1, D2` to `d0, d1, d2` and skips the shape read.
pub(crate) fn generate_typed_d3_prelude(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0]; // Z
    let d1_dim = &dims[1]; // Y
    let d2_dim = &dims[2]; // X
    quote! {
        let d0: usize = #d0_dim; // Z
        let d1: usize = #d1_dim; // Y
        let d2: usize = #d2_dim; // X
        let batch_size = indices.dims()[0];
        let _device = indices.device();

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
        let x0_i = x0.clone().clamp(0.0, (#d2_dim - 1) as f64).int();
        let y0_i = y0.clone().clamp(0.0, (#d1_dim - 1) as f64).int();
        let z0_i = z0.clone().clamp(0.0, (#d0_dim - 1) as f64).int();
        let x1_i = x1.clamp(0.0, (#d2_dim - 1) as f64).int();
        let y1_i = y1.clamp(0.0, (#d1_dim - 1) as f64).int();
        let z1_i = z1.clamp(0.0, (#d0_dim - 1) as f64).int();

        let stride_z = (d1 * d2) as i32;
        let stride_y = d2 as i32;

        let _ = (&batch_size, &_device);
    }
}

/// D=4 typed linear prelude: aliases `D0..D3` to `d0..d3` and skips the shape read.
pub(crate) fn generate_typed_d4_prelude(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0]; // W
    let d1_dim = &dims[1]; // Z
    let d2_dim = &dims[2]; // Y
    let d3_dim = &dims[3]; // X
    quote! {
        let d0: usize = #d0_dim; // W
        let d1: usize = #d1_dim; // Z
        let d2: usize = #d2_dim; // Y
        let d3: usize = #d3_dim; // X
        let batch_size = indices.dims()[0];
        let _device = indices.device();

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
        let x0_i = x0.clone().clamp(0.0, (#d3_dim - 1) as f64).int();
        let y0_i = y0.clone().clamp(0.0, (#d2_dim - 1) as f64).int();
        let z0_i = z0.clone().clamp(0.0, (#d1_dim - 1) as f64).int();
        let w0_i = w0.clone().clamp(0.0, (#d0_dim - 1) as f64).int();
        let x1_i = x1.clamp(0.0, (#d3_dim - 1) as f64).int();
        let y1_i = y1.clamp(0.0, (#d2_dim - 1) as f64).int();
        let z1_i = z1.clamp(0.0, (#d1_dim - 1) as f64).int();
        let w1_i = w1.clamp(0.0, (#d0_dim - 1) as f64).int();

        let stride_w = (d1 * d2 * d3) as i32;
        let stride_z = (d2 * d3) as i32;
        let stride_y = d3 as i32;
        let strides = [stride_y, stride_z, stride_w];

        let _ = (&batch_size, &_device, &strides);
    }
}

// ── Typed nearest-neighbor prelude generators (Sprint 361) ─────────────
//
// Parallel to the typed linear preludes above, but for nearest-neighbor
// interpolation. The key differences:
//   1. **Rounding**: nearest uses `floor(coord + 0.5)` (round-to-nearest)
//      instead of `floor(coord)` (lower corner) and `ceil(coord)`
//      (upper corner) used by linear.
//   2. **One index per axis**: nearest only needs one int index per axis
//      (no upper/lower pair), so the prelude is simpler.
//   3. **Pre-clamp floor values**: the pre-clamp `x_f`/`y_f`/`z_f`/
//      `w_f` values are bound for use by the nearest mask generators
//      (the mask checks if the *rounded* coordinate is in bounds, not
//      the clamped int index).
//   4. **No weights**: nearest has no `wx`/`wy`/`wz`/`ww` (no lerp cascade).

/// D=1 typed nearest prelude: aliases `D0` to `d0` and binds `x_f`/`x_i`.
pub(crate) fn generate_typed_nearest_d1_prelude(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0];
    quote! {
        let d0: usize = #d0_dim;
        let batch_size = indices.dims()[0];
        let _device = indices.device();

        // narrow consumes self.
        let x = indices.narrow(1, 0, 1).squeeze_dims(&[1]);

        // floor(coord + 0.5) gives standard round-to-nearest behavior.
        // x_f is the pre-clamp floor value (used for in_bounds_mask).
        let x_f = (x + 0.5).floor();
        let x_i = x_f.clone().clamp(0.0, (#d0_dim - 1) as f64).int();

        let _ = (&batch_size, &_device);
    }
}

/// D=2 typed nearest prelude: aliases `D0, D1` to `d0, d1`.
pub(crate) fn generate_typed_nearest_d2_prelude(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0]; // Y
    let d1_dim = &dims[1]; // X
    quote! {
        let d0: usize = #d0_dim; // Y
        let d1: usize = #d1_dim; // X
        let batch_size = indices.dims()[0];
        let _device = indices.device();

        // narrow consumes self, so clone indices once and narrow each column.
        let indices_local = indices;
        let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
        let y = indices_local.narrow(1, 1, 1).squeeze_dims(&[1]);

        // floor(coord + 0.5) gives standard round-to-nearest behavior.
        let x_f = (x + 0.5).floor();
        let y_f = (y + 0.5).floor();
        let x_i = x_f.clone().clamp(0.0, (#d1_dim - 1) as f64).int();
        let y_i = y_f.clone().clamp(0.0, (#d0_dim - 1) as f64).int();

        let stride_y = d1 as i32;

        let _ = (&batch_size, &_device, &stride_y);
    }
}

/// D=3 typed nearest prelude: aliases `D0, D1, D2` to `d0, d1, d2`.
pub(crate) fn generate_typed_nearest_d3_prelude(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0]; // Z
    let d1_dim = &dims[1]; // Y
    let d2_dim = &dims[2]; // X
    quote! {
        let d0: usize = #d0_dim; // Z
        let d1: usize = #d1_dim; // Y
        let d2: usize = #d2_dim; // X
        let batch_size = indices.dims()[0];
        let _device = indices.device();

        // narrow consumes self, so clone indices once and narrow each column.
        let indices_local = indices;
        let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
        let y = indices_local.clone().narrow(1, 1, 1).squeeze_dims(&[1]);
        let z = indices_local.narrow(1, 2, 1).squeeze_dims(&[1]);

        // floor(coord + 0.5) gives standard round-to-nearest behavior.
        let x_f = (x + 0.5).floor();
        let y_f = (y + 0.5).floor();
        let z_f = (z + 0.5).floor();
        let x_i = x_f.clone().clamp(0.0, (#d2_dim - 1) as f64).int();
        let y_i = y_f.clone().clamp(0.0, (#d1_dim - 1) as f64).int();
        let z_i = z_f.clone().clamp(0.0, (#d0_dim - 1) as f64).int();

        let stride_z = (d1 * d2) as i32;
        let stride_y = d2 as i32;

        let _ = (&batch_size, &_device);
    }
}

/// D=4 typed nearest prelude: aliases `D0..D3` to `d0..d3`.
pub(crate) fn generate_typed_nearest_d4_prelude(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0]; // W
    let d1_dim = &dims[1]; // Z
    let d2_dim = &dims[2]; // Y
    let d3_dim = &dims[3]; // X
    quote! {
        let d0: usize = #d0_dim; // W
        let d1: usize = #d1_dim; // Z
        let d2: usize = #d2_dim; // Y
        let d3: usize = #d3_dim; // X
        let batch_size = indices.dims()[0];
        let _device = indices.device();

        // narrow consumes self, so clone indices once and narrow each column.
        let indices_local = indices;
        let x = indices_local.clone().narrow(1, 0, 1).squeeze_dims(&[1]);
        let y = indices_local.clone().narrow(1, 1, 1).squeeze_dims(&[1]);
        let z = indices_local.clone().narrow(1, 2, 1).squeeze_dims(&[1]);
        let w = indices_local.narrow(1, 3, 1).squeeze_dims(&[1]);

        // floor(coord + 0.5) gives standard round-to-nearest behavior.
        let x_f = (x + 0.5).floor();
        let y_f = (y + 0.5).floor();
        let z_f = (z + 0.5).floor();
        let w_f = (w + 0.5).floor();
        let x_i = x_f.clone().clamp(0.0, (#d3_dim - 1) as f64).int();
        let y_i = y_f.clone().clamp(0.0, (#d2_dim - 1) as f64).int();
        let z_i = z_f.clone().clamp(0.0, (#d1_dim - 1) as f64).int();
        let w_i = w_f.clone().clamp(0.0, (#d0_dim - 1) as f64).int();

        let stride_w = (d1 * d2 * d3) as i32;
        let stride_z = (d2 * d3) as i32;
        let stride_y = d3 as i32;
        let strides = [stride_y, stride_z, stride_w];

        let _ = (&batch_size, &_device, &strides);
    }
}
