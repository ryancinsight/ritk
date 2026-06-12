//! Procedural macros for `ritk-core`.
//!
//! Currently provides [`interp_dim_template!`], which generates
//! `interpolate_<D>d` functions for the linear interpolation kernels
//! (`crates/ritk-core/src/interpolation/kernel/linear/dim{1,2,3,4}.rs`).
//!
//! # Why a proc-macro?
//!
//! The `macro_rules!` version of this template
//! (`crates/ritk-core/src/interpolation/kernel/macros.rs`) hit a
//! fundamental `macro_rules!` hygiene barrier: variables defined in the
//! macro arm's prelude (e.g. `x0_i`, `wz`) and identifiers referenced
//! from the call-site body were treated as different hygiene contexts,
//! so the body failed to compile with "cannot find value `wz` in this
//! scope". A proc-macro has no such barrier — the body tokens are
//! spliced directly into the generated function's scope, so the body's
//! references to prelude variables resolve normally.
//!
//! See `docs/audit_optimization_sprint_350.md` §4.2.2 / `DRY_353_02_STATUS`
//! for the full background.

mod mask;
mod parse;
mod prelude;

use mask::*;
use parse::{InterpDimInput, InterpDimTypedInput};
use prelude::*;
use proc_macro2::TokenStream;
use quote::quote;

/// Generate a per-D `interpolate_<D>d` function from a caller-supplied body.
///
/// # Arguments
/// * `dim` — `1`, `2`, `3`, or `4`
/// * `func` — function name (e.g. `interpolate_3d`)
/// * `coords` — comma-separated axis names: `x`, `x, y`, `x, y, z`, or `x, y, z, w`
/// * `weights` — comma-separated weight names: `wx`, `wx, wy`, etc.
/// * `d_max` — per-axis max-index expressions for the in-bounds mask
///   (e.g. `d2 - 1, d1 - 1, d0 - 1` for D=3 with [Z, Y, X] layout)
/// * `body` — token stream that performs the 2^D gather and lerp
///   cascade, and binds the result to `result: Tensor<B, 1>`.
///
/// # Generated function
/// ```ignore
/// pub(crate) fn <func><B: Backend, const D: usize>(
/// data: &Tensor<B, D>,
/// indices: Tensor<B, 2>,
/// mode: OutOfBoundsMode,
/// ) -> Tensor<B, 1> { /* prelude + body + mask */ }
/// ```
#[proc_macro]
pub fn interp_dim_template(input: ::proc_macro::TokenStream) -> ::proc_macro::TokenStream {
    let input = proc_macro2::TokenStream::from(input);
    let output = interp_dim_template_inner(input);
    ::proc_macro::TokenStream::from(output)
}

fn interp_dim_template_inner(input: TokenStream) -> TokenStream {
    let parsed = match syn::parse2::<InterpDimInput>(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error(),
    };

    let dim_val = match parsed.dim.base10_parse::<usize>() {
        Ok(n) => n,
        Err(e) => return e.to_compile_error(),
    };

    let func = &parsed.func;
    let body = &parsed.body;
    let d_max = &parsed.d_max;
    let coords = &parsed.coords;
    let weights = &parsed.weights;

    // Validate counts match dim.
    let n = dim_val;
    if coords.len() != n || weights.len() != n || d_max.len() != n {
        let err = syn::Error::new_spanned(
            &parsed.dim,
            format!(
                "dim {} requires {} coords, {} weights, {} d_max values (got {}, {}, {})",
                n,
                n,
                n,
                n,
                coords.len(),
                weights.len(),
                d_max.len()
            ),
        );
        return err.to_compile_error();
    }

    // Build the prelude (coord extraction, floor/ceil, weights, clamped int
    // indices, strides) and the mask application based on dim.
    let (prelude, mask) = match dim_val {
        1 => (
            generate_d1_prelude(coords, weights, d_max),
            generate_d1_mask(coords, d_max),
        ),
        2 => (
            generate_d2_prelude(coords, weights, d_max),
            generate_d2_mask(coords, d_max),
        ),
        3 => (
            generate_d3_prelude(coords, weights, d_max),
            generate_d3_mask(coords, d_max),
        ),
        4 => (
            generate_d4_prelude(coords, weights, d_max),
            generate_d4_mask(coords, d_max),
        ),
        _ => {
            let err = syn::Error::new_spanned(
                &parsed.dim,
                format!("dim must be 1, 2, 3, or 4 (got {})", dim_val),
            );
            return err.to_compile_error();
        }
    };

    let output = quote! {
        pub(crate) fn #func<B: ::burn::tensor::backend::Backend, const D: usize>(
            data: &::burn::tensor::Tensor<B, D>,
            indices: ::burn::tensor::Tensor<B, 2>,
            mode: crate::interpolation::shared::OutOfBoundsMode,
        ) -> ::burn::tensor::Tensor<B, 1> {
            #prelude
            let result = { #body };
            #mask
        }
    };
    output
}

// ════════════════════════════════════════════════════════════════════════
// Const-generic shape specialization layer (audit §8 351-01)
// ════════════════════════════════════════════════════════════════════════
//
// The runtime-shape macro above reads `data.shape().dims[i]` for the
// volume bounds. The typed variant takes the shape as **const generics**
// (e.g. `const D0: usize, const D1: usize, const D2: usize`), enabling:
//
// 1. **Compile-time bounds**: the `clamp(0.0, (D2 - 1) as f64)` calls
//    become `(D2 - 1) as f64` constants — no runtime arithmetic.
// 2. **Mask inlining**: `in_bounds_mask(x0, (D2 - 1) as f64, mode)`
//    gets a compile-time-known max — the compiler can inline and
//    eliminate the function call for `OutOfBoundsMode::Extend`.
// 3. **No `data.shape()` read**: saves 3 memory loads per call (3×
//    `usize` reads from the shape metadata buffer).
// 4. **Monomorphization**: each `(D0, D1, D2)` triple is a separate
//    monomorphized function — the compiler can fully unroll the
//    8-corner gather cascade for the specific shape.
//
// Trade-off: the caller must know the shape at compile time. This is
// intended for hot paths where the volume size is fixed (e.g. a
// registration pipeline that loads a fixed-size volume, or a benchmark
// with a fixed test image). For dynamic-shape callers, the existing
// `interp_dim_template!` macro is the right choice.
//
// # Design
//
// Input format (parallel to the runtime macro):
// ```text
// interp_dim_template_typed!(
// <dim: int>, // 1, 2, 3, or 4
// <func: ident>, // function name (e.g. interpolate_3d_typed)
// <coords: idents>,>, // x / x, y / x, y, z / x, y, z, w
// <weights: idents>,>, // wx / wx, wy / wx, wy, wz / wx, wy, wz, ww
// <d_max: exprs>,>, // D0-1 / D1-1, D0-1 / D2-1, D1-1, D0-1 / D3-1, D2-1, D1-1, D0-1
// <dims: idents>,>, // D0 / D0, D1 / D0, D1, D2 / D0, D1, D2, D3
// { <body: tokens> } // gather + lerp cascade; uses d0/d1/d2 as aliases for D0/D1/D2
// );
// ```
//
// The `dims` field is a list of const generic identifiers (one per axis,
// in `[Z, Y, X]` order to match the runtime convention). The generated
// function takes these as `const DIMS: usize` parameters and aliases
// them to `d0/d1/d2` so the body can use the same names as the runtime
// version.

/// Generate a per-D `interpolate_<D>d_typed` function with **const-generic
/// shape** (audit §8 351-01).
///
/// The generated function takes the shape as `const D0: usize, const D1:
/// usize, ...` parameters and uses them as compile-time bounds for the
/// gather indices and in-bounds mask. See the module-level doc comment
/// for the full design rationale.
#[proc_macro]
pub fn interp_dim_template_typed(input: ::proc_macro::TokenStream) -> ::proc_macro::TokenStream {
    let input = proc_macro2::TokenStream::from(input);
    let output = interp_dim_template_typed_inner(input);
    ::proc_macro::TokenStream::from(output)
}

fn interp_dim_template_typed_inner(input: TokenStream) -> TokenStream {
    let parsed = match syn::parse2::<InterpDimTypedInput>(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error(),
    };

    let dim_val = match parsed.dim.base10_parse::<usize>() {
        Ok(n) => n,
        Err(e) => return e.to_compile_error(),
    };

    let func = &parsed.func;
    let body = &parsed.body;
    let d_max = &parsed.d_max;
    let coords = &parsed.coords;
    let weights = &parsed.weights;
    let dims = &parsed.dims;

    // Validate counts match dim.
    let n = dim_val;
    if coords.len() != n || weights.len() != n || d_max.len() != n || dims.len() != n {
        let err = syn::Error::new_spanned(
            &parsed.dim,
            format!(
                "dim {} requires {} coords, {} weights, {} d_max, {} dims (got {}, {}, {}, {})",
                n,
                n,
                n,
                n,
                n,
                coords.len(),
                weights.len(),
                d_max.len(),
                dims.len()
            ),
        );
        return err.to_compile_error();
    }

    // Build the const-generic function signature: `<B, const D0: usize, const D1: usize, ...>`.
    // The `dims` list is in `[Z, Y, X]` order to match the runtime convention
    // (d0 = outermost axis, d_{D-1} = innermost). The aliases `d0, d1, ...`
    // in the prelude use the same ordering, so the body can refer to `d0/d1/...`
    // as if they were runtime usize values.
    let const_generics = dims.iter().map(|d| {
        quote! { const #d: usize }
    });

    // Build the prelude that aliases the const generics to `d0/d1/...`
    // and the mask that uses the const generics for bounds.
    let (prelude, mask) = match dim_val {
        1 => (
            generate_typed_d1_prelude(dims),
            generate_d1_mask(coords, d_max),
        ),
        2 => (
            generate_typed_d2_prelude(dims),
            generate_d2_mask(coords, d_max),
        ),
        3 => (
            generate_typed_d3_prelude(dims),
            generate_d3_mask(coords, d_max),
        ),
        4 => (
            generate_typed_d4_prelude(dims),
            generate_d4_mask(coords, d_max),
        ),
        _ => {
            let err = syn::Error::new_spanned(
                &parsed.dim,
                format!("dim must be 1, 2, 3, or 4 (got {})", dim_val),
            );
            return err.to_compile_error();
        }
    };

    // The function signature is specialized to the rank (D = dim_val),
    // not generic over it — the body assumes the rank matches the
    // const-generic shape length. The const generics parameterize the
    // shape, not the rank.
    let output = quote! {
        pub(crate) fn #func<
            B: ::burn::tensor::backend::Backend,
            #( #const_generics ),*
        >(
            data: &::burn::tensor::Tensor<B, #dim_val>,
            indices: ::burn::tensor::Tensor<B, 2>,
            mode: crate::interpolation::shared::OutOfBoundsMode,
        ) -> ::burn::tensor::Tensor<B, 1> {
            #prelude
            let result = { #body };
            #mask
        }
    };
    output
}

// ══════════════════════════════════════════════════════════════════════
// Typed nearest-neighbor specialization layer (Sprint 361)
// ══════════════════════════════════════════════════════════════════════
//
// Parallel to [`interp_dim_template_typed!`] but for nearest-neighbor
// interpolation. The key differences from the linear typed macro:
//   1. **Rounding**: uses `floor(coord + 0.5)` (round-to-nearest)
//      instead of `floor(coord)` + `ceil(coord)` (lower/upper corners).
//   2. **One index per axis**: nearest only needs one int index per
//      axis (no upper/lower pair), so the body is a single gather.
//   3. **Pre-clamp floor values**: the `x_f`/`y_f`/`z_f`/`w_f` values
//      are bound for use by the nearest mask (the mask checks if the
//      *rounded* coordinate is in bounds, not the clamped int index).
//   4. **No weights**: nearest has no `wx`/`wy`/`wz`/`ww` (no lerp
//      cascade).
//
// The input format is identical to [`interp_dim_template_typed!`], so
// the same `InterpDimTypedInput` parser is reused.

/// Generate a per-D `interpolate_nearest_<D>d_typed` function with
/// **const-generic shape** (Sprint 361 — 351-01-NN-TYPED).
///
/// Parallel to [`interp_dim_template_typed!`] but for nearest-neighbor
/// interpolation. See the module-level doc comment for the full design
/// rationale.
#[proc_macro]
pub fn interp_dim_template_nearest_typed(
    input: ::proc_macro::TokenStream,
) -> ::proc_macro::TokenStream {
    let input = proc_macro2::TokenStream::from(input);
    let output = interp_dim_template_nearest_typed_inner(input);
    ::proc_macro::TokenStream::from(output)
}

fn interp_dim_template_nearest_typed_inner(input: TokenStream) -> TokenStream {
    let parsed = match syn::parse2::<InterpDimTypedInput>(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error(),
    };

    let dim_val = match parsed.dim.base10_parse::<usize>() {
        Ok(n) => n,
        Err(e) => return e.to_compile_error(),
    };

    let func = &parsed.func;
    let body = &parsed.body;
    let dims = &parsed.dims;

    // Validate counts match dim.
    let n = dim_val;
    if dims.len() != n {
        let err = syn::Error::new_spanned(
            &parsed.dim,
            format!("dim {} requires {} dims (got {})", n, n, dims.len()),
        );
        return err.to_compile_error();
    }

    // Build the const-generic function signature.
    let const_generics = dims.iter().map(|d| {
        quote! { const #d: usize }
    });

    // Build the nearest-neighbor prelude (rounding, clamping) and mask
    // (using pre-clamp floor values for in_bounds_mask).
    let (prelude, mask) = match dim_val {
        1 => (
            generate_typed_nearest_d1_prelude(dims),
            generate_nearest_d1_mask(dims),
        ),
        2 => (
            generate_typed_nearest_d2_prelude(dims),
            generate_nearest_d2_mask(dims),
        ),
        3 => (
            generate_typed_nearest_d3_prelude(dims),
            generate_nearest_d3_mask(dims),
        ),
        4 => (
            generate_typed_nearest_d4_prelude(dims),
            generate_nearest_d4_mask(dims),
        ),
        _ => {
            let err = syn::Error::new_spanned(
                &parsed.dim,
                format!("dim must be 1, 2, 3, or 4 (got {})", dim_val),
            );
            return err.to_compile_error();
        }
    };

    // The function signature is specialized to the rank (D = dim_val),
    // not generic over it — the body assumes the rank matches the
    // const-generic shape length. The const generics parameterize the
    // shape, not the rank.
    let output = quote! {
        pub(crate) fn #func<
            B: ::burn::tensor::backend::Backend,
            #( #const_generics ),*
        >(
            data: &::burn::tensor::Tensor<B, #dim_val>,
            indices: ::burn::tensor::Tensor<B, 2>,
            mode: crate::interpolation::shared::OutOfBoundsMode,
        ) -> ::burn::tensor::Tensor<B, 1> {
            #prelude
            let result = { #body };
            #mask
        }
    };
    output
}
