//! Mask generators: emit in-bounds masking logic for each dimensionality variant.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, Ident};

pub(crate) fn generate_d1_mask(coords: &[Ident], d_max: &[Expr]) -> TokenStream {
    let d0_max = &d_max[0];
    let x = &coords[0];
    let x0 = Ident::new(&format!("{}0", x), x.span());
    quote! {
        let x_mask = crate::interpolation::shared::in_bounds_mask(#x0, (#d0_max) as f64, mode);
        match x_mask {
            Some(xm) => result * xm,
            _ => result,
        }
    }
}

pub(crate) fn generate_d2_mask(coords: &[Ident], d_max: &[Expr]) -> TokenStream {
    let d1_max = &d_max[0];
    let d0_max = &d_max[1];
    let x = &coords[0];
    let y = &coords[1];
    let x0 = Ident::new(&format!("{}0", x), x.span());
    let y0 = Ident::new(&format!("{}0", y), y.span());
    quote! {
        let x_mask = crate::interpolation::shared::in_bounds_mask(#x0, (#d1_max) as f64, mode);
        let y_mask = crate::interpolation::shared::in_bounds_mask(#y0, (#d0_max) as f64, mode);
        match (x_mask, y_mask) {
            (Some(xm), Some(ym)) => result * xm * ym,
            _ => result,
        }
    }
}

pub(crate) fn generate_d3_mask(coords: &[Ident], d_max: &[Expr]) -> TokenStream {
    let d2_max = &d_max[0];
    let d1_max = &d_max[1];
    let d0_max = &d_max[2];
    let x = &coords[0];
    let y = &coords[1];
    let z = &coords[2];
    let x0 = Ident::new(&format!("{}0", x), x.span());
    let y0 = Ident::new(&format!("{}0", y), y.span());
    let z0 = Ident::new(&format!("{}0", z), z.span());
    quote! {
        let x_mask = crate::interpolation::shared::in_bounds_mask(#x0, (#d2_max) as f64, mode);
        let y_mask = crate::interpolation::shared::in_bounds_mask(#y0, (#d1_max) as f64, mode);
        let z_mask = crate::interpolation::shared::in_bounds_mask(#z0, (#d0_max) as f64, mode);
        match (x_mask, y_mask, z_mask) {
            (Some(xm), Some(ym), Some(zm)) => result * xm * ym * zm,
            _ => result,
        }
    }
}

// ── Nearest-neighbor mask generators (Sprint 361) ────────────────────
//
// Parallel to `generate_dN_mask` above, but for nearest-neighbor
// interpolation. The key difference: nearest masks use the
// pre-clamp `x_f`/`y_f`/`z_f`/`w_f` values (the `floor(coord + 0.5)`
// rounding result) instead of the linear `x0`/`y0`/`z0`/`w0` (the
// `floor(coord)` result). The mask checks if the *rounded* coordinate
// is in bounds, not the clamped int index.

/// D=1 nearest mask: use `x_f` (pre-clamp floor) for in_bounds_mask.
pub(crate) fn generate_nearest_d1_mask(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0];
    quote! {
        let x_mask = crate::interpolation::shared::in_bounds_mask(
            x_f,
            (#d0_dim - 1) as f64,
            mode,
        );
        match x_mask {
            Some(xm) => result * xm,
            _ => result,
        }
    }
}

/// D=2 nearest mask: use `x_f`, `y_f` for in_bounds_mask.
pub(crate) fn generate_nearest_d2_mask(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0]; // Y
    let d1_dim = &dims[1]; // X
    quote! {
        let x_mask = crate::interpolation::shared::in_bounds_mask(
            x_f,
            (#d1_dim - 1) as f64,
            mode,
        );
        let y_mask = crate::interpolation::shared::in_bounds_mask(
            y_f,
            (#d0_dim - 1) as f64,
            mode,
        );
        match (x_mask, y_mask) {
            (Some(xm), Some(ym)) => result * xm * ym,
            _ => result,
        }
    }
}

/// D=3 nearest mask: use `x_f`, `y_f`, `z_f` for in_bounds_mask.
pub(crate) fn generate_nearest_d3_mask(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0]; // Z
    let d1_dim = &dims[1]; // Y
    let d2_dim = &dims[2]; // X
    quote! {
        let x_mask = crate::interpolation::shared::in_bounds_mask(
            x_f,
            (#d2_dim - 1) as f64,
            mode,
        );
        let y_mask = crate::interpolation::shared::in_bounds_mask(
            y_f,
            (#d1_dim - 1) as f64,
            mode,
        );
        let z_mask = crate::interpolation::shared::in_bounds_mask(
            z_f,
            (#d0_dim - 1) as f64,
            mode,
        );
        match (x_mask, y_mask, z_mask) {
            (Some(xm), Some(ym), Some(zm)) => result * xm * ym * zm,
            _ => result,
        }
    }
}

/// D=4 nearest mask: use `x_f`, `y_f`, `z_f`, `w_f` for in_bounds_mask.
pub(crate) fn generate_nearest_d4_mask(dims: &[Ident]) -> TokenStream {
    let d0_dim = &dims[0]; // W
    let d1_dim = &dims[1]; // Z
    let d2_dim = &dims[2]; // Y
    let d3_dim = &dims[3]; // X
    quote! {
        let x_mask = crate::interpolation::shared::in_bounds_mask(
            x_f,
            (#d3_dim - 1) as f64,
            mode,
        );
        let y_mask = crate::interpolation::shared::in_bounds_mask(
            y_f,
            (#d2_dim - 1) as f64,
            mode,
        );
        let z_mask = crate::interpolation::shared::in_bounds_mask(
            z_f,
            (#d1_dim - 1) as f64,
            mode,
        );
        let w_mask = crate::interpolation::shared::in_bounds_mask(
            w_f,
            (#d0_dim - 1) as f64,
            mode,
        );
        match (x_mask, y_mask, z_mask, w_mask) {
            (Some(xm), Some(ym), Some(zm), Some(wm)) => result * xm * ym * zm * wm,
            _ => result,
        }
    }
}

pub(crate) fn generate_d4_mask(coords: &[Ident], d_max: &[Expr]) -> TokenStream {
    let d3_max = &d_max[0];
    let d2_max = &d_max[1];
    let d1_max = &d_max[2];
    let d0_max = &d_max[3];
    let x = &coords[0];
    let y = &coords[1];
    let z = &coords[2];
    let w = &coords[3];
    let x0 = Ident::new(&format!("{}0", x), x.span());
    let y0 = Ident::new(&format!("{}0", y), y.span());
    let z0 = Ident::new(&format!("{}0", z), z.span());
    let w0 = Ident::new(&format!("{}0", w), w.span());
    quote! {
        let x_mask = crate::interpolation::shared::in_bounds_mask(#x0, (#d3_max) as f64, mode);
        let y_mask = crate::interpolation::shared::in_bounds_mask(#y0, (#d2_max) as f64, mode);
        let z_mask = crate::interpolation::shared::in_bounds_mask(#z0, (#d1_max) as f64, mode);
        let w_mask = crate::interpolation::shared::in_bounds_mask(#w0, (#d0_max) as f64, mode);
        match (x_mask, y_mask, z_mask, w_mask) {
            (Some(xm), Some(ym), Some(zm), Some(wm)) => result * xm * ym * zm * wm,
            _ => result,
        }
    }
}
