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
