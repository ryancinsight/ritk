//! Parsed input types for the `interp_dim_template!` proc-macros.

use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, LitInt, Token};

/// Parsed input to `interp_dim_template!`.
///
/// Input format (matches the `macro_rules!` version):
/// ```text
/// interp_dim_template!(
/// <dim: int>, // 1, 2, 3, or 4
/// <func: ident>, // function name suffix (e.g. interpolate_3d)
/// <coords: idents>,>, // x / x, y / x, y, z / x, y, z, w
/// <weights: idents>,>, // wx / wx, wy / wx, wy, wz / wx, wy, wz, ww
/// <d_max: exprs>,>, // d0-1 / d1-1, d0-1 / d2-1, d1-1, d0-1 / d3-1, d2-1, d1-1, d0-1
/// { <body: tokens> } // gather + lerp cascade; must bind `result`
/// );
/// ```
pub(crate) struct InterpDimInput {
    pub dim: LitInt,
    pub func: Ident,
    pub _comma1: Token![,],
    pub coords: Vec<Ident>,
    pub _comma2: Token![,],
    pub weights: Vec<Ident>,
    pub _comma3: Token![,],
    pub d_max: Vec<Expr>,
    pub _comma4: Token![,],
    pub body: TokenStream,
}

impl Parse for InterpDimInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let dim: LitInt = input.parse()?;
        let dim_val: usize = dim.base10_parse()?;
        let _comma1: Token![,] = input.parse()?;
        let func: Ident = input.parse()?;
        let _comma2: Token![,] = input.parse()?;

        // Coords: comma-separated idents.
        let mut coords = Vec::with_capacity(dim_val);
        for i in 0..dim_val {
            coords.push(input.parse::<Ident>()?);
            if i + 1 < dim_val {
                let _: Token![,] = input.parse()?;
            }
        }

        let _comma3: Token![,] = input.parse()?;

        // Weights: comma-separated idents.
        let mut weights = Vec::with_capacity(dim_val);
        for i in 0..dim_val {
            weights.push(input.parse::<Ident>()?);
            if i + 1 < dim_val {
                let _: Token![,] = input.parse()?;
            }
        }

        let _comma4: Token![,] = input.parse()?;

        // d_max: comma-separated expressions. Use `Vec<Expr>` + manual loop
        // (not `Punctuated::<Expr, _, _>::parse_terminated`) because
        // `Expr::parse` in a `Punctuated` context doesn't reliably
        // continue past the first sub-expression of a binary op like
        // `d0 - 1` — the parser stops after `d0` and then expects `,`,
        // failing on the `-`. The manual loop calls `input.parse::<Expr>()`
        // for each element, which correctly handles binary operators
        // (Expr::parse is greedy and consumes the full expression).
        let mut d_max = Vec::with_capacity(dim_val);
        for i in 0..dim_val {
            d_max.push(input.parse::<Expr>()?);
            if i + 1 < dim_val {
                let _: Token![,] = input.parse()?;
            }
        }

        let _comma5: Token![,] = input.parse()?;
        let body: TokenStream = input.parse()?;

        Ok(InterpDimInput {
            dim,
            func,
            _comma1,
            coords,
            _comma2,
            weights,
            _comma3,
            d_max,
            _comma4,
            body,
        })
    }
}

/// Parsed input to `interp_dim_template_typed!`.
///
/// Parallel to `InterpDimInput` with an additional `dims` field for the
/// const-generic shape identifiers.
pub(crate) struct InterpDimTypedInput {
    pub dim: LitInt,
    pub func: Ident,
    pub _comma1: Token![,],
    pub coords: Vec<Ident>,
    pub _comma2: Token![,],
    pub weights: Vec<Ident>,
    pub _comma3: Token![,],
    pub d_max: Vec<Expr>,
    pub _comma4: Token![,],
    pub dims: Vec<Ident>,
    pub _comma5: Token![,],
    pub body: TokenStream,
}

impl Parse for InterpDimTypedInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let dim: LitInt = input.parse()?;
        let dim_val: usize = dim.base10_parse()?;
        let _comma1: Token![,] = input.parse()?;
        let func: Ident = input.parse()?;
        let _comma2: Token![,] = input.parse()?;

        // Coords: comma-separated idents.
        let mut coords = Vec::with_capacity(dim_val);
        for i in 0..dim_val {
            coords.push(input.parse::<Ident>()?);
            if i + 1 < dim_val {
                let _: Token![,] = input.parse()?;
            }
        }

        let _comma3: Token![,] = input.parse()?;

        // Weights: comma-separated idents.
        let mut weights = Vec::with_capacity(dim_val);
        for i in 0..dim_val {
            weights.push(input.parse::<Ident>()?);
            if i + 1 < dim_val {
                let _: Token![,] = input.parse()?;
            }
        }

        let _comma4: Token![,] = input.parse()?;

        // d_max: comma-separated expressions (typically `D2 - 1, D1 - 1, D0 - 1`).
        let mut d_max = Vec::with_capacity(dim_val);
        for i in 0..dim_val {
            d_max.push(input.parse::<Expr>()?);
            if i + 1 < dim_val {
                let _: Token![,] = input.parse()?;
            }
        }

        let _comma5: Token![,] = input.parse()?;

        // dims: comma-separated const-generic idents (e.g. `D0, D1, D2`).
        let mut dims = Vec::with_capacity(dim_val);
        for i in 0..dim_val {
            dims.push(input.parse::<Ident>()?);
            if i + 1 < dim_val {
                let _: Token![,] = input.parse()?;
            }
        }

        let _comma6: Token![,] = input.parse()?;
        let body: TokenStream = input.parse()?;

        Ok(InterpDimTypedInput {
            dim,
            func,
            _comma1,
            coords,
            _comma2,
            weights,
            _comma3,
            d_max,
            _comma4,
            dims,
            _comma5,
            body,
        })
    }
}
