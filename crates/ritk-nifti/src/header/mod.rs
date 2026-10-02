//! NIfTI-1 / NIfTI-2 single-file header model, parsing, and encoding.
//!
//! This module owns the [`NiftiHeader`] domain type and the NIfTI-1/2 byte
//! layout. Parsing ([`parse`]) and encoding ([`encode`]) of that layout,
//! the byte-field codec ([`raw`]), field validation ([`validate`]),
//! `f64`→`f32` narrowing ([`convert`]), `datatype` codes ([`datatype`]), and
//! the `scl_slope`/`scl_inter` rescale ([`scaling`]) live in focused sibling
//! modules.

mod convert;
mod datatype;
mod encode;
mod parse;
mod raw;
mod scaling;
mod types;
mod validate;

pub(crate) use types::*;
