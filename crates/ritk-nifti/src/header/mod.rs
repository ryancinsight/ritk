//! NIfTI-1 / NIfTI-2 single-file header model, parsing, and encoding.
//!
//! This module owns the [`NiftiHeader`] domain type and the NIfTI-1/2 byte
//! layout. The byte-field codec ([`raw`]), field validation ([`validate`]), and
//! `f64`→`f32` narrowing ([`convert`]) live in focused sibling modules.

mod convert;
mod raw;
mod types;
mod validate;

pub(crate) use types::*;
