//! NIfTI-1 / NIfTI-2 single-file header model, parsing, and encoding.
//!
//! This module owns the [`NiftiHeader`] domain type and the NIfTI-1/2 byte
//! layout. The byte-field codec ([`raw`]), field validation ([`validate`]), and
//! `f64`→`f32` narrowing ([`convert`]) live in focused sibling modules.

mod convert;
mod datatype;
mod lane;
mod raw;
mod types;
mod validate;

#[cfg(test)]
mod tests;

pub(crate) use datatype::NiftiDatatype;
pub(crate) use lane::NiftiLane;
pub(crate) use types::*;
pub(crate) use validate::{checked_spatial_pixdim, qfac_from_pixdim};
