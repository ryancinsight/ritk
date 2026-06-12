//! Structuring element types — re-exported from the [`ritk_morphology`] crate.
//!
//! The concrete types ([`Cube`], [`Cross`], [`Ball`], [`StructuringElement`],
//! [`Offset3D`], [`SeShape`]) and `const fn` cardinality evaluators live in
//! `ritk_morphology`.  This module is a thin compatibility shim so existing
//! `ritk_core::morphology::*` import paths continue to resolve.

pub use ritk_morphology::*;
