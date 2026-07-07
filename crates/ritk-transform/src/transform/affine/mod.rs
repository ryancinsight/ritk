//! Affine transform family: affine, rigid, scale, translation, versor.

#[allow(clippy::module_inception)]
pub mod affine;
pub mod atlas_affine;
pub mod rigid;
pub mod scale;
pub mod translation;
pub mod versor;

pub use atlas_affine::{AtlasAffineError, AtlasAffineTransform};
