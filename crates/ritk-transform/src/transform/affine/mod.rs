//! Affine transform family: affine, rigid, scale, translation, versor.

#[expect(clippy::module_inception, reason = "ratchet RITK-LINT-1")]
pub mod affine;
pub mod atlas_affine;
pub mod rigid;
pub mod scale;
pub mod translation;
pub mod versor;

pub use atlas_affine::{AtlasAffineError, AtlasAffineTransform};
