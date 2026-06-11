//! Transform sub-modules — the concrete transform types moved from ritk-core.
//!
//! Module declarations match the original `ritk-core/src/transform/` layout.

// ── Transform trait (from ritk-core) ─────────────────────────────────────────
pub use ritk_core::transform::{Resampleable, Transform};

// ── Sub-modules ──────────────────────────────────────────────────────────────
pub mod affine;
pub mod bspline;
pub mod composition;
pub mod displacement_field;

// ── Concrete type re-exports ─────────────────────────────────────────────────

// Affine family
pub use affine::affine::AffineTransform;
pub use affine::rigid::RigidTransform;
pub use affine::scale::ScaleTransform;
pub use affine::translation::TranslationTransform;
pub use affine::versor::VersorRigid3DTransform;

// B-spline
pub use bspline::BSplineTransform;

// Composition
pub use composition::chain::ChainedTransform;
pub use composition::io::{CompositeTransform, TransformDescription};

// Displacement fields
pub use displacement_field::static_::field::{
    StaticDisplacementField, StaticDisplacementFieldTransform,
};
pub use displacement_field::{DisplacementField, DisplacementFieldTransform};
