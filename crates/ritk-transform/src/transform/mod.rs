//! Transform sub-modules â€” the concrete transform types moved from ritk-core.
//!
//! Module declarations match the original `ritk-core/src/transform/` layout.

// â”€â”€ Transform trait (from ritk-core) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pub use ritk_core::transform::{Resampleable, Transform};

// â”€â”€ Sub-modules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
pub mod affine;
pub mod composition;
pub mod displacement_field;

// â”€â”€ Concrete type re-exports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// Affine family
pub use affine::affine::AffineTransform;
pub use affine::rigid::RigidTransform;
pub use affine::scale::ScaleTransform;
pub use affine::translation::TranslationTransform;
pub use affine::versor::VersorRigid3DTransform;

// Composition
pub use composition::chain::ChainedTransform;
pub use composition::io::{CompositeTransform, TransformDescription};

// Displacement fields
pub use displacement_field::static_::field::{
    StaticDisplacementField, StaticDisplacementFieldTransform,
};
pub use displacement_field::{
    DisplacementField, DisplacementFieldError, DisplacementFieldTransform,
    DisplacementTransformError, ResampleError,
};
