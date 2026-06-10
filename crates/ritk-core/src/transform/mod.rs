//! Transform types and operations.
//!
//! Deep vertical hierarchy (Sprint 350 Phase 3, ARCH-352-01):
//!
//! ```text
//! transform/
//! ├── trait_.rs              Transform trait, Resampleable
//! ├── affine/                Affine family (linear transforms)
//! │   ├── mod.rs
//! │   ├── affine.rs           AffineTransform
//! │   ├── rigid.rs            RigidTransform
//! │   ├── scale.rs            ScaleTransform
//! │   ├── translation.rs      TranslationTransform
//! │   └── versor.rs           VersorRigid3DTransform
//! ├── bspline/               B-spline free-form deformations
//! │   ├── ffd/                FFD-specific kernels
//! │   ├── interpolation/      B-spline interpolation kernels
//! │   ├── mapping.rs
//! │   └── mod.rs              BSplineTransform
//! ├── displacement_field/    Dense displacement fields
//! │   ├── static/             StaticDisplacementField (parametric, fixed grid)
//! │   ├── parametric/         Resampleable parametric DF
//! │   ├── core.rs             Core tensor ops
//! │   ├── grid.rs             Grid sampling
//! │   ├── resample.rs         Resample to new grid
//! │   ├── transform.rs        DisplacementFieldTransform
//! │   └── mod.rs              DisplacementField re-exports
//! ├── composition/           Transform composition & IO
//! │   ├── chain.rs            ChainedTransform
//! │   ├── io.rs               CompositeTransform, TransformDescription
//! │   └── tests.rs
//! └── mod.rs                 (this file)
//! ```
//
// All public re-exports below preserve the legacy flat-path API
// (`ritk_core::transform::AffineTransform`, etc.) so external callers
// in `ritk-registration`, `ritk-cli`, `ritk-python`, `ritk-model`,
// and `ritk-core/tests` continue to compile unchanged.

pub mod trait_;

// ── Affine family ────────────────────────────────────────────────────────────
pub mod affine;

// ── B-spline free-form deformations ──────────────────────────────────────────
pub mod bspline;

// ── Dense displacement fields ────────────────────────────────────────────────
pub mod displacement_field;

// ── Transform composition & IO ───────────────────────────────────────────────
pub mod composition;

// ── Legacy re-exports (preserve public API) ─────────────────────────────────
pub use affine::affine::AffineTransform;
pub use affine::rigid::RigidTransform;
pub use affine::scale::ScaleTransform;
pub use affine::translation::TranslationTransform;
pub use affine::versor::VersorRigid3DTransform;
pub use bspline::BSplineTransform;
pub use composition::chain::ChainedTransform;
pub use composition::io::{CompositeTransform, TransformDescription};
pub use displacement_field::static_::field::{
    StaticDisplacementField, StaticDisplacementFieldTransform,
};
pub use displacement_field::{DisplacementField, DisplacementFieldTransform};
pub use trait_::{Resampleable, Transform};
