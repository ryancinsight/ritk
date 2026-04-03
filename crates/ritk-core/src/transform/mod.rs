//! Transform types and operations.
//!
//! This module provides transform traits and implementations
//! for spatial coordinate transformations.

pub mod affine;
pub mod bspline;
pub mod chained;
pub mod displacement_field;
pub mod rigid;
pub mod scale;
pub mod static_displacement_field;
pub mod trait_;
pub mod translation;
pub mod versor;

pub use affine::AffineTransform;
pub use bspline::BSplineTransform;
pub use chained::ChainedTransform;
pub use displacement_field::{
    DisplacementField, DisplacementField2D, DisplacementField3D, DisplacementFieldTransform,
    DisplacementFieldTransform2D, DisplacementFieldTransform3D,
};
pub use rigid::RigidTransform;
pub use scale::ScaleTransform;
pub use static_displacement_field::{
    StaticDisplacementField, StaticDisplacementField2D, StaticDisplacementField3D,
    StaticDisplacementFieldTransform, StaticDisplacementFieldTransform2D,
    StaticDisplacementFieldTransform3D,
};
pub use trait_::{Resampleable, Transform};
pub use translation::TranslationTransform;
pub use versor::VersorRigid3DTransform;
