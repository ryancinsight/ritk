//! Transform types and operations.
//!
//! This module provides transform traits and implementations
//! for spatial coordinate transformations.

pub mod trait_;
pub mod translation;
pub mod rigid;
pub mod versor;
pub mod affine;
pub mod bspline;
pub mod chained;
pub mod scale;
pub mod displacement_field;

pub use trait_::{Transform, Resampleable};
pub use translation::TranslationTransform;
pub use rigid::RigidTransform;
pub use versor::VersorRigid3DTransform;
pub use affine::AffineTransform;
pub use bspline::BSplineTransform;
pub use chained::ChainedTransform;
pub use scale::ScaleTransform;
pub use displacement_field::{
    DisplacementFieldTransform, DisplacementField,
    DisplacementFieldTransform2D, DisplacementFieldTransform3D, 
    DisplacementField2D, DisplacementField3D
};
