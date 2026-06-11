//! Spatial transformation types extracted from `ritk-core`.
//!
//! Provides concrete implementations of the [`Transform`] and [`Resampleable`]
//! traits (which remain in `ritk-core`) — affine, rigid, scale, translation,
//! versor, B-spline, displacement field, and composite transforms.
//!
//! # Re-exports from `ritk-core`
//!
//! * [`Transform`] — maps points from one physical space to another
//! * [`Resampleable`] — adapts a transform to a new grid/resolution

pub mod transform;

// Re-export everything from the transform module at the crate root for flat-path access
pub use transform::*;
