//! Transform traits — the core spatial transform abstraction.
//!
//! Concrete transform types (affine, rigid, B-spline, displacement field,
//! composition) have been extracted to the [`ritk_transform`] crate.
//!
//! This module retains only the foundational traits:
//! * [`Transform`] — map points from one physical space to another
//! * [`Resampleable`] — adapt a transform to a new grid/resolution
//!
//! # Usage
//!
//! ```ignore
//! use ritk_core::transform::Transform;
//! use ritk_transform::AffineTransform;
//! ```

pub mod trait_;
pub use trait_::{Resampleable, Transform};
