//! Interpolation types and operations.
//!
//! This module provides interpolation traits and implementations
//! for sampling values at continuous coordinates.

pub mod bspline;
pub mod linear;
pub mod nearest;
pub mod trait_;

pub use bspline::BSplineInterpolator;
pub use linear::LinearInterpolator;
pub use nearest::NearestNeighborInterpolator;
pub use trait_::Interpolator;
