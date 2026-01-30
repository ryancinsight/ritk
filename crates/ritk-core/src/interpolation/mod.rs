//! Interpolation types and operations.
//!
//! This module provides interpolation traits and implementations
//! for sampling values at continuous coordinates.

pub mod trait_;
pub mod linear;
pub mod nearest;

pub use trait_::Interpolator;
pub use linear::LinearInterpolator;
pub use nearest::NearestNeighborInterpolator;
