//! Interpolation types and operations.
//!
//! This module provides interpolation traits and implementations
//! for sampling values at continuous coordinates.

pub mod bspline;
pub mod linear;
pub mod nearest;
pub mod sinc;
pub mod tensor_trilinear;
pub mod trait_;

pub use bspline::BSplineInterpolator;
pub use linear::LinearInterpolator;
pub use nearest::NearestNeighborInterpolator;
pub use sinc::{Lanczos4Interpolator, Lanczos5Interpolator, LanczosInterpolator, SincInterpolator};
pub use tensor_trilinear::trilinear_interpolation;
pub use trait_::Interpolator;
