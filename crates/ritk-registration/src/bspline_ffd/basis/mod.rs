//! Cubic B-spline basis functions, pre-computed axis cache, and displacement evaluation.
//!
//! Implements the Rueckert (1999) uniform cubic B-spline for the
//! BSpline FFD deformable registration engine.
//!
//! # Optimization
//!
//! The [`BasisCache`] pre-computes the 4 basis values + control-point indices once
//! per axis coordinate (Sprint 308), converting the hot path from compute to lookup.
//! Interior-range detection eliminates ~1B bounds-check branches for a 256Â³ volume.

pub(super) mod cache;
pub(super) mod evaluate;
pub(super) mod scalar;

pub use cache::BasisCache;
pub use evaluate::{
    evaluate_bspline_displacement, evaluate_bspline_displacement_fast,
    evaluate_bspline_displacement_fast_into, init_control_grid };
pub use scalar::{cubic_bspline_basis, AxisBasis};
