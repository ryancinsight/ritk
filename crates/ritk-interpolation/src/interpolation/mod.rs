//! Interpolation types and operations.
//!
//! Concrete interpolator implementations extracted from `ritk-core`.
//!
//! The `Interpolator` trait remains in `ritk-core::interpolation::Interpolator`
//! (re-exported here for convenience). All concrete interpolators live here.

pub mod dispatch;
pub mod fused;
pub(crate) mod kernel;
pub mod shared;
pub mod tensor_trilinear;
pub mod trilinear;

#[cfg(test)]
mod tests;

// ── Re-export the Interpolator trait from ritk-core ────────────────────────
pub use ritk_core::interpolation::Interpolator;

// ── Concrete types defined in this crate ───────────────────────────────────
pub use fused::{transform_and_interpolate, FusedInterpolationResult};
pub use kernel::bspline::{bspline_decomposition_coefficients, BSplineInterpolator};
pub use kernel::linear::LinearInterpolator;
pub use kernel::nearest::NearestNeighborInterpolator;
pub use kernel::sinc::{
    Lanczos4Interpolator, Lanczos5Interpolator, LanczosInterpolator, SincInterpolator,
};
pub use kernel::BoundsPolicy;
pub use shared::OutOfBoundsMode;
pub use trilinear::trilinear_interpolation;
