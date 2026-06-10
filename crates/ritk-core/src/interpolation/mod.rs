//! Interpolation types and operations.
//!
//! Deep vertical hierarchy (Sprint 353, ARCH-353-01):
//!
//! ```text
//! interpolation/
//! ├── mod.rs                 (public API: trait, flat-path re-exports)
//! ├── trait_.rs              Interpolator trait
//! ├── kernel/                ← NEW (per-shape interpolation kernels)
//! │   ├── mod.rs
//! │   ├── macros.rs          ← NEW (DRY-353-02 macro_rules! template)
//! │   ├── linear/            ← from ./linear/
//! │   ├── bspline/           ← from ./bspline/
//! │   ├── nearest.rs         ← from ./
//! │   └── sinc.rs            ← from ./
//! ├── dispatch.rs            (D = 2/3/4 specialization — unchanged)
//! ├── fused.rs               (transform+interpolate fusion — unchanged)
//! ├── tensor_trilinear.rs    (Burn-tensor-based trilinear — unchanged)
//! ├── shared/                (in_bounds_mask, etc. — unchanged)
//! ```
//!
//! All public re-exports below preserve the legacy flat-path API
//! (`ritk_core::interpolation::LinearInterpolator`, etc.) so external callers
//! in `ritk-registration`, `ritk-cli`, `ritk-python`, `ritk-model`,
//! and `ritk-core/tests` continue to compile unchanged.
pub mod dispatch;
pub mod fused;
pub(crate) mod kernel;
pub mod shared;
pub mod tensor_trilinear;
pub mod trait_;

#[cfg(test)]
mod tests;

// ── Legacy flat-path re-exports (preserve public API) ────────────────────────
pub use fused::{transform_and_interpolate_3d, FusedInterpolationResult};
pub use kernel::bspline::BSplineInterpolator;
pub use kernel::linear::LinearInterpolator;
pub use kernel::nearest::NearestNeighborInterpolator;
pub use kernel::sinc::{
    Lanczos4Interpolator, Lanczos5Interpolator, LanczosInterpolator, SincInterpolator,
};
pub use kernel::BoundsPolicy;
pub use shared::OutOfBoundsMode;
pub use tensor_trilinear::trilinear_interpolation;
pub use trait_::Interpolator;
