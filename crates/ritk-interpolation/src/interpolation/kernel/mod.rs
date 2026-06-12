//! Per-shape interpolation kernels (Sprint 353, ARCH-353-01).
//!
//! Groups all concrete interpolator implementations under a single `kernel/`
//! subfolder so that:
//!
//! 1. The dispatch surface (`interpolation::dispatch`) is decoupled from
//!    the per-kernel implementations.
//! 2. New kernels (e.g. Lanczos variants) can be added without touching
//!    `interpolation/mod.rs`.
//! 3. The `macros.rs` template (DRY-353-02) can generate per-D linear
//!    interpolation from a single source.
//!
//! All public types are re-exported at `interpolation::*` (flat path) for
//! backward compatibility with external callers.

use serde::{Deserialize, Serialize};

use crate::interpolation::shared::OutOfBoundsMode;

/// Boundary handling policy for interpolation queries outside the volume.
///
/// - `Extend`: out-of-bounds samples clamp to the nearest edge voxel
///   (or use weight renormalization for B-Spline). Default.
/// - `ZeroPad`: out-of-bounds samples return `0.0`, preventing spurious
///   correlation peaks in MI-based registration metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BoundsPolicy {
    /// Clamp to the nearest edge voxel (or renormalize weights for B-Spline).
    #[default]
    Extend,
    /// Return `0.0` for out-of-bounds samples.
    ZeroPad,
}

impl BoundsPolicy {
    /// Convert to the canonical [`OutOfBoundsMode`] used by interpolation kernels.
    ///
    /// - [`BoundsPolicy::ZeroPad`] → [`OutOfBoundsMode::ZeroPad`]
    /// - [`BoundsPolicy::Extend`]  → [`OutOfBoundsMode::Clamp`]
    #[inline]
    pub fn as_out_of_bounds_mode(self) -> OutOfBoundsMode {
        match self {
            Self::ZeroPad => OutOfBoundsMode::ZeroPad,
            Self::Extend => OutOfBoundsMode::Clamp,
        }
    }
}

pub mod bspline;
pub mod linear;
pub mod nearest;
pub mod sinc;

/// DRY-353-02: macro_rules! template for per-D interpolation.
///
/// Generates `interpolate_1d`, `interpolate_2d`, `interpolate_3d`, and
/// `interpolate_4d` from a single template, eliminating the ~95% duplication
/// across `kernel/linear/dim{1,2,3,4}.rs`.
///
/// The template takes the per-D body as a token tree, so callers can plug in
/// shape-specific gather/weight/lerp logic. See `macros.rs` for usage examples
/// and the migration plan.
#[macro_use]
pub mod macros;
