//! Per-shape interpolation kernels.
//!
//! Groups all concrete interpolator implementations under a single `kernel/`
//! subfolder. All public types are re-exported at `interpolation::*` (flat path)
//! for backward compatibility with external callers.

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
    /// - [`BoundsPolicy::ZeroPad`] -> [`OutOfBoundsMode::ZeroPad`]
    /// - [`BoundsPolicy::Extend`]  -> [`OutOfBoundsMode::Clamp`]
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
