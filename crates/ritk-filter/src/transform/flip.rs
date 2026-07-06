//! Axis-flip image filter.
//!
//! # Mathematical Specification
//!
//! For a 3-D image `I : ℤ³ → ℝ` with shape `[nz, ny, nx]` and flip policies
//! `flip = [fz, fy, fx]` where each `f*` is [`FlipPolicy::Keep`] or [`FlipPolicy::Flip`]:
//!
//! `out(iz, iy, ix) = I(iz', iy', ix')`
//!
//! where:
//! - `iz' = if fz { nz − 1 − iz } else { iz }`
//! - `iy' = if fy { ny − 1 − iy } else { iy }`
//! - `ix' = if fx { nx − 1 − ix } else { ix }`
//!
//! # Properties
//!
//! - Involutory: applying the same flip twice returns the original image.
//! - Preserves shape and all spatial metadata.
//! - `flip = [Keep, Keep, Keep]` is the identity transform.
//! - O(N) time and O(N) output space.
//!
//! # ITK / ImageJ Parity
//!
//! | Filter             | ITK class          | ImageJ (Image > Transform)  |
//! |--------------------|--------------------|-----------------------------|
//! | `FlipImageFilter`  | `FlipImageFilter`  | Flip Horizontally / Vertically |

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};
use serde::{Deserialize, Serialize};

/// Per-axis flip policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum FlipPolicy {
    /// Keep the axis as-is.
    #[default]
    Keep,
    /// Flip this axis.
    Flip,
}

impl From<bool> for FlipPolicy {
    fn from(flip: bool) -> Self {
        if flip {
            FlipPolicy::Flip
        } else {
            FlipPolicy::Keep
        }
    }
}

impl From<FlipPolicy> for bool {
    fn from(policy: FlipPolicy) -> Self {
        matches!(policy, FlipPolicy::Flip)
    }
}

/// Flip a 3-D image along any combination of the Z, Y, and X axes.
///
/// # Example
///
/// ```rust,ignore
/// // Flip along the Z axis only
/// let out = FlipImageFilter::new([FlipPolicy::Flip, FlipPolicy::Keep, FlipPolicy::Keep])
///     .apply(&image)?;
/// ```
#[derive(Debug, Clone)]
pub struct FlipImageFilter {
    /// Which axes to flip: `[flip_z, flip_y, flip_x]`.
    pub axes: [FlipPolicy; 3],
}

impl FlipImageFilter {
    pub fn new(axes: [FlipPolicy; 3]) -> Self {
        Self { axes }
    }

    /// Construct from a boolean array (`true` = flip, `false` = keep).
    pub fn from_bools(axes: [bool; 3]) -> Self {
        Self {
            axes: axes.map(FlipPolicy::from),
        }
    }

    /// Convenience constructor: flip Z axis only.
    pub fn flip_z() -> Self {
        Self::new([FlipPolicy::Flip, FlipPolicy::Keep, FlipPolicy::Keep])
    }

    /// Convenience constructor: flip Y axis only.
    pub fn flip_y() -> Self {
        Self::new([FlipPolicy::Keep, FlipPolicy::Flip, FlipPolicy::Keep])
    }

    /// Convenience constructor: flip X axis only.
    pub fn flip_x() -> Self {
        Self::new([FlipPolicy::Keep, FlipPolicy::Keep, FlipPolicy::Flip])
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let vals = &vals_vec;

        let [nz, ny, nx] = dims;
        let [fz, fy, fx] = self.axes;
        let mut out = vec![0.0f32; nz * ny * nx];

        for iz in 0..nz {
            let iz_src = if matches!(fz, FlipPolicy::Flip) {
                nz - 1 - iz
            } else {
                iz
            };
            for iy in 0..ny {
                let iy_src = if matches!(fy, FlipPolicy::Flip) {
                    ny - 1 - iy
                } else {
                    iy
                };
                for ix in 0..nx {
                    let ix_src = if matches!(fx, FlipPolicy::Flip) {
                        nx - 1 - ix
                    } else {
                        ix
                    };
                    let dst = iz * ny * nx + iy * nx + ix;
                    let src = iz_src * ny * nx + iy_src * nx + ix_src;
                    out[dst] = vals[src];
                }
            }
        }

        Ok(rebuild(out, dims, image))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_flip.rs"]
mod tests_flip;
