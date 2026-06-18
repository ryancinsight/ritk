//! Voting binary hole-filling filter (`itk::VotingBinaryHoleFillingImageFilter`).
//!
//! # Mathematical Specification
//!
//! A specialisation of voting-binary that **only fills** background voxels and
//! never removes foreground. A background voxel `p` becomes foreground when the
//! number of foreground voxels in its `(2r+1)` neighbourhood reaches the
//! majority threshold:
//!
//! ```text
//! threshold = (W − 1) / 2 + majority_threshold,   W = Π_d (2 r_d + 1)
//! I_out(p) = fg   if I(p) = fg                                   (foreground survives)
//!          = fg   if I(p) = bg  AND  N_fg(p) ≥ threshold         (hole filled)
//!          = I(p) otherwise
//! ```
//!
//! # Boundary
//!
//! Replicate (clamp) — out-of-bounds neighbours take the edge voxel's value, and
//! the neighbourhood size `W` is the **full** `(2r+1)^D` regardless of image
//! extent. On a `z = 1` (2-D) volume the `z` neighbours clamp onto the single
//! plane, so each in-plane pixel is counted three times and `W = 27` for
//! `r = 1`. Pinned against `sitk.VotingBinaryHoleFilling`: a corner background
//! voxel with in-bounds foreground neighbours fills (clamped fg count 15 ≥ 14),
//! which a constant/zero boundary (count 9) would not.
//!
//! # ITK Parity
//!
//! Corresponds to `itk::VotingBinaryHoleFillingImageFilter`. Defaults
//! `Radius = 1`, `MajorityThreshold = 1`, `ForegroundValue = 1`,
//! `BackgroundValue = 0`. Unlike [`super::voting_binary::VotingBinaryImageFilter`]
//! (which uses a shrink window), this clamps the boundary to match ITK on 2-D
//! and bordering voxels.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Voting binary hole-filling filter (ITK `VotingBinaryHoleFillingImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct VotingBinaryHoleFillingImageFilter {
    /// Per-axis neighbourhood radii `[rz, ry, rx]`. ITK default `[1, 1, 1]`.
    pub radius: [usize; 3],
    /// Votes above the half-window majority required to fill. ITK default `1`.
    pub majority_threshold: usize,
    /// Foreground intensity. ITK default `1.0`.
    pub foreground_value: f32,
    /// Background intensity. ITK default `0.0`.
    pub background_value: f32,
}

impl VotingBinaryHoleFillingImageFilter {
    /// Construct with explicit parameters.
    pub fn new(
        radius: [usize; 3],
        majority_threshold: usize,
        foreground_value: f32,
        background_value: f32,
    ) -> Self {
        Self {
            radius,
            majority_threshold,
            foreground_value,
            background_value,
        }
    }

    /// Apply the hole-filling pass to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let [rz, ry, rx] = self.radius;
        let fg = self.foreground_value;
        let window = (2 * rz + 1) * (2 * ry + 1) * (2 * rx + 1);
        let threshold = (window - 1) / 2 + self.majority_threshold;

        let (rz, ry, rx) = (rz as isize, ry as isize, rx as isize);
        let (snz, sny, snx) = (nz as isize, ny as isize, nx as isize);
        let mut out = vec![0.0f32; vals.len()];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let p = (iz * ny + iy) * nx + ix;
                    if vals[p] == fg {
                        out[p] = fg; // foreground always survives
                        continue;
                    }
                    let mut count = 0usize;
                    for dz in -rz..=rz {
                        let zz = (iz as isize + dz).clamp(0, snz - 1) as usize;
                        for dy in -ry..=ry {
                            let yy = (iy as isize + dy).clamp(0, sny - 1) as usize;
                            let base = (zz * ny + yy) * nx;
                            for dx in -rx..=rx {
                                let xx = (ix as isize + dx).clamp(0, snx - 1) as usize;
                                if vals[base + xx] == fg {
                                    count += 1;
                                }
                            }
                        }
                    }
                    out[p] = if count >= threshold {
                        fg
                    } else {
                        self.background_value
                    };
                }
            }
        }
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
#[path = "tests_voting_hole_filling.rs"]
mod tests_voting_hole_filling;
