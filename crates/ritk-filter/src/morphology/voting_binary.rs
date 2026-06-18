//! Voting binary image filter — cellular-automata label voting.
//!
//! # Mathematical Specification
//!
//! A **voting binary filter** applies a single-step cellular automaton to a
//! binary image using a cubic neighbourhood of half-width `radius`:
//!
//! Let `N_fg(p)` = number of foreground neighbours of `p` in the neighbourhood.
//!
//! The output at `p` is:
//! ```text
//! I_out(p) = fg_value  if I(p) = bg_value  AND  N_fg(p) >= birth_threshold
//!          = bg_value  if I(p) = fg_value  AND  N_fg(p) <  survival_threshold
//!          = I(p)      otherwise
//! ```
//!
//! - `birth_threshold`: minimum foreground count for a background voxel to become foreground.
//! - `survival_threshold`: minimum foreground count for an existing foreground voxel to survive.
//!
//! # ITK Parity
//!
//! Corresponds to `itk::VotingBinaryImageFilter<TInputImage, TOutputImage>`.
//! ITK defaults: `Radius = 1`, `BirthThreshold = 1`, `SurvivalThreshold = 1`,
//! `ForegroundValue = 1`, `BackgroundValue = 0`.
//!
//! # Reference
//!
//! - Breu, H. et al. (1995). Linear time Euclidean distance algorithms.
//!   *IEEE Trans. PAMI* 17(5):529–533.

use super::types::ForegroundValue;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

/// Voting binary image filter.
///
/// Each voxel casts a "vote" by examining how many of its neighbours are foreground.
/// Background voxels may turn foreground (birth); foreground voxels may turn background
/// (death). A single iteration is applied.
#[derive(Debug, Clone)]
pub struct VotingBinaryImageFilter {
    /// Half-width of the cubic neighbourhood in voxels. Default 1.
    pub radius: usize,
    /// Minimum foreground neighbour count for a background voxel to become foreground.
    pub birth_threshold: usize,
    /// Minimum foreground neighbour count for a foreground voxel to survive.
    pub survival_threshold: usize,
    /// Foreground intensity value. Default 1.0.
    pub foreground_value: ForegroundValue,
    /// Background intensity value. Default 0.0.
    pub background_value: f32,
}

impl VotingBinaryImageFilter {
    /// Construct with explicit parameters.
    pub fn new(
        radius: usize,
        birth_threshold: usize,
        survival_threshold: usize,
        foreground_value: impl Into<ForegroundValue>,
        background_value: f32,
    ) -> Self {
        Self {
            radius,
            birth_threshold,
            survival_threshold,
            foreground_value: foreground_value.into(),
            background_value,
        }
    }
}

impl Default for VotingBinaryImageFilter {
    fn default() -> Self {
        Self::new(1, 1, 1, ForegroundValue::ONE, 0.0)
    }
}

impl VotingBinaryImageFilter {
    /// Apply a single voting step to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;
        let device = image.data().device();

        let r = self.radius;
        let birth = self.birth_threshold;
        let survival = self.survival_threshold;
        let fg = f32::from(self.foreground_value);
        let bg = self.background_value;

        let slab = ny * nx;
        // PERF-378-01: parallelise over flat voxel index — each voxel reads a window from
        // vals (read-only) with no inter-voxel output dependencies; bit-identical to serial.
        let out = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny * nx, |flat| {
            let iz = flat / slab;
            let rem = flat - iz * slab;
            let iy = rem / nx;
            let ix = rem - iy * nx;

            let v = vals[flat];
            let is_fg = (v - fg).abs() < 1e-5;

            let z0 = iz.saturating_sub(r);
            let z1 = (iz + r).min(nz - 1);
            let y0 = iy.saturating_sub(r);
            let y1 = (iy + r).min(ny - 1);
            let x0 = ix.saturating_sub(r);
            let x1 = (ix + r).min(nx - 1);

            let mut fg_count = 0usize;
            for kz in z0..=z1 {
                for ky in y0..=y1 {
                    for kx in x0..=x1 {
                        if (vals[kz * slab + ky * nx + kx] - fg).abs() < 1e-5 {
                            fg_count += 1;
                        }
                    }
                }
            }

            if !is_fg {
                if fg_count >= birth {
                    fg
                } else {
                    bg
                }
            } else if fg_count >= survival {
                fg
            } else {
                bg
            }
        });

        let shape = Shape::new([nz, ny, nx]);
        let data = TensorData::new(out, shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_voting_binary.rs"]
mod tests_voting_binary;
