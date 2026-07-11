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
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
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
        let out = self.vote_values(&vals, dims);
        let device = image.data().device();
        let shape = Shape::new(dims);
        let data = TensorData::new(out, shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }

    /// Apply a voting step to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        ritk_image::native::Image::from_flat_on(
            self.vote_values(image.data_slice()?, image.shape()),
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }

    fn vote_values(&self, values: &[f32], [nz, ny, nx]: [usize; 3]) -> Vec<f32> {
        let (r, birth, survival) = (self.radius, self.birth_threshold, self.survival_threshold);
        let (fg, bg) = (f32::from(self.foreground_value), self.background_value);
        let slab = ny * nx;
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(values.len(), |flat| {
            let z = flat / slab;
            let rem = flat - z * slab;
            let y = rem / nx;
            let x = rem - y * nx;
            let count = (z.saturating_sub(r)..=(z + r).min(nz - 1))
                .flat_map(|z| (y.saturating_sub(r)..=(y + r).min(ny - 1)).map(move |y| (z, y)))
                .flat_map(|(z, y)| {
                    (x.saturating_sub(r)..=(x + r).min(nx - 1)).map(move |x| (z, y, x))
                })
                .filter(|&(z, y, x)| (values[z * slab + y * nx + x] - fg).abs() < 1e-5)
                .count();
            if (values[flat] - fg).abs() < 1e-5 {
                if count >= survival {
                    fg
                } else {
                    bg
                }
            } else if count >= birth {
                fg
            } else {
                bg
            }
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_voting_binary.rs"]
mod tests_voting_binary;
