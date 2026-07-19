//! Additive Gaussian noise filter.

use super::fastnorm::{hash, FastNorm};
use super::DEFAULT_NOISE_SEED;
use crate::native_support::map_flat_image;
use anyhow::Result;
use coeus_core::ComputeBackend;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_image::Image as NativeImage;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Additive Gaussian noise filter.
///
/// Adds independent Gaussian noise to every voxel:
///
/// ```text
/// I'(x) = I(x) + N(Î¼, Ïƒ)
/// ```
///
/// where `N(Î¼, Ïƒ)` is a normally-distributed random variable with mean `Î¼`
/// and standard deviation `Ïƒ`.
///
/// The variates come from an exact port of `itk::Statistics::NormalVariateGenerator`
/// (FastNorm), so the output is bit-identical to `sitk.AdditiveGaussianNoise`
/// run single-threaded (whole image = one region, `seed = userSeedÂ·2654435761`,
/// scanline order).
///
/// # Complexity
/// O(N) where N is the number of voxels.
pub struct AdditiveGaussianNoiseFilter {
    /// Mean of the Gaussian noise distribution (default: 0.0).
    pub mean: f64,
    /// Standard deviation of the Gaussian noise distribution.
    pub std: f64,
    /// Random seed (matched against SimpleITK's `uint32` seed; default: 42).
    pub seed: u32,
}

impl AdditiveGaussianNoiseFilter {
    /// Create a filter with the given standard deviation.
    ///
    /// Mean defaults to 0.0, seed to 42.
    pub fn new(std: f64) -> Self {
        Self {
            mean: 0.0,
            std,
            seed: DEFAULT_NOISE_SEED as u32,
        }
    }

    /// Set the noise mean (builder pattern).
    pub fn with_mean(mut self, mean: f64) -> Self {
        self.mean = mean;
        self
    }

    /// Set the random seed (builder pattern).
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Apply additive Gaussian noise to a 3-D image. The image is treated as a
    /// single region (start index 0), so `seed = Hash(userSeed, 0)`; the FastNorm
    /// generator is stepped once per voxel in scanline order.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        Ok(rebuild(self.apply_values(&vals), dims, image))
    }

    /// Apply deterministic additive Gaussian noise to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &NativeImage<f32, B, 3>,
        backend: &B,
    ) -> Result<NativeImage<f32, B, 3>>
    where
        B: ComputeBackend,
    {
        map_flat_image(image, backend, |values, _| self.apply_values(values))
    }

    fn apply_values(&self, values: &[f32]) -> Vec<f32> {
        let mut gen = FastNorm::new(hash(self.seed, 0) as i32);
        values
            .iter()
            .map(|&v| (v as f64 + self.mean + self.std * gen.variate()) as f32)
            .collect()
    }
}

impl Default for AdditiveGaussianNoiseFilter {
    fn default() -> Self {
        Self::new(1.0)
    }
}
