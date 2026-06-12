//! Speckle (multiplicative) noise filter.

use super::DEFAULT_NOISE_SEED;
use anyhow::Result;
use burn::tensor::backend::Backend;
use rand::prelude::*;
use rand::rngs::StdRng;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Speckle (multiplicative) noise filter.
///
/// Applies multiplicative Gaussian noise:
///
/// ```text
/// I'(x) = I(x) · (1 + N(0, σ))
/// ```
///
/// Speckle noise is characteristic of coherent imaging modalities (ultrasound,
/// SAR, optical coherence tomography).
///
/// # Use cases
/// - Simulate ultrasound B-mode speckle
/// - Test speckle-reducing filters (e.g., Lee, Kuan, Frost)
///
/// # Complexity
/// O(N) where N is the number of voxels.
pub struct SpeckleNoiseFilter {
    /// Standard deviation of the multiplicative noise factor (default: 0.05).
    pub std: f64,
    /// Random seed for reproducibility (default: 42).
    pub seed: u64,
}

impl SpeckleNoiseFilter {
    /// Create a filter with the given multiplicative noise std.
    pub fn new(std: f64) -> Self {
        Self {
            std,
            seed: DEFAULT_NOISE_SEED,
        }
    }

    /// Set the random seed (builder pattern).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Apply speckle noise to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        // Pre-generate Gaussian variates sequentially for deterministic ordering.
        let gaussians: Vec<f64> = vals
            .iter()
            .map(|_| {
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                super::box_muller(u1, u2) * self.std
            })
            .collect();
        let out: Vec<f32> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(vals.len(), |i| {
                (vals[i] as f64 * (1.0 + gaussians[i])) as f32
            });
        Ok(rebuild(out, dims, image))
    }
}

impl Default for SpeckleNoiseFilter {
    fn default() -> Self {
        Self::new(0.05)
    }
}
