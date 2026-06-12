//! Additive Gaussian noise filter.

use super::DEFAULT_NOISE_SEED;
use anyhow::Result;
use burn::tensor::backend::Backend;
use rand::prelude::*;
use rand::rngs::StdRng;
use ritk_tensor_ops::{extract_vec, rebuild};
use ritk_core::image::Image;

/// Additive Gaussian noise filter.
///
/// Adds independent Gaussian noise to every voxel:
///
/// ```text
/// I'(x) = I(x) + N(μ, σ)
/// ```
///
/// where `N(μ, σ)` is a normally-distributed random variable with mean `μ`
/// and standard deviation `σ`.
///
/// # Use cases
/// - Simulate thermal/electronic noise in CT/MR acquisition
/// - Test registration robustness to Gaussian perturbation
///
/// # Complexity
/// O(N) where N is the number of voxels.
pub struct AdditiveGaussianNoiseFilter {
    /// Mean of the Gaussian noise distribution (default: 0.0).
    pub mean: f64,
    /// Standard deviation of the Gaussian noise distribution.
    pub std: f64,
    /// Random seed for reproducibility (default: 42).
    pub seed: u64,
}

impl AdditiveGaussianNoiseFilter {
    /// Create a filter with the given standard deviation.
    ///
    /// Mean defaults to 0.0, seed to 42.
    pub fn new(std: f64) -> Self {
        Self {
            mean: 0.0,
            std,
            seed: DEFAULT_NOISE_SEED,
        }
    }

    /// Set the noise mean (builder pattern).
    pub fn with_mean(mut self, mean: f64) -> Self {
        self.mean = mean;
        self
    }

    /// Set the random seed (builder pattern).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Apply additive Gaussian noise to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        // Pre-generate all random variates from a single sequential RNG
        // to guarantee deterministic output for a given seed.
        let gaussians: Vec<f64> = vals
            .iter()
            .map(|_| {
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                super::box_muller(u1, u2)
            })
            .collect();
        let out: Vec<f32> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(vals.len(), |i| {
                (vals[i] as f64 + gaussians[i] * self.std + self.mean) as f32
            });
        Ok(rebuild(out, dims, image))
    }
}

impl Default for AdditiveGaussianNoiseFilter {
    fn default() -> Self {
        Self::new(1.0)
    }
}
