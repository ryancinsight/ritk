//! Salt-and-pepper (impulse) noise filter.

use super::DEFAULT_NOISE_SEED;
use anyhow::Result;
use burn::tensor::backend::Backend;
use rand::prelude::*;
use rand::rngs::StdRng;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Salt-and-pepper (impulse) noise filter.
///
/// Independently replaces each voxel with either the minimum or maximum value
/// of the image at the given probability, simulating dead/stuck pixels.
///
/// ```text
/// With probability p:      I'(x) = max(I)  (salt)  with prob p/2
///                          I'(x) = min(I)  (pepper) with prob p/2
/// With probability 1 − p:  I'(x) = I(x)    (unchanged)
/// ```
///
/// # Use cases
/// - Simulate sensor defects / dead pixels
/// - Test median filter and morphological filter robustness
///
/// # Complexity
/// O(N) where N is the number of voxels.
pub struct SaltAndPepperNoiseFilter {
    /// Probability of a voxel being replaced (0.0–1.0).
    pub probability: f64,
    /// Random seed for reproducibility (default: 42).
    pub seed: u64,
}

impl SaltAndPepperNoiseFilter {
    /// Create a filter with the given replacement probability.
    pub fn new(probability: f64) -> Self {
        Self {
            probability,
            seed: DEFAULT_NOISE_SEED,
        }
    }

    /// Set the random seed (builder pattern).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Apply salt-and-pepper noise to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let half_p = self.probability / 2.0;
        // Pre-generate random draws sequentially for deterministic ordering.
        let draws: Vec<f64> = vals.iter().map(|_| rng.random()).collect();
        let out: Vec<f32> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(vals.len(), |i| {
                let (v, r) = (vals[i], draws[i]);
                if r < half_p {
                    min_val // pepper
                } else if r < self.probability {
                    max_val // salt
                } else {
                    v // unchanged
                }
            });
        Ok(rebuild(out, dims, image))
    }
}

impl Default for SaltAndPepperNoiseFilter {
    fn default() -> Self {
        Self::new(0.05)
    }
}
