//! Poisson (shot) noise filter.

use anyhow::Result;
use burn::tensor::backend::Backend;
use rand::prelude::*;
use rand::rngs::StdRng;
use ritk_core::filter::ops::{extract_vec, rebuild};
use ritk_core::image::Image;

/// Poisson (shot) noise filter for low-photon-count simulation.
///
/// Applies Poisson-distributed noise scaled by a factor `λ`:
///
/// ```text
/// I'(x) = Poisson(λ · max(I(x), 0)) / λ
/// ```
///
/// Voxels with I(x) < 0 are clamped to 0 before Poisson sampling.
/// The `scale` parameter controls the noise level: smaller `scale`
/// yields higher relative noise (fewer photons per unit intensity).
///
/// # Use cases
/// - Simulate low-dose CT / low-count PET acquisition
/// - Test denoising algorithms under Poisson noise models
/// - Radiographic quantum noise simulation
///
/// # Complexity
/// O(N) where N is the number of voxels.
pub struct ShotNoiseFilter {
    /// Scale factor for photon count (higher = less noise).
    /// Typical values: 0.1 (very noisy) to 100.0 (nearly noiseless).
    pub scale: f64,
    /// Random seed for reproducibility (default: 42).
    pub seed: u64,
}

impl ShotNoiseFilter {
    /// Create a filter with the given photon-count scale.
    pub fn new(scale: f64) -> Self {
        Self { scale, seed: 42 }
    }

    /// Set the random seed (builder pattern).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Apply Poisson shot noise to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        // Zero scale means no photons — all output is zero.
        if self.scale <= 0.0 {
            let out = vec![0.0_f32; vals.len()];
            return Ok(rebuild(out, dims, image));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        // Pre-generate Poisson samples sequentially for deterministic ordering.
        let samples: Vec<f64> = vals
            .iter()
            .map(|&v| {
                let intensity = (v as f64).max(0.0);
                let lambda = intensity * self.scale;
                poisson_sample(&mut rng, lambda)
            })
            .collect();
        let out: Vec<f32> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(vals.len(), |i| {
                (samples[i] / self.scale) as f32
            });
        Ok(rebuild(out, dims, image))
    }
}

impl Default for ShotNoiseFilter {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Sample from Poisson(λ) using Knuth's method for small λ
/// and normal approximation for λ ≥ 30.
fn poisson_sample(rng: &mut StdRng, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 0.0;
    }
    if lambda < 30.0 {
        // Knuth's algorithm: generate exponential inter-arrival times.
        let l = (-lambda).exp();
        let mut k = 0.0_f64;
        let mut p = 1.0_f64;
        loop {
            k += 1.0;
            p *= rng.random::<f64>();
            if p <= l {
                return k - 1.0;
            }
        }
    } else {
        // Normal approximation: Poisson(λ) ≈ N(λ, λ) for large λ.
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z = (-2.0_f64 * u1.max(f64::MIN_POSITIVE).ln()).sqrt()
            * (2.0 * std::f64::consts::TAU * u2).cos();
        (lambda + z * lambda.sqrt()).max(0.0).round()
    }
}
