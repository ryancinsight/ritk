//! Noise simulation filters for medical image robustness testing (GAP-262-FLT-05).
//!
//! These filters add synthetic noise to images, useful for:
//! - Evaluating segmentation/registration robustness under varying noise levels
//! - Generating training data with realistic noise profiles
//! - Testing filter and denoising algorithm quality
//!
//! # Filters
//!
//! | Filter | Noise model | Parameters |
//! |---|---|---|
//! | `AdditiveGaussianNoiseFilter` | `I'(x) = I(x) + N(mean, std)` | `mean`, `std` |
//! | `SaltAndPepperNoiseFilter` | Random pixel replacement with min/max | `probability` |
//! | `ShotNoiseFilter` | Poisson process: `I'(x) ~ Poisson(λ·I(x)) / λ` | `scale` |
//! | `SpeckleNoiseFilter` | Multiplicative: `I'(x) = I(x) · (1 + N(0, std))` | `std` |

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;
use rand::prelude::*;
use rand::rngs::StdRng;

// ── AdditiveGaussianNoiseFilter ───────────────────────────────────────────────

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
            seed: 42,
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
    pub fn apply_3d<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let out: Vec<f32> = vals
            .iter()
            .map(|&v| {
                // Box-Muller transform for Gaussian random variate
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let n = (-2.0_f64 * u1.max(f64::MIN_POSITIVE).ln()).sqrt()
                    * (2.0 * std::f64::consts::TAU * u2).cos();
                (v as f64 + n * self.std + self.mean) as f32
            })
            .collect();
        Ok(rebuild(out, dims, image))
    }
}

// ── SaltAndPepperNoiseFilter ──────────────────────────────────────────────────

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
            seed: 42,
        }
    }

    /// Set the random seed (builder pattern).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Apply salt-and-pepper noise to a 3-D image.
    pub fn apply_3d<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut rng = StdRng::seed_from_u64(self.seed);
        let half_p = self.probability / 2.0;
        let out: Vec<f32> = vals
            .iter()
            .map(|&v| {
                let r: f64 = rng.gen();
                if r < half_p {
                    min_val // pepper
                } else if r < self.probability {
                    max_val // salt
                } else {
                    v // unchanged
                }
            })
            .collect();
        Ok(rebuild(out, dims, image))
    }
}

// ── ShotNoiseFilter ───────────────────────────────────────────────────────────

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
    pub fn apply_3d<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        // Zero scale means no photons — all output is zero.
        if self.scale <= 0.0 {
            let out = vec![0.0_f32; vals.len()];
            return Ok(rebuild(out, dims, image));
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let out: Vec<f32> = vals
            .iter()
            .map(|&v| {
                let intensity = (v as f64).max(0.0);
                let lambda = intensity * self.scale;
                let k = poisson_sample(&mut rng, lambda);
                (k / self.scale) as f32
            })
            .collect();
        Ok(rebuild(out, dims, image))
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
            p *= rng.gen::<f64>();
            if p <= l {
                return k - 1.0;
            }
        }
    } else {
        // Normal approximation: Poisson(λ) ≈ N(λ, λ) for large λ.
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let z = (-2.0_f64 * u1.max(f64::MIN_POSITIVE).ln()).sqrt()
            * (2.0 * std::f64::consts::TAU * u2).cos();
        (lambda + z * lambda.sqrt()).max(0.0).round()
    }
}

// ── SpeckleNoiseFilter ────────────────────────────────────────────────────────

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
        Self { std, seed: 42 }
    }

    /// Set the random seed (builder pattern).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Apply speckle noise to a 3-D image.
    pub fn apply_3d<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let out: Vec<f32> = vals
            .iter()
            .map(|&v| {
                // Box-Muller for Gaussian
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let n = (-2.0_f64 * u1.max(f64::MIN_POSITIVE).ln()).sqrt()
                    * (2.0 * std::f64::consts::TAU * u2).cos()
                    * self.std;
                (v as f64 * (1.0 + n)) as f32
            })
            .collect();
        Ok(rebuild(out, dims, image))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_noise.rs"]
mod tests_noise;
