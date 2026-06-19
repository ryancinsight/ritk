//! Poisson (shot) noise filter.

use super::fastnorm::{hash, FastNorm};
use super::mersenne::MersenneTwister;
use super::DEFAULT_NOISE_SEED;
use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Poisson (shot) noise filter for low-photon-count simulation.
///
/// Applies Poisson-distributed noise scaled by a factor `λ`:
///
/// ```text
/// in = scale · I(x)
/// I'(x) = Poisson(in) / scale            if in < 50  (Knuth, MT19937 uniforms)
/// I'(x) = (in + √in · N(0,1)) / scale    if in ≥ 50  (Normal approx, FastNorm)
/// ```
///
/// Matches `sitk.ShotNoise` (run single-threaded): both ITK generators — the
/// MersenneTwister (Poisson) and the NormalVariateGenerator (Gaussian
/// approximation) — are seeded from the same region seed and stepped only on the
/// branch taken, reproducing ITK bit-for-bit.
///
/// # Complexity
/// O(N) where N is the number of voxels (Poisson sampling is O(λ) per voxel).
pub struct ShotNoiseFilter {
    /// Scale factor for photon count (higher = less noise).
    pub scale: f64,
    /// Random seed (matched against SimpleITK's `uint32` seed; default: 42).
    pub seed: u32,
}

impl ShotNoiseFilter {
    /// Create a filter with the given photon-count scale.
    pub fn new(scale: f64) -> Self {
        Self {
            scale,
            seed: DEFAULT_NOISE_SEED as u32,
        }
    }

    /// Set the random seed (builder pattern).
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Apply Poisson shot noise to a 3-D image, matching `sitk.ShotNoise`
    /// single-threaded.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        // Zero/negative scale: no photons (degenerate, avoids 0/0).
        if self.scale <= 0.0 {
            return Ok(rebuild(vec![0.0f32; vals.len()], dims, image));
        }
        let seed = hash(self.seed, 0);
        let mut rand = MersenneTwister::new(seed);
        let mut randn = FastNorm::new(seed as i32);
        let scale = self.scale;

        let out: Vec<f32> = vals
            .iter()
            .map(|&v| {
                let inp = scale * v as f64;
                if inp < 50.0 {
                    // Knuth Poisson over the MT19937 uniform stream.
                    let l = (-inp).exp();
                    let mut k: i64 = 0;
                    let mut p = 1.0f64;
                    loop {
                        k += 1;
                        p *= rand.variate();
                        if p <= l {
                            break;
                        }
                    }
                    ((k - 1) as f64 / scale) as f32
                } else {
                    // Normal approximation: Poisson(λ) ≈ N(λ, λ).
                    let out = inp + inp.sqrt() * randn.variate();
                    (out / scale) as f32
                }
            })
            .collect();
        Ok(rebuild(out, dims, image))
    }
}

impl Default for ShotNoiseFilter {
    fn default() -> Self {
        Self::new(1.0)
    }
}
