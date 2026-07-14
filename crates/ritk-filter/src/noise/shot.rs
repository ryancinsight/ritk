//! Poisson (shot) noise filter.

use super::fastnorm::{hash, FastNorm};
use super::mersenne::MersenneTwister;
use super::DEFAULT_NOISE_SEED;
use crate::native_support::map_flat_image;
use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_core::image::Image;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::Backend;
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
        Ok(rebuild(self.apply_values(&vals), dims, image))
    }

    /// Apply seeded Poisson shot noise to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &NativeImage<f32, B, 3>,
        backend: &B,
    ) -> Result<NativeImage<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        map_flat_image(image, backend, |values, _| self.apply_values(values))
    }

    fn apply_values(&self, values: &[f32]) -> Vec<f32> {
        // Zero/negative scale: no photons (degenerate, avoids 0/0).
        if self.scale <= 0.0 {
            return vec![0.0; values.len()];
        }
        let seed = hash(self.seed, 0);
        let mut rand = MersenneTwister::new(seed);
        let mut randn = FastNorm::new(seed as i32);
        let scale = self.scale;

        values
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
            .collect()
    }
}

impl Default for ShotNoiseFilter {
    fn default() -> Self {
        Self::new(1.0)
    }
}
