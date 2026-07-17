//! Salt-and-pepper (impulse) noise filter.

use super::fastnorm::hash;
use super::mersenne::MersenneTwister;
use super::DEFAULT_NOISE_SEED;
use crate::native_support::map_flat_image;
use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_core::image::Image;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Salt-and-pepper (impulse) noise filter.
///
/// Independently replaces each voxel with either the minimum or maximum value
/// of the image at the given probability, simulating dead/stuck pixels.
///
/// ```text
/// With probability p:      I'(x) = salt_value   if the 2nd draw < 0.5
///                          I'(x) = pepper_value otherwise
/// With probability 1 − p:  I'(x) = I(x)         (unchanged)
/// ```
///
/// Matches `sitk.SaltAndPepperNoise` (run single-threaded): the MT19937 generator
/// is drawn once per voxel to test the probability, and a second time (only when
/// the voxel is hit) to choose salt vs pepper. `salt_value`/`pepper_value` default
/// to ITK's `±NumericTraits<float>::max()`.
///
/// # Complexity
/// O(N) where N is the number of voxels.
pub struct SaltAndPepperNoiseFilter {
    /// Probability of a voxel being replaced (0.0–1.0).
    pub probability: f64,
    /// Random seed (matched against SimpleITK's `uint32` seed; default: 42).
    pub seed: u32,
    /// Value written for "salt" (ITK default `f32::MAX`).
    pub salt_value: f32,
    /// Value written for "pepper" (ITK default `-f32::MAX`).
    pub pepper_value: f32,
}

impl SaltAndPepperNoiseFilter {
    /// Create a filter with the given replacement probability.
    pub fn new(probability: f64) -> Self {
        Self {
            probability,
            seed: DEFAULT_NOISE_SEED as u32,
            salt_value: f32::MAX,
            pepper_value: -f32::MAX,
        }
    }

    /// Set the random seed (builder pattern).
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Apply salt-and-pepper noise to a 3-D image. Single region ⇒
    /// `seed = Hash(userSeed, 0)`; the generator is stepped in scanline order.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        Ok(rebuild(self.apply_values(&vals), dims, image))
    }

    /// Apply seeded salt-and-pepper noise to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &NativeImage<f32, B, 3>,
        backend: &B) -> Result<NativeImage<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        map_flat_image(image, backend, |values, _| self.apply_values(values))
    }

    fn apply_values(&self, values: &[f32]) -> Vec<f32> {
        let mut gen = MersenneTwister::new(hash(self.seed, 0));
        values
            .iter()
            .map(|&v| {
                if gen.variate() < self.probability {
                    if gen.variate() < 0.5 {
                        self.salt_value
                    } else {
                        self.pepper_value
                    }
                } else {
                    v
                }
            })
            .collect()
    }
}

impl Default for SaltAndPepperNoiseFilter {
    fn default() -> Self {
        Self::new(0.05)
    }
}
