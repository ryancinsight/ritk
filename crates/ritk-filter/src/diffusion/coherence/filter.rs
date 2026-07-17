use crate::edge::GaussianSigma;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

use super::pde::ced_diffuse;
use super::scratch::CedScratch;

// ── Public types ──────────────────────────────────────────────────────────────

/// CED configuration.
#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    /// Integration scale ρ — standard deviation of the Gaussian structure-tensor
    /// pre-smoothing kernel. Must be > 0.  Default: 3.0.
    pub sigma: GaussianSigma,
    /// Contrast parameter C. Default: 1e-10.
    pub contrast: f64,
    /// Smoothing parameter α in flat regions. Default: 0.001.
    pub alpha: f64,
    /// Time step Δt. Default: 0.0625 (1/16).
    pub time_step: f64,
    /// Number of iterations. Default: 10.
    pub n_iterations: usize,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            sigma: GaussianSigma::new_unchecked(3.0),
            contrast: 1e-10,
            alpha: 0.001,
            time_step: 0.0625,
            n_iterations: 10,
        }
    }
}

/// Coherence-Enhancing Diffusion filter.
///
/// Smooths images along coherent structures while preserving them across the
/// structure orientation, using the structure tensor to drive anisotropic
/// diffusion (Weickert 1999).
#[derive(Debug, Clone)]
pub struct CoherenceEnhancingDiffusionFilter {
    /// Algorithm configuration.
    pub config: CoherenceConfig,
}

impl CoherenceEnhancingDiffusionFilter {
    /// Create a filter with the given configuration.
    #[inline]
    pub fn new(config: CoherenceConfig) -> Self {
        Self { config }
    }

    /// Apply the CED filter to a 3-D image, returning a diffused copy.
    ///
    /// For images of dimension D < 3, the result is identical to the input
    /// (CED is defined only for 3-D volumes).
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let (vals_vec, dims) = extract_vec_infallible(image);

        let result = if D >= 3 && dims.iter().all(|&d| d >= 3) {
            // Extract the leading 3 dimensions.
            let d3 = [dims[0], dims[1], dims[2]];
            let n3 = d3[0] * d3[1] * d3[2];
            let vals3: Vec<f64> = vals_vec[..n3].iter().map(|&v| v as f64).collect();
            let out3 = ced_diffuse(&vals3, d3, &self.config);
            // Write back, converting f64 → f32.
            let mut result = vals_vec;
            for i in 0..n3 {
                result[i] = out3[i] as f32;
            }
            result
        } else {
            // CED undefined for < 3-D or tiny volumes; return input unchanged.
            vals_vec
        };

        rebuild(result, dims, image)
    }
    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::native::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals_vec, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;

        let result = if D >= 3 && dims.iter().all(|&d| d >= 3) {
            // Extract the leading 3 dimensions.
            let d3 = [dims[0], dims[1], dims[2]];
            let n3 = d3[0] * d3[1] * d3[2];
            let vals3: Vec<f64> = vals_vec[..n3].iter().map(|&v| v as f64).collect();
            let out3 = ced_diffuse(&vals3, d3, &self.config);
            // Write back, converting f64 → f32.
            let mut result = vals_vec;
            for i in 0..n3 {
                result[i] = out3[i] as f32;
            }
            result
        } else {
            // CED undefined for < 3-D or tiny volumes; return input unchanged.
            vals_vec
        };

        crate::native_support::rebuild_image(result, dims, image, backend)
    }

    /// Apply the CED filter to a 3-D image, reusing scratch storage across calls.
    ///
    /// This avoids per-call allocations for the gradient, structure-tensor,
    /// smoothed structure-tensor, and divergence buffers.
    pub fn apply_with_scratch<B: Backend, const D: usize>(
        &self,
        image: &Image<B, D>,
        scratch: &mut CedScratch,
    ) -> Image<B, D> {
        let (vals_vec, dims) = extract_vec_infallible(image);

        let result = if D >= 3 && dims.iter().all(|&d| d >= 3) {
            let d3 = [dims[0], dims[1], dims[2]];
            let n3 = d3[0] * d3[1] * d3[2];
            let out3 = scratch.run_f32(&vals_vec[..n3], d3, &self.config);
            let mut result = vals_vec;
            for i in 0..n3 {
                result[i] = out3[i] as f32;
            }
            result
        } else {
            vals_vec
        };

        rebuild(result, dims, image)
    }
}
