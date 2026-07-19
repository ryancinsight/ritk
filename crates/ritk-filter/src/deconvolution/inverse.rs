//! Direct inverse-filter deconvolution — 2-D and 3-D.
//!
//! # Theory
//!
//! The plain inverse filter, matching ITK's `InverseDeconvolutionImageFilter`:
//!
//! ```text
//! U(ω) = G(ω) / H(ω) = G(ω)·H*(ω) / |H(ω)|²   if |H(ω)| >= τ, else 0
//! ```
//!
//! It divides directly by the optical transfer function, zeroing frequencies
//! whose magnitude falls below `τ` (the kernel-zero-magnitude threshold) to avoid
//! dividing by near-zero OTF values. Unlike [`super::TikhonovDeconvolution`] it
//! adds no ridge term, so it is the sharpest (and noisiest) of the linear
//! restorations.

use super::regularization::{apply_single_pass, InverseRule};
use anyhow::Result;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Direct inverse-filter deconvolution, matching ITK's
/// `InverseDeconvolutionImageFilter`:
///
/// ```text
/// U(ω) = G(ω) / H(ω)   if |H(ω)| >= τ, else 0
/// ```
///
/// # Complexity
/// O(N log N).
pub struct InverseDeconvolution {
    /// Magnitude threshold `τ` below which an OTF frequency is zeroed
    /// (ITK `KernelZeroMagnitudeThreshold`, SimpleITK default 1e-4).
    pub kernel_zero_magnitude_threshold: f32,
}

impl InverseDeconvolution {
    /// Create a new inverse-filter deconvolution with the given zero-magnitude
    /// threshold.
    pub fn new(kernel_zero_magnitude_threshold: f32) -> Self {
        Self {
            kernel_zero_magnitude_threshold,
        }
    }

    /// Apply inverse-filter deconvolution to a D-dimensional image.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<f32, B, D>,
        kernel: &Image<f32, B, D>,
    ) -> Result<Image<f32, B, D>> {
        let (img_vals, img_dims) = extract_vec(image)?;
        let (ker_vals, ker_dims) = extract_vec(kernel)?;
        let out_vals = apply_single_pass::<D, _>(
            &img_vals,
            &img_dims,
            &ker_vals,
            &ker_dims,
            InverseRule {
                kernel_zero_magnitude_threshold: self.kernel_zero_magnitude_threshold,
            },
        );
        Ok(rebuild(out_vals, img_dims, image))
    }

    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        kernel: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (img_vals, img_dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let (ker_vals, ker_dims) = ritk_tensor_ops::native::extract_image_vec(kernel)?;
        let out_vals = apply_single_pass::<D, _>(
            &img_vals,
            &img_dims,
            &ker_vals,
            &ker_dims,
            InverseRule {
                kernel_zero_magnitude_threshold: self.kernel_zero_magnitude_threshold,
            },
        );
        crate::native_support::rebuild_image(out_vals, img_dims, image, backend)
    }
}

impl Default for InverseDeconvolution {
    fn default() -> Self {
        Self::new(1e-4)
    }
}
