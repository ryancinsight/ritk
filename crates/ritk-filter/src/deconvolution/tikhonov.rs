//! Tikhonov-regularized deconvolution filter â€” 2-D and 3-D.
//!
//! # Theory
//!
//! Constant-regularised inverse filter, matching ITK's
//! `TikhonovDeconvolutionImageFilter`:
//!
//! ```text
//! U(Ï‰) = G(Ï‰) Â· H*(Ï‰) / (|H(Ï‰)|Â² + Î»)
//! ```
//!
//! `Î»` trades inversion sharpness against noise amplification uniformly across
//! frequency. For a noise-/signal-adaptive regularisation use the Wiener filter
//! ([`super::WienerDeconvolution`]).

use super::regularization::{apply_single_pass, TikhonovRule};
use anyhow::Result;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Tikhonov-regularized deconvolution (constant ridge term in the frequency
/// domain), matching ITK's `TikhonovDeconvolutionImageFilter`:
///
/// ```text
/// U(Ï‰) = G(Ï‰) Â· H*(Ï‰) / (|H(Ï‰)|Â² + Î»)
/// ```
///
/// # Comparison to Wiener
/// Tikhonov adds a constant `Î»`; Wiener adds a frequency-adaptive
/// noise-to-signal term estimated from the input power spectrum.
///
/// # Complexity
/// O(N log N).
pub struct TikhonovDeconvolution {
    /// Regularization parameter Î» (default: 0.01).
    pub lambda: f32,
}

impl TikhonovDeconvolution {
    /// Create a new Tikhonov deconvolution filter with the given regularization parameter.
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }

    /// Apply Tikhonov deconvolution to a D-dimensional image.
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
            TikhonovRule {
                lambda: self.lambda,
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
            TikhonovRule {
                lambda: self.lambda,
            },
        );
        crate::native_support::rebuild_image(out_vals, img_dims, image, backend)
    }
}
