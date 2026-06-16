//! Wiener deconvolution filter — 2-D and 3-D.
//!
//! # Theory
//!
//! Given `g = h ∗ u + n`, the Wiener filter minimises MSE by, per ITK's
//! `WienerDeconvolutionImageFilter`:
//!
//! ```text
//! U(ω) = G(ω) · H*(ω) / ( |H(ω)|² + Pn / (|G(ω)|² − Pn) )
//! ```
//!
//! where `Pn` is the noise power spectral density (the `noise_variance`
//! parameter). The regularisation is frequency-adaptive: it estimates the signal
//! power as `|G(ω)|² − Pn`, so weak-signal frequencies are suppressed more. For a
//! *constant*-regularisation inverse filter use [`super::TikhonovDeconvolution`].

use super::regularization::{apply_single_pass, WienerRule};
use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Wiener deconvolution filter (minimum mean-square error restoration),
/// matching ITK's `WienerDeconvolutionImageFilter`.
///
/// Restores a degraded image `g = h ∗ u + n` given the PSF kernel `h` and the
/// noise power spectral density `Pn`:
///
/// ```text
/// U(ω) = G(ω) · H*(ω) / ( |H(ω)|² + Pn / (|G(ω)|² − Pn) )
/// ```
///
/// # Use cases
/// - Motion blur correction
/// - Out-of-focus (defocus) restoration
/// - Medical image deconvolution with known PSF
///
/// # Complexity
/// O(N log N) for FFT-based execution.
pub struct WienerDeconvolution {
    /// Noise power spectral density `Pn` (ITK `NoiseVariance`, default: 0.01).
    pub noise_variance: f32,
}

impl WienerDeconvolution {
    /// Create a new Wiener deconvolution filter with the given noise variance.
    pub fn new(noise_variance: f32) -> Self {
        Self { noise_variance }
    }

    /// Apply Wiener deconvolution to a D-dimensional image with a D-dimensional PSF kernel.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<B, D>,
        kernel: &Image<B, D>,
    ) -> Result<Image<B, D>> {
        let (img_vals, img_dims) = extract_vec(image)?;
        let (ker_vals, ker_dims) = extract_vec(kernel)?;
        let out_vals = apply_single_pass::<D, _>(
            &img_vals,
            &img_dims,
            &ker_vals,
            &ker_dims,
            WienerRule {
                noise_variance: self.noise_variance,
            },
        );
        Ok(rebuild(out_vals, img_dims, image))
    }
}
