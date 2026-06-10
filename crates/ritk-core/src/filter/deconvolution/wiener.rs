//! Wiener deconvolution filter — 2-D and 3-D.
//!
//! # Theory
//!
//! Given `g = h ∗ u + n`, the Wiener filter minimises MSE by:
//!
//! ```text
//! U(ω) = G(ω) · H*(ω) / (|H(ω)|² + K)
//! ```
//!
//! where `K = Pn / Ps` is the noise-to-signal power ratio.
//! When `K = 0`, this reduces to direct inverse filtering (noisy but exact).

use super::regularization::{apply_single_pass, WienerRule};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;

/// Wiener deconvolution filter (minimum mean-square error restoration).
///
/// Restores a degraded image `g = h ∗ u + n` given the PSF kernel `h` and
/// an estimate of the noise-to-signal power ratio `K = Pn / Ps`.
///
/// In the frequency domain:
///
/// ```text
/// U(ω) = G(ω) · H*(ω) / (|H(ω)|² + K)
/// ```
///
/// When `K = 0`, this reduces to direct inverse filtering (noisy).
/// When `K → ∞`, the output tends to zero (overly smooth).
///
/// # Use cases
/// - Motion blur correction
/// - Out-of-focus (defocus) restoration
/// - Medical image deconvolution with known PSF
///
/// # Complexity
/// O(N log N) for FFT-based execution.
pub struct WienerDeconvolution {
    /// Noise-to-signal power ratio K = Pn / Ps (default: 0.01).
    pub noise_to_signal: f32,
}

impl WienerDeconvolution {
    /// Create a new Wiener deconvolution filter with the given noise-to-signal ratio.
    pub fn new(noise_to_signal: f32) -> Self {
        Self { noise_to_signal }
    }

    /// Apply Wiener deconvolution to a 2-D image with a 2-D PSF kernel.
    pub fn apply_2d<B: Backend>(
        &self,
        image: &Image<B, 2>,
        kernel: &Image<B, 2>,
    ) -> Result<Image<B, 2>> {
        let (img_vals, img_dims) = extract_vec(image)?;
        let (ker_vals, ker_dims) = extract_vec(kernel)?;
        let out_vals = apply_single_pass::<2, _>(
            &img_vals,
            &img_dims,
            &ker_vals,
            &ker_dims,
            WienerRule {
                noise_to_signal: self.noise_to_signal,
            },
        );
        Ok(rebuild(out_vals, img_dims, image))
    }

    /// Apply Wiener deconvolution to a 3-D image with a 3-D PSF kernel.
    pub fn apply_3d<B: Backend>(
        &self,
        image: &Image<B, 3>,
        kernel: &Image<B, 3>,
    ) -> Result<Image<B, 3>> {
        let (img_vals, img_dims) = extract_vec(image)?;
        let (ker_vals, ker_dims) = extract_vec(kernel)?;
        let out_vals = apply_single_pass::<3, _>(
            &img_vals,
            &img_dims,
            &ker_vals,
            &ker_dims,
            WienerRule {
                noise_to_signal: self.noise_to_signal,
            },
        );
        Ok(rebuild(out_vals, img_dims, image))
    }
}
