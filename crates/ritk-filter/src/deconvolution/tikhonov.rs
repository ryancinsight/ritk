//! Tikhonov-regularized deconvolution filter — 2-D and 3-D.
//!
//! # Theory
//!
//! Constant-regularised inverse filter, matching ITK's
//! `TikhonovDeconvolutionImageFilter`:
//!
//! ```text
//! U(ω) = G(ω) · H*(ω) / (|H(ω)|² + λ)
//! ```
//!
//! `λ` trades inversion sharpness against noise amplification uniformly across
//! frequency. For a noise-/signal-adaptive regularisation use the Wiener filter
//! ([`super::WienerDeconvolution`]).

use super::regularization::{apply_single_pass, TikhonovRule};
use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Tikhonov-regularized deconvolution (constant ridge term in the frequency
/// domain), matching ITK's `TikhonovDeconvolutionImageFilter`:
///
/// ```text
/// U(ω) = G(ω) · H*(ω) / (|H(ω)|² + λ)
/// ```
///
/// # Comparison to Wiener
/// Tikhonov adds a constant `λ`; Wiener adds a frequency-adaptive
/// noise-to-signal term estimated from the input power spectrum.
///
/// # Complexity
/// O(N log N).
pub struct TikhonovDeconvolution {
    /// Regularization parameter λ (default: 0.01).
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
            TikhonovRule {
                lambda: self.lambda,
            },
        );
        Ok(rebuild(out_vals, img_dims, image))
    }
}
