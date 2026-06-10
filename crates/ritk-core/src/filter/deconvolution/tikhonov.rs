//! Tikhonov-regularized deconvolution filter — 2-D and 3-D.
//!
//! # Theory
//!
//! Minimises `||g − h ∗ u||² + λ||L ∗ u||²` where L is the Laplacian operator.
//!
//! In the frequency domain:
//!
//! ```text
//! U(ω) = G(ω) · H*(ω) / (|H(ω)|² + λ · |L(ω)|²)
//! ```
//!
//! 2-D Laplacian: `|L(ω)|² = (4 − 2cos(ωx) − 2cos(ωy))²`
//! 3-D Laplacian: `|L(ω)|² = (6 − 2cos(ωx) − 2cos(ωy) − 2cos(ωz))²`
//!
//! Higher λ → smoother output (higher regularization strength).

use super::regularization::{apply_single_pass, TikhonovRule};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;

/// Tikhonov-regularized deconvolution (ridge regression in frequency domain).
///
/// Minimizes `||g − h ∗ u||² + λ||L ∗ u||²` where L is the Laplacian operator.
///
/// In the frequency domain:
///
/// ```text
/// U(ω) = G(ω) · H*(ω) / (|H(ω)|² + λ · |L(ω)|²)
/// ```
///
/// where `|L(ω)|² = (4 − 2cos(ωx) − 2cos(ωy))²` for 2-D discrete Laplacian
/// and `|L(ω)|² = (6 − 2cos(ωx) − 2cos(ωy) − 2cos(ωz))²` for 3-D.
///
/// # Comparison to Wiener
/// Tikhonov uses a smoothness prior (λ|Lu|²) rather than a noise-to-signal
/// ratio. It tends to produce smoother restorations.
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

    /// Apply Tikhonov deconvolution to a 2-D image.
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
            TikhonovRule {
                lambda: self.lambda,
            },
        );
        Ok(rebuild(out_vals, img_dims, image))
    }

    /// Apply Tikhonov deconvolution to a 3-D image.
    ///
    /// Uses the 3-D discrete Laplacian:
    /// `|L(ω)|² = (6 − 2cos(ωx) − 2cos(ωy) − 2cos(ωz))²`
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
            TikhonovRule {
                lambda: self.lambda,
            },
        );
        Ok(rebuild(out_vals, img_dims, image))
    }
}
