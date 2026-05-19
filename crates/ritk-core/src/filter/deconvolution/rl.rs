//! Richardson-Lucy iterative deconvolution — 2-D and 3-D.
//!
//! # Theory
//!
//! For Poisson noise model (photon-counting detectors), EM update:
//!
//! ```text
//! u₀ = g
//! uₖ₊₁ = uₖ · (h* ⋆ (g / (h ⋆ uₖ)))
//! ```
//!
//! where `h*` is the reversed (transposed) PSF kernel.
//!
//! # Properties
//! - Preserves non-negativity when initialized with non-negative input
//! - Preserves total flux: Σ uₖ = Σ g for all k
//! - Converges to the maximum-likelihood estimate under Poisson noise

use super::helpers::{convolve_2d, convolve_3d};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;

/// Richardson-Lucy iterative deconvolution (expectation-maximization).
///
/// For Poisson noise model (appropriate for photon-counting detectors):
///
/// ```text
/// u₀ = g  (or uniform)
/// uₖ₊₁ = uₖ · (h* ⋆ (g / (h ⋆ uₖ)))
/// ```
///
/// where `h*` is the reversed (transposed) PSF kernel.
///
/// # Properties
/// - Preserves non-negativity if initialized with non-negative values
/// - Preserves total flux: Σ uₖ = Σ g for all k
/// - Converges to the maximum-likelihood estimate under Poisson noise
///
/// # Complexity
/// O(iterations · N log N) for FFT-based convolution.
pub struct RichardsonLucyDeconvolution {
    /// Maximum number of iterations (default: 30).
    pub max_iterations: usize,
    /// Convergence tolerance for relative change (default: 1e-6).
    pub tolerance: f32,
}

impl RichardsonLucyDeconvolution {
    /// Create a new Richardson-Lucy filter with default parameters.
    pub fn new() -> Self {
        Self {
            max_iterations: 30,
            tolerance: 1e-6,
        }
    }

    /// Set the maximum number of EM iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the convergence tolerance (max relative ratio change per iteration).
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Apply Richardson-Lucy deconvolution to a 2-D image.
    pub fn apply_2d<B: Backend>(
        &self,
        image: &Image<B, 2>,
        kernel: &Image<B, 2>,
    ) -> Result<Image<B, 2>> {
        let (img_vals, img_dims) = extract_vec(image)?;
        let (ker_vals, ker_dims) = extract_vec(kernel)?;
        let [ih, iw] = img_dims;
        let [kh, kw] = ker_dims;

        // Build reversed kernel h*(-y, -x)
        let mut ker_rev = vec![0.0_f32; kh * kw];
        for ky in 0..kh {
            for kx in 0..kw {
                ker_rev[(kh - 1 - ky) * kw + (kw - 1 - kx)] = ker_vals[ky * kw + kx];
            }
        }

        let mut estimate: Vec<f32> = img_vals.clone();

        for _iter in 0..self.max_iterations {
            let forward = convolve_2d(&estimate, ih, iw, &ker_vals, kh, kw);
            let mut ratio = vec![1.0_f32; ih * iw];
            let mut max_ratio = 0.0_f32;
            for i in 0..ratio.len() {
                if forward[i] > 1e-20 {
                    let r = img_vals[i] / forward[i];
                    ratio[i] = r;
                    max_ratio = max_ratio.max((r - 1.0).abs());
                }
            }
            let correction = convolve_2d(&ratio, ih, iw, &ker_rev, kh, kw);
            for i in 0..estimate.len() {
                estimate[i] *= correction[i];
            }
            if max_ratio < self.tolerance {
                break;
            }
        }

        Ok(rebuild(estimate, img_dims, image))
    }

    /// Apply Richardson-Lucy deconvolution to a 3-D image.
    pub fn apply_3d<B: Backend>(
        &self,
        image: &Image<B, 3>,
        kernel: &Image<B, 3>,
    ) -> Result<Image<B, 3>> {
        let (img_vals, img_dims) = extract_vec(image)?;
        let (ker_vals, ker_dims) = extract_vec(kernel)?;
        let [id, ih, iw] = img_dims;
        let [kd, kh, kw] = ker_dims;

        // Build reversed kernel h*(-z, -y, -x)
        let mut ker_rev = vec![0.0_f32; kd * kh * kw];
        for kz in 0..kd {
            for ky in 0..kh {
                for kx in 0..kw {
                    ker_rev[(kd - 1 - kz) * kh * kw + (kh - 1 - ky) * kw + (kw - 1 - kx)] =
                        ker_vals[kz * kh * kw + ky * kw + kx];
                }
            }
        }

        let mut estimate: Vec<f32> = img_vals.clone();

        for _iter in 0..self.max_iterations {
            let forward = convolve_3d(&estimate, id, ih, iw, &ker_vals, kd, kh, kw);
            let mut ratio = vec![1.0_f32; id * ih * iw];
            let mut max_ratio = 0.0_f32;
            for i in 0..ratio.len() {
                if forward[i] > 1e-20 {
                    let r = img_vals[i] / forward[i];
                    ratio[i] = r;
                    max_ratio = max_ratio.max((r - 1.0).abs());
                }
            }
            let correction = convolve_3d(&ratio, id, ih, iw, &ker_rev, kd, kh, kw);
            for i in 0..estimate.len() {
                estimate[i] *= correction[i];
            }
            if max_ratio < self.tolerance {
                break;
            }
        }

        Ok(rebuild(estimate, img_dims, image))
    }
}

impl Default for RichardsonLucyDeconvolution {
    fn default() -> Self {
        Self::new()
    }
}
