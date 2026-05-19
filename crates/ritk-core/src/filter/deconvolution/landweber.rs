//! Landweber iterative deconvolution — 2-D and 3-D.
//!
//! # Theory
//!
//! Minimizes `||g − h ∗ u||²` via steepest descent:
//!
//! ```text
//! u₀ = g
//! uₖ₊₁ = uₖ + α · h* ⋆ (g − h ⋆ uₖ)
//! ```
//!
//! # Convergence condition
//! α must satisfy `0 < α < 2 / σ_max²` where σ_max is the largest singular
//! value of the convolution operator H (≈ max|H(ω)| in frequency domain).
//!
//! # Properties
//! - Guaranteed convergence for sufficiently small α
//! - Slower than conjugate-gradient methods but simple and analyzable

use super::helpers::{convolve_2d, convolve_3d};
use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;

/// Landweber iterative deconvolution (gradient descent).
///
/// Minimizes `||g − h ∗ u||²` via steepest descent:
///
/// ```text
/// u₀ = g
/// uₖ₊₁ = uₖ + α · h* ⋆ (g − h ⋆ uₖ)
/// ```
///
/// where α must satisfy `0 < α < 2 / σ_max²` for convergence.
///
/// # Properties
/// - Simple to implement and analyze
/// - Slower convergence than conjugate gradient methods
/// - Guaranteed convergence for small enough α
///
/// # Complexity
/// O(iterations · N log N).
pub struct LandweberDeconvolution {
    /// Step size α (default: 0.1).
    pub step_size: f32,
    /// Maximum number of iterations (default: 100).
    pub max_iterations: usize,
    /// Convergence tolerance (default: 1e-6).
    pub tolerance: f32,
}

impl LandweberDeconvolution {
    /// Create a new Landweber filter with default parameters.
    pub fn new() -> Self {
        Self {
            step_size: 0.1,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }

    /// Set the gradient descent step size α.
    pub fn with_step_size(mut self, alpha: f32) -> Self {
        self.step_size = alpha;
        self
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the convergence tolerance (max absolute residual).
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Apply Landweber deconvolution to a 2-D image.
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
        let alpha = self.step_size;

        for _iter in 0..self.max_iterations {
            let forward = convolve_2d(&estimate, ih, iw, &ker_vals, kh, kw);
            let mut residual = vec![0.0_f32; ih * iw];
            let mut max_residual = 0.0_f32;
            for i in 0..residual.len() {
                let r = img_vals[i] - forward[i];
                residual[i] = r;
                max_residual = max_residual.max(r.abs());
            }
            let correction = convolve_2d(&residual, ih, iw, &ker_rev, kh, kw);
            for i in 0..estimate.len() {
                estimate[i] += alpha * correction[i];
            }
            if max_residual < self.tolerance {
                break;
            }
        }

        Ok(rebuild(estimate, img_dims, image))
    }

    /// Apply Landweber deconvolution to a 3-D image.
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
        let alpha = self.step_size;

        for _iter in 0..self.max_iterations {
            let forward = convolve_3d(&estimate, id, ih, iw, &ker_vals, kd, kh, kw);
            let mut residual = vec![0.0_f32; id * ih * iw];
            let mut max_residual = 0.0_f32;
            for i in 0..residual.len() {
                let r = img_vals[i] - forward[i];
                residual[i] = r;
                max_residual = max_residual.max(r.abs());
            }
            let correction = convolve_3d(&residual, id, ih, iw, &ker_rev, kd, kh, kw);
            for i in 0..estimate.len() {
                estimate[i] += alpha * correction[i];
            }
            if max_residual < self.tolerance {
                break;
            }
        }

        Ok(rebuild(estimate, img_dims, image))
    }
}

impl Default for LandweberDeconvolution {
    fn default() -> Self {
        Self::new()
    }
}
