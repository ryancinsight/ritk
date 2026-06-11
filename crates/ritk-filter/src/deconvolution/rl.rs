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

use super::regularization::{apply_iterative, IterativeAlgorithm, IterativeParams};
use ritk_core::filter::ops::{extract_vec, rebuild};
use ritk_image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;

/// Richardson-Lucy iterative deconvolution (expectation-maximization).
///
/// For Poisson noise model (appropriate for photon-counting detectors):
///
/// ```text
/// u₀ = g (or uniform)
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

    /// Apply Richardson-Lucy deconvolution to a D-dimensional image.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<B, D>,
        kernel: &Image<B, D>,
    ) -> Result<Image<B, D>> {
        let (img_vals, img_dims) = extract_vec(image)?;
        let (ker_vals, ker_dims) = extract_vec(kernel)?;
        let out_vals = apply_iterative::<D>(
            &img_vals,
            &img_dims,
            &IterativeParams {
                ker_vals: &ker_vals,
                ker_dims: &ker_dims,
                max_iterations: self.max_iterations,
                tolerance: self.tolerance,
                algorithm: IterativeAlgorithm::RichardsonLucy,
            },
        );
        Ok(rebuild(out_vals, img_dims, image))
    }

    /// Apply Richardson-Lucy deconvolution to a 2-D image.
    #[deprecated(since = "0.57.0", note = "use `apply::<2>` instead")]
    pub fn apply_2d<B: Backend>(
        &self,
        image: &Image<B, 2>,
        kernel: &Image<B, 2>,
    ) -> Result<Image<B, 2>> {
        self.apply(image, kernel)
    }

    /// Apply Richardson-Lucy deconvolution to a 3-D image.
    #[deprecated(since = "0.57.0", note = "use `apply::<3>` instead")]
    pub fn apply_3d<B: Backend>(
        &self,
        image: &Image<B, 3>,
        kernel: &Image<B, 3>,
    ) -> Result<Image<B, 3>> {
        self.apply(image, kernel)
    }
}

impl Default for RichardsonLucyDeconvolution {
    fn default() -> Self {
        Self::new()
    }
}
