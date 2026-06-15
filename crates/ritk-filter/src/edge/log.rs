//! Laplacian of Gaussian (LoG) filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! The Laplacian of Gaussian is defined as:
//!
//!   LoG(x) = ∇²G_σ * I = G_σ * ∇²I
//!
//! where G_σ is the Gaussian kernel with standard deviation σ and ∇² is the
//! Laplacian operator. By the commutativity of convolution and the linearity
//! of differentiation, the two orderings are equivalent.
//!
//! The closed-form 3-D LoG kernel is:
//!
//!   LoG(r) = −(1/(πσ⁴)) · [1 − r²/(2σ²)] · exp(−r²/(2σ²))
//!
//! where r² = x² + y² + z². This implementation uses the separable approach:
//! first apply Gaussian smoothing (via `GaussianFilter`), then compute the
//! discrete Laplacian (via `LaplacianFilter`). This reuses existing verified
//! components and avoids constructing a large 3-D kernel.
//!
//! # Properties
//!
//! - **LoG of a constant field is zero**: ∇²(constant) = 0.
//! - **Zero-crossing detection**: Edges correspond to zero crossings of the
//!   LoG response.
//! - **Blob detection**: The LoG response is negative at the centre of a
//!   bright Gaussian blob with matching scale, enabling blob detection via
//!   scale-space extrema.
//!
//! # Complexity
//!
//! O(N) for the Laplacian stage, plus the cost of the separable Gaussian
//! convolution (O(D · N · k) where k is the kernel half-width per dimension).
//!
//! # References
//!
//! - Marr, D. & Hildreth, E. (1980). Theory of edge detection. *Proceedings
//!   of the Royal Society of London B*, 207(1167), pp. 187–217.
//! - Lindeberg, T. (1994). *Scale-Space Theory in Computer Vision*. Springer.

use super::GaussianSigma;
use crate::edge::LaplacianFilter;
use crate::GaussianFilter;
use burn::tensor::backend::Backend;
use ritk_image::Image;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Laplacian of Gaussian (LoG) filter for 3-D images.
///
/// Computes ∇²(G_σ * I) by first applying Gaussian smoothing with standard
/// deviation σ in each dimension (respecting physical spacing), then computing
/// the discrete Laplacian via second-order finite differences.
#[derive(Debug, Clone)]
pub struct LaplacianOfGaussianFilter {
    /// Standard deviation of the Gaussian in physical units (mm).
    sigma: GaussianSigma,
}

impl LaplacianOfGaussianFilter {
    /// Create a new LoG filter with the given sigma (physical units).
    pub fn new(sigma: GaussianSigma) -> Self {
        Self { sigma }
    }

    /// Set the Gaussian sigma.
    pub fn with_sigma(mut self, sigma: GaussianSigma) -> Self {
        self.sigma = sigma;
        self
    }

    /// Apply the LoG filter to a 3-D image.
    ///
    /// Computes G_σ * ∇²I by:
    /// 1. Smoothing the image with a Gaussian of standard deviation σ.
    /// 2. Computing the discrete Laplacian of the smoothed image.
    ///
    /// The output has the same shape and spatial metadata as the input.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as
    /// `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let sp = *image.spacing();

        // Stage 1: Gaussian smoothing
        let gauss = GaussianFilter::<B>::new(vec![self.sigma, self.sigma, self.sigma]);
        let smoothed = gauss.apply(image);

        // Stage 2: Laplacian via second-order finite differences
        let laplacian = LaplacianFilter::new(sp);
        laplacian.apply(&smoothed)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_log.rs"]
mod tests;
