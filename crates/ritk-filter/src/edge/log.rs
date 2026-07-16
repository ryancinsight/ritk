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
use crate::recursive_gaussian::laplacian_recursive_gaussian;
use ritk_image::native::Image;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Laplacian of Gaussian (LoG) filter for 3-D images.
///
/// Computes `∇²(G_σ * I) = Σ_d ∂²/∂x_d² (G_σ * I)` via the separable Deriche
/// recursive Gaussian (second-order along each axis, zero-order along the
/// others, summed), matching ITK / SimpleITK `LaplacianRecursiveGaussian`.
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

    /// Apply the LoG filter to a 3-D Coeus-native image.
    ///
    /// Computes `∇²(G_σ * I)` via the separable Deriche IIR recursion
    /// (second-order along each axis, zero-order along the others, summed),
    /// matching ITK / SimpleITK `LaplacianRecursiveGaussian`. Spatial metadata
    /// is preserved.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as
    /// `f32`.
    pub fn apply<B>(
        &self,
        image: &Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        laplacian_recursive_gaussian(image, self.sigma.get(), backend)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_log.rs"]
mod tests;
