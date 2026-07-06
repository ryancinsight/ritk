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

use super::regularization::{
    apply_iterative, IterativeAlgorithm, IterativeParams, LandweberProjection,
    DEFAULT_ITERATIVE_TOLERANCE,
};
use anyhow::Result;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

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
    /// Per-iteration projection constraint (default: [`LandweberProjection::None`]).
    /// Set to [`LandweberProjection::NonNegative`] for ITK
    /// `ProjectedLandweberDeconvolutionImageFilter` behaviour.
    pub projection: LandweberProjection,
}

impl LandweberDeconvolution {
    /// Create a new Landweber filter with default parameters.
    pub fn new() -> Self {
        Self {
            step_size: 0.1,
            max_iterations: 100,
            tolerance: DEFAULT_ITERATIVE_TOLERANCE,
            projection: LandweberProjection::None,
        }
    }

    /// Set the per-iteration projection constraint (non-negativity for the
    /// projected Landweber variant).
    pub fn with_projection(mut self, projection: LandweberProjection) -> Self {
        self.projection = projection;
        self
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

    /// Apply Landweber deconvolution to a D-dimensional image.
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
                algorithm: IterativeAlgorithm::Landweber {
                    step_size: self.step_size,
                    projection: self.projection,
                },
            },
        );
        Ok(rebuild(out_vals, img_dims, image))
    }
}

impl Default for LandweberDeconvolution {
    fn default() -> Self {
        Self::new()
    }
}
