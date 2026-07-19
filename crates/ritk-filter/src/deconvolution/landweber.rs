//! Landweber iterative deconvolution â€” 2-D and 3-D.
//!
//! # Theory
//!
//! Minimizes `||g âˆ’ h âˆ— u||Â²` via steepest descent:
//!
//! ```text
//! uâ‚€ = g
//! uâ‚–â‚Šâ‚ = uâ‚– + Î± Â· h* â‹† (g âˆ’ h â‹† uâ‚–)
//! ```
//!
//! # Convergence condition
//! Î± must satisfy `0 < Î± < 2 / Ïƒ_maxÂ²` where Ïƒ_max is the largest singular
//! value of the convolution operator H (â‰ˆ max|H(Ï‰)| in frequency domain).
//!
//! # Properties
//! - Guaranteed convergence for sufficiently small Î±
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
/// Minimizes `||g âˆ’ h âˆ— u||Â²` via steepest descent:
///
/// ```text
/// uâ‚€ = g
/// uâ‚–â‚Šâ‚ = uâ‚– + Î± Â· h* â‹† (g âˆ’ h â‹† uâ‚–)
/// ```
///
/// where Î± must satisfy `0 < Î± < 2 / Ïƒ_maxÂ²` for convergence.
///
/// # Properties
/// - Simple to implement and analyze
/// - Slower convergence than conjugate gradient methods
/// - Guaranteed convergence for small enough Î±
///
/// # Complexity
/// O(iterations Â· N log N).
pub struct LandweberDeconvolution {
    /// Step size Î± (default: 0.1).
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

    /// Set the gradient descent step size Î±.
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
        image: &Image<f32, B, D>,
        kernel: &Image<f32, B, D>,
    ) -> Result<Image<f32, B, D>> {
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

    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        kernel: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (img_vals, img_dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let (ker_vals, ker_dims) = ritk_tensor_ops::native::extract_image_vec(kernel)?;
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
        crate::native_support::rebuild_image(out_vals, img_dims, image, backend)
    }
}

impl Default for LandweberDeconvolution {
    fn default() -> Self {
        Self::new()
    }
}
