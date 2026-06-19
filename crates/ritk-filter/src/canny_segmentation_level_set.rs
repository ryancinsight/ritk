//! Canny-edge-guided level set segmentation filter for 3-D volumes.
//!
//! # Algorithm
//!
//! Approximates `itk::CannySegmentationLevelSetImageFilter`. The level set is
//! evolved toward image edges detected via Gaussian-gradient edge strength.
//!
//! ## Steps
//!
//! 1. **Gaussian pre-smoothing** of `feature_image` with σ = sqrt(`canny_variance`).
//! 2. **Gradient magnitude** |∇I_smooth| of the smoothed feature image.
//! 3. **Edge potential**: F(x) = exp(−|∇I_smooth(x)|² / `canny_threshold`).
//!    - F ≈ 1 in homogeneous regions (small gradient → level set propagates freely).
//!    - F ≈ 0 near strong edges (large gradient → level set slows / stops).
//! 4. **Level set evolution** (Euler forward):
//!
//! ```text
//! ∂φ/∂t = F(x) · [ curvature_scaling · κ · |∇φ|
//!                  − propagation_scaling · |∇φ| ]
//! ```
//!
//! where κ = div(∇φ / |∇φ|) is the mean curvature.
//!
//! ## Sign convention
//!
//! Follows ITK's convention: **φ < 0 inside** the evolving region, φ > 0 outside.
//!
//! ## Curvature term sign
//!
//! With φ < 0 = inside: the curvature κ of a convex sphere is positive
//! (outward normal pointing away from centre). A positive `curvature_scaling`
//! term drives φ more positive → the inside region contracts, regularising the
//! contour toward a smaller, smoother shape.
//!
//! ## Propagation term sign
//!
//! A positive `propagation_scaling` drives φ more negative → the inside region
//! expands. A negative value contracts it.
//!
//! ## Convergence
//!
//! Iteration stops when `max|Δφ| / dt < max_rms_error` or `number_of_iterations`
//! is reached.
//!
//! ## Complexity
//!
//! O(number_of_iterations · N) where N = total voxels.
//! The Gaussian smoothing and gradient computation are O(N) pre-processing steps.
//!
//! # References
//!
//! - Whitaker, R. T. (1998). "A Level-Set Approach to 3D Reconstruction from
//!   Range Data." *IJCV*, 29(3), 203–231.
//! - Lorigo, L. M. et al. (2001). "CURVES: Curve evolution for vessel
//!   segmentation." *Medical Image Analysis*, 5(3), 195–206.

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

use crate::level_set_helpers::{
    compute_curvature_into, compute_field_gradient, compute_gradient_magnitude, gaussian_smooth,
};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Canny-edge-guided level set segmentation for 3-D volumes.
///
/// Returns the evolved level set function φ (not a binary mask). Threshold at
/// φ = 0: voxels with φ < 0 belong to the segmented region.
///
/// # Default parameters
///
/// | Field | Default |
/// |-------|---------|
/// | `canny_threshold` | 1.0 |
/// | `canny_variance` | 1.0 |
/// | `number_of_iterations` | 100 |
/// | `max_rms_error` | 0.01 |
/// | `propagation_scaling` | 0.0 |
/// | `curvature_scaling` | 1.0 |
/// | `dt` | 0.0625 |
#[derive(Debug, Clone)]
pub struct CannySegmentationLevelSet {
    /// Threshold τ for the edge potential F = exp(−|∇I|² / τ).
    pub canny_threshold: f32,
    /// Gaussian variance σ² applied to the feature image before gradient computation.
    /// Actual kernel sigma = sqrt(canny_variance).
    pub canny_variance: f32,
    /// Maximum number of PDE iterations.
    pub number_of_iterations: usize,
    /// Convergence criterion: stop when max|Δφ| / dt < max_rms_error.
    pub max_rms_error: f32,
    /// Propagation (balloon) force scaling. Positive → expansion (φ more negative).
    pub propagation_scaling: f32,
    /// Curvature regularisation weight. Positive → contraction for convex shapes.
    pub curvature_scaling: f32,
    /// Euler forward time step Δt.
    pub dt: f32,
}

impl Default for CannySegmentationLevelSet {
    fn default() -> Self {
        Self {
            canny_threshold: 1.0,
            canny_variance: 1.0,
            number_of_iterations: 100,
            max_rms_error: 0.01,
            propagation_scaling: 0.0,
            curvature_scaling: 1.0,
            dt: 0.0625,
        }
    }
}

impl CannySegmentationLevelSet {
    /// Evolve a level set toward Canny edges in the feature image.
    ///
    /// # Arguments
    /// - `initial_level_set`: φ₀ with **φ < 0 inside** the region of interest.
    /// - `feature_image`: u₀ — the image from which edges are derived.
    ///   Must have the same shape as `initial_level_set`.
    ///
    /// # Returns
    /// The evolved level set φ as a floating-point image.
    ///
    /// # Errors
    /// Returns `Err` if tensor extraction fails or shapes differ.
    pub fn apply<B: Backend>(
        &self,
        initial_level_set: &Image<B, 3>,
        feature_image: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = initial_level_set.shape();
        if dims != feature_image.shape() {
            anyhow::bail!(
                "initial_level_set shape {:?} and feature_image shape {:?} must match",
                dims,
                feature_image.shape()
            );
        }

        let (phi_init, _) = extract_vec(initial_level_set)?;
        let (feat, _) = extract_vec(feature_image)?;

        // Promote to f64 for PDE accuracy.
        let mut phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();
        let feat_f64: Vec<f64> = feat.iter().map(|&v| v as f64).collect();

        self.evolve(&mut phi, &feat_f64, dims);

        let result: Vec<f32> = phi.iter().map(|&v| v as f32).collect();
        Ok(rebuild(result, dims, initial_level_set))
    }
}

// ── Core PDE ───────────────────────────────────────────────────────────────────

impl CannySegmentationLevelSet {
    fn evolve(&self, phi: &mut [f64], feat: &[f64], dims: [usize; 3]) {
        let n = dims[0] * dims[1] * dims[2];
        let dt = self.dt as f64;
        let curv_w = self.curvature_scaling as f64;
        let prop_w = self.propagation_scaling as f64;
        let thresh = (self.canny_threshold as f64).max(1e-20);
        let tol = self.max_rms_error as f64;

        // ── Pre-processing: edge potential F (computed once) ─────────────────
        // sigma = sqrt(canny_variance); skip smoothing if variance ≤ 0.
        let sigma = (self.canny_variance as f64).sqrt();
        let smoothed = gaussian_smooth(feat, dims, sigma);
        let grad_mag = compute_gradient_magnitude(&smoothed, dims);
        // F(x) = exp(−|∇I_smooth(x)|² / threshold)
        let f_edge: Vec<f64> = grad_mag
            .iter()
            .map(|&g| (-(g * g) / thresh).exp())
            .collect();

        let mut kappa = vec![0.0_f64; n];

        for _ in 0..self.number_of_iterations {
            // Curvature κ = div(∇φ/|∇φ|).
            compute_curvature_into(phi, dims, &mut kappa);

            // Gradient of φ for |∇φ| term.
            let (gz, gy, gx) = compute_field_gradient(phi, dims);

            let mut max_change = 0.0_f64;

            for i in 0..n {
                let grad_phi_mag = (gz[i] * gz[i] + gy[i] * gy[i] + gx[i] * gx[i]).sqrt();

                // ∂φ/∂t = F · (curv_w · κ · |∇φ| − prop_w · |∇φ|)
                let dphi =
                    dt * f_edge[i] * (curv_w * kappa[i] * grad_phi_mag - prop_w * grad_phi_mag);

                phi[i] += dphi;

                let abs_change = dphi.abs() / dt;
                if abs_change > max_change {
                    max_change = abs_change;
                }
            }

            if max_change < tol {
                break;
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_canny_segmentation_level_set.rs"]
mod tests;
