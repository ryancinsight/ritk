//! Geodesic Active Contour level set segmentation (Caselles et al. 1997).
//!
//! # Mathematical Specification
//!
//! The Geodesic Active Contour (GAC) evolves a level set function φ according
//! to the PDE:
//!
//! ```text
//!   ∂φ/∂t = g(|∇I|)·(κ + ν)·|∇φ| + ∇g·∇φ
//! ```
//!
//! **Sign convention (φ < 0 inside):** The implementation uses the equivalent
//! discretised form:
//!
//! ```text
//!   ∂φ/∂t = w_c·g·κ·|∇φ| − w_p·g·|∇φ| − w_a·∇g·∇φ
//! ```
//!
//! where `w_p > 0` causes expansion (decreases φ, enlarging the φ < 0 region),
//! `w_c > 0` regularises via curvature (positive κ for convex shapes contracts),
//! and `w_a > 0` attracts the contour toward edges.
//!
//! where:
//! - **g(|∇I|) = 1 / (1 + (|∇I| / k)²)** is the edge stopping function,
//!   which approaches 0 near strong image edges and 1 in homogeneous regions.
//! - **κ = div(∇φ / |∇φ|)** is the mean curvature of the zero level set.
//! - **ν** is the balloon (propagation) force that drives expansion or
//!   contraction of the contour in the absence of edges.
//! - **∇g·∇φ** is the advection term that attracts the contour toward edges
//!   by flowing along the gradient of the edge stopping function.
//!
//! ## Discretisation
//!
//! All spatial derivatives use central finite differences with clamped boundary
//! conditions. The image gradient magnitude |∇I| is computed after optional
//! Gaussian pre-smoothing with standard deviation σ.
//!
//! The curvature κ is computed as:
//! ```text
//!   κ = div(∇φ / |∇φ|)
//! ```
//! expanded via the quotient rule into second-order central differences.
//!
//! ## Convergence
//!
//! The iteration terminates when:
//! - `max |φ^{n+1} − φ^n| / dt < tolerance`, or
//! - `iteration == max_iterations`.
//!
//! ## Output
//!
//! The final binary segmentation mask is obtained by thresholding:
//!   mask(x) = 1.0 if φ(x) < 0, else 0.0.
//!
//! ## Complexity
//!
//! - Per iteration: O(N) where N = total voxels.
//! - Gradient and edge stopping: O(N) precomputed once.
//! - Total: O(max_iterations · N).
//!
//! # References
//!
//! - Caselles, V., Kimmel, R., & Sapiro, G. (1997). "Geodesic Active Contours."
//!   *International Journal of Computer Vision*, 22(1), 61–79.
//! - Malladi, R., Sethian, J. A., & Vemuri, B. C. (1995). "Shape Modeling
//!   with Front Propagation: A Level Set Approach." *IEEE TPAMI*, 17(2).

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

use super::helpers;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Geodesic Active Contour level set segmentation.
///
/// Evolves an initial level set function toward image edges using the GAC PDE.
/// Requires an explicit initial level set (`initial_phi`) whose zero level set
/// defines the starting contour.
///
/// # Fields
///
/// | Parameter            | Symbol | Role                                      |
/// |----------------------|--------|-------------------------------------------|
/// | `propagation_weight` | ν      | Balloon force (expansion if > 0)          |
/// | `curvature_weight`   | —      | Weight on curvature regularisation term    |
/// | `advection_weight`   | —      | Weight on ∇g·∇φ edge attraction term      |
/// | `edge_k`             | k      | Edge stopping sensitivity parameter       |
/// | `sigma`              | σ      | Gaussian pre-smoothing for gradient        |
/// | `dt`                 | Δt     | Euler time step                           |
/// | `max_iterations`     | —      | Upper bound on PDE iterations             |
/// | `tolerance`          | —      | Convergence: max |Δφ|/dt < tol ⇒ stop    |
#[derive(Debug, Clone)]
pub struct GeodesicActiveContourSegmentation {
    /// Balloon (propagation) force ν. Positive expands, negative contracts.
    pub propagation_weight: f64,
    /// Weight on the curvature regularisation term κ.
    pub curvature_weight: f64,
    /// Weight on the advection term ∇g·∇φ.
    pub advection_weight: f64,
    /// Edge stopping parameter k in g(s) = 1/(1 + (s/k)²).
    pub edge_k: f64,
    /// Standard deviation of Gaussian pre-smoothing for gradient computation.
    pub sigma: f64,
    /// Euler forward time step Δt.
    pub dt: f64,
    /// Maximum number of PDE iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on max |Δφ|/dt.
    pub tolerance: f64,
}

impl GeodesicActiveContourSegmentation {
    /// Construct with default parameters.
    ///
    /// | Parameter            | Default |
    /// |----------------------|---------|
    /// | `propagation_weight` | 1.0     |
    /// | `curvature_weight`   | 1.0     |
    /// | `advection_weight`   | 1.0     |
    /// | `edge_k`             | 1.0     |
    /// | `sigma`              | 1.0     |
    /// | `dt`                 | 0.05    |
    /// | `max_iterations`     | 200     |
    /// | `tolerance`          | 1e-3    |
    pub fn new() -> Self {
        Self {
            propagation_weight: 1.0,
            curvature_weight: 1.0,
            advection_weight: 1.0,
            edge_k: 1.0,
            sigma: 1.0,
            dt: 0.05,
            max_iterations: 200,
            tolerance: 1e-3,
        }
    }

    /// Apply GAC segmentation to a 3D image with an explicit initial level set.
    ///
    /// # Arguments
    /// - : input scalar 3D image.
    /// - : initial level set function (same shape as ).
    ///   φ < 0 inside the initial contour, φ > 0 outside.
    ///
    /// # Returns
    /// Binary mask image: 1.0 where φ < 0 (inside), 0.0 elsewhere.
    ///
    /// # Errors
    /// Returns  if tensor data cannot be read as  or shapes mismatch.
    pub fn apply<B: Backend>(
        &self,
        image: &Image<B, 3>,
        initial_phi: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let phi_dims = initial_phi.shape();
        if dims != phi_dims {
            anyhow::bail!(
                "image shape {:?} and initial_phi shape {:?} must match",
                dims,
                phi_dims
            );
        }

        let device = image.data().device();

        let img_td = image.data().clone().into_data();
        let img_vals: Vec<f32> = img_td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("GAC requires f32 image data: {:?}", e))?
            .to_vec();

        let phi_td = initial_phi.data().clone().into_data();
        let phi_f32: Vec<f32> = phi_td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("GAC requires f32 phi data: {:?}", e))?
            .to_vec();

        // Convert to f64 for the entire PDE pipeline.
        let img_f64: Vec<f64> = img_vals.iter().map(|&v| v as f64).collect();
        let mut phi: Vec<f64> = phi_f32.iter().map(|&v| v as f64).collect();

        // Precompute smoothed image.
        let smoothed = if self.sigma > 0.0 {
            helpers::gaussian_smooth_3d(&img_f64, dims, self.sigma)
        } else {
            img_f64.clone()
        };

        // Precompute gradient magnitude of smoothed image.
        let grad_mag = helpers::compute_gradient_magnitude(&smoothed, dims);

        // Precompute edge stopping function g and its gradient ∇g.
        let g = helpers::compute_edge_stopping(&grad_mag, self.edge_k);
        let (g_grad_z, g_grad_y, g_grad_x) = helpers::compute_field_gradient(&g, dims);

        let n = phi.len();
        let mut kappa = vec![0.0_f64; n];
        let mut phi_new = vec![0.0_f64; n];

        for _iter in 0..self.max_iterations {
            // Compute curvature and gradient of phi.
            helpers::compute_curvature_into(&phi, dims, &mut kappa);
            let (phi_gz, phi_gy, phi_gx) = helpers::compute_field_gradient(&phi, dims);

            let mut max_change: f64 = 0.0;

            for i in 0..n {
                let grad_phi_mag =
                    (phi_gz[i] * phi_gz[i] + phi_gy[i] * phi_gy[i] + phi_gx[i] * phi_gx[i]).sqrt();

                // Curvature term (positive κ for convex → contracts): w_c·g·κ·|∇φ|
                let curv = self.curvature_weight * g[i] * kappa[i] * grad_phi_mag;

                // Propagation term (positive w_p → expansion): −w_p·g·|∇φ|
                let prop = self.propagation_weight * g[i] * grad_phi_mag;

                // Advection term (attracts toward edges): −w_a·∇g·∇φ
                let advection = self.advection_weight
                    * (g_grad_z[i] * phi_gz[i] + g_grad_y[i] * phi_gy[i] + g_grad_x[i] * phi_gx[i]);

                let dphi = self.dt * (curv - prop - advection);
                phi_new[i] = phi[i] + dphi;

                let change = dphi.abs() / self.dt;
                if change > max_change {
                    max_change = change;
                }
            }

            std::mem::swap(&mut phi, &mut phi_new);

            if max_change < self.tolerance {
                break;
            }
        }

        // Threshold: φ < 0 → inside (1.0), else outside (0.0).
        let mask: Vec<f32> = phi
            .iter()
            .map(|&v| if v < 0.0 { 1.0_f32 } else { 0.0_f32 })
            .collect();

        let tensor = Tensor::<B, 3>::from_data(TensorData::new(mask, Shape::new(dims)), &device);

        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

impl Default for GeodesicActiveContourSegmentation {
    fn default() -> Self {
        Self::new()
    }
}

// ── Test-only wrappers ─────────────────────────────────────────────────────────────────────────
//
// The existing tests call  and
// with f32 data. These thin wrappers delegate to the shared f64 helpers and
// convert back to f32, preserving the test-facing signatures without modifying
// any test function.

#[cfg(test)]
fn compute_edge_stopping(grad_mag: &[f32], k: f64) -> Vec<f32> {
    let grad_f64: Vec<f64> = grad_mag.iter().map(|&v| v as f64).collect();
    helpers::compute_edge_stopping(&grad_f64, k)
        .iter()
        .map(|&v| v as f32)
        .collect()
}

#[cfg(test)]
fn build_gaussian_kernel_1d(sigma: f64, radius: i64) -> Vec<f32> {
    helpers::build_gaussian_kernel_1d(sigma, radius as usize)
        .iter()
        .map(|&v| v as f32)
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_geodesic_active_contour.rs"]
mod tests;
