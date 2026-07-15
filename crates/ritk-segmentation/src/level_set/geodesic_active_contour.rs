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
//! - `RMS(Δφ) = sqrt(sum(Δφ²) / N) < tolerance` (matches ITK's
//!   `FiniteDifferenceImageFilter::GetRMSChange()` criterion), or
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

use super::helpers;
use ritk_filter::edge::GaussianSigma;
use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

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
/// | `tolerance`          | —      | Convergence: RMS(Δφ) < tol ⇒ stop        |
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
    /// Must be > 0.
    pub sigma: GaussianSigma,
    /// Euler forward time step Δt.
    pub dt: f64,
    /// Maximum number of PDE iterations.
    pub max_iterations: usize,
    /// Convergence: RMS(Δφ) < tol ⇒ stop (matches ITK's
    /// `FiniteDifferenceImageFilter::GetRMSChange()` criterion).
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
            sigma: GaussianSigma::new_unchecked(1.0),
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
        let [nz, ny, nx] = dims;
        let phi_dims = initial_phi.shape();
        if dims != phi_dims {
            anyhow::bail!(
                "image shape {:?} and initial_phi shape {:?} must match",
                dims,
                phi_dims
            );
        }
        let device = image.data().device();

        let (img_vals, _) = extract_vec(image)?;
        let (phi_init, _) = extract_vec(initial_phi)?;
        // Convert to f64 for the entire PDE pipeline.
        let img_wide: Vec<f64> = img_vals.iter().map(|&v| v as f64).collect();
        let mut phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();

        // Precompute smoothed image.
        let smoothed = helpers::smooth_or_borrow(&img_wide, dims, self.sigma.get());

        // Precompute gradient magnitude of smoothed image.
        let grad_mag = helpers::compute_gradient_magnitude(&smoothed, dims);

        // Precompute edge stopping function g and its gradient ∇g.
        let g = helpers::compute_edge_stopping(&grad_mag, self.edge_k);
        let (g_grad_z, g_grad_y, g_grad_x) = helpers::compute_field_gradient(&g, dims);

        let n = phi.len();
        let mut kappa = vec![0.0_f64; n];
        let mut phi_new = phi.clone();
        // SEG-01: pre-allocate per-iteration scratch buffers outside the loop so
        // that compute_field_gradient_into / upwind_advection_into reuse them,
        // eliminating 4 × N×8 heap allocations per PDE iteration.
        let mut phi_gz = vec![0.0_f64; n];
        let mut phi_gy = vec![0.0_f64; n];
        let mut phi_gx = vec![0.0_f64; n];
        let mut adv = vec![0.0_f64; n];
        let mut sum_sqs = vec![0.0_f64; nz];

        for _iter in 0..self.max_iterations {
            // Compute curvature and gradient of phi.
            helpers::compute_curvature_into(&phi, dims, &mut kappa);
            helpers::compute_field_gradient_into(&phi, dims, &mut phi_gz, &mut phi_gy, &mut phi_gx);
            // Upwind discretisation of the advection (transport) term ∇g·∇φ;
            // central differencing it is unstable and leaks the front past edges.
            helpers::upwind_advection_into(&phi, dims, &g_grad_z, &g_grad_y, &g_grad_x, &mut adv);

            let slice_len = ny * nx;

            helpers::evolve_slices_with_metric(
                &mut phi_new,
                &mut sum_sqs,
                slice_len,
                |iz, phi_new_s| {
                    let base = iz * slice_len;
                    let mut local_sum_sq = 0.0_f64;
                    for (i, phi_new_val) in phi_new_s.iter_mut().enumerate() {
                        let idx = base + i;
                        let grad_phi_mag = (phi_gz[idx] * phi_gz[idx]
                            + phi_gy[idx] * phi_gy[idx]
                            + phi_gx[idx] * phi_gx[idx])
                            .sqrt();

                        // Curvature term (positive κ for convex → contracts): w_c·g·κ·|∇φ|
                        let curv = self.curvature_weight * g[idx] * kappa[idx] * grad_phi_mag;

                        // Propagation term (positive w_p → expansion): −w_p·g·|∇φ|
                        let prop = self.propagation_weight * g[idx] * grad_phi_mag;

                        // Advection term (attracts the front toward edges): +w_a·∇g·∇φ,
                        // upwind-discretised for stability.
                        let advection = self.advection_weight * adv[idx];

                        let dphi = self.dt * (curv - prop + advection);
                        *phi_new_val = phi[idx] + dphi;

                        // Accumulate squared change for RMS convergence criterion.
                        local_sum_sq += dphi * dphi;
                    }
                    local_sum_sq
                },
            );

            std::mem::swap(&mut phi, &mut phi_new);

            // ITK RMS criterion: sqrt(sum(Δφ²) / N) < tolerance.
            let sum_sq: f64 = sum_sqs.iter().sum();
            let rms = (sum_sq / n as f64).sqrt();
            if rms < self.tolerance {
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

    /// Apply GAC segmentation to Coeus-native images.
    ///
    /// # Errors
    ///
    /// Returns the same shape-validation errors as [`Self::apply`], plus an
    /// error when either tensor is not host-addressable/contiguous or the native
    /// output image cannot be constructed.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        initial_phi: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let dims = image.shape();
        let [nz, ny, nx] = dims;
        let phi_dims = initial_phi.shape();
        if dims != phi_dims {
            anyhow::bail!(
                "image shape {:?} and initial_phi shape {:?} must match",
                dims,
                phi_dims
            );
        }

        let img_vals = image.data_slice()?;
        let phi_init = initial_phi.data_slice()?;
        let img_wide: Vec<f64> = img_vals.iter().map(|&v| v as f64).collect();
        let mut phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();

        let smoothed = helpers::smooth_or_borrow(&img_wide, dims, self.sigma.get());
        let grad_mag = helpers::compute_gradient_magnitude(&smoothed, dims);
        let g = helpers::compute_edge_stopping(&grad_mag, self.edge_k);
        let (g_grad_z, g_grad_y, g_grad_x) = helpers::compute_field_gradient(&g, dims);

        let n = phi.len();
        let mut kappa = vec![0.0_f64; n];
        let mut phi_new = phi.clone();
        let mut phi_gz = vec![0.0_f64; n];
        let mut phi_gy = vec![0.0_f64; n];
        let mut phi_gx = vec![0.0_f64; n];
        let mut adv = vec![0.0_f64; n];
        let mut sum_sqs = vec![0.0_f64; nz];

        for _iter in 0..self.max_iterations {
            helpers::compute_curvature_into(&phi, dims, &mut kappa);
            helpers::compute_field_gradient_into(&phi, dims, &mut phi_gz, &mut phi_gy, &mut phi_gx);
            helpers::upwind_advection_into(&phi, dims, &g_grad_z, &g_grad_y, &g_grad_x, &mut adv);

            let slice_len = ny * nx;

            helpers::evolve_slices_with_metric(
                &mut phi_new,
                &mut sum_sqs,
                slice_len,
                |iz, phi_new_s| {
                    let base = iz * slice_len;
                    let mut local_sum_sq = 0.0_f64;
                    for (i, phi_new_val) in phi_new_s.iter_mut().enumerate() {
                        let idx = base + i;
                        let grad_phi_mag = (phi_gz[idx] * phi_gz[idx]
                            + phi_gy[idx] * phi_gy[idx]
                            + phi_gx[idx] * phi_gx[idx])
                            .sqrt();

                        let curv = self.curvature_weight * g[idx] * kappa[idx] * grad_phi_mag;
                        let prop = self.propagation_weight * g[idx] * grad_phi_mag;
                        let advection = self.advection_weight * adv[idx];

                        let dphi = self.dt * (curv - prop + advection);
                        *phi_new_val = phi[idx] + dphi;
                        local_sum_sq += dphi * dphi;
                    }
                    local_sum_sq
                },
            );

            std::mem::swap(&mut phi, &mut phi_new);

            let sum_sq: f64 = sum_sqs.iter().sum();
            let rms = (sum_sq / n as f64).sqrt();
            if rms < self.tolerance {
                break;
            }
        }

        crate::native_output::from_values(
            image,
            phi.iter()
                .map(|&v| if v < 0.0 { 1.0_f32 } else { 0.0_f32 })
                .collect(),
            backend,
        )
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
    let grad_wide: Vec<f64> = grad_mag.iter().map(|&v| v as f64).collect();
    helpers::compute_edge_stopping(&grad_wide, k)
        .iter()
        .map(|&v| v as f32)
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_geodesic_active_contour.rs"]
mod tests;
