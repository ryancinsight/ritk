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
use ritk_image::tensor::{Backend, Tensor};
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
    /// - `image`: input scalar 3D image.
    /// - `initial_phi`: initial level set function (same shape as `image`).
    ///   φ < 0 inside the initial contour, φ > 0 outside.
    ///
    /// # Returns
    /// Binary mask image: 1.0 where φ < 0 (inside), 0.0 elsewhere.
    ///
    /// # Errors
    /// Returns `Err` if tensor data cannot be read as `f32` or shapes mismatch.
    pub fn apply<B: Backend>(
        &self,
        image: &Image<f32, B, 3>,
        initial_phi: &Image<f32, B, 3>,
    ) -> anyhow::Result<Image<f32, B, 3>> {
        let dims = helpers::checked_dims(image.shape(), initial_phi.shape())?;
        let device = B::default();

        let (img_vals, _) = extract_vec(image)?;
        let (phi_init, _) = extract_vec(initial_phi)?;
        // Convert to f64 for the entire PDE pipeline.
        let img_wide: Vec<f64> = img_vals.iter().map(|&v| v as f64).collect();
        let phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();

        let fields = helpers::edge_stopping_fields(&img_wide, dims, self.sigma.get(), self.edge_k);

        // Per-iteration scratch (curvature, φ-gradient, upwind advection) is
        // owned by the shared engine and allocated once for the whole
        // evolution — the SEG-01 rationale: pre-allocating the scratch
        // outside the loop eliminates 4 × N×8 heap allocations per PDE
        // iteration.
        let phi = helpers::evolve_to_convergence::<helpers::RootMeanSquare, _, _>(
            phi,
            dims,
            self.dt,
            self.max_iterations,
            self.tolerance,
            |phi, scratch| {
                // Upwind discretisation of the advection (transport) term ∇g·∇φ;
                // central differencing it is unstable and leaks the front past edges.
                helpers::upwind_advection_into(
                    phi,
                    dims,
                    &fields.gz,
                    &fields.gy,
                    &fields.gx,
                    &mut scratch.extra,
                );
            },
            |idx, grad_phi_mag, scratch| {
                // Curvature term (positive κ for convex → contracts): w_c·g·κ·|∇φ|
                let curv =
                    self.curvature_weight * fields.g[idx] * scratch.kappa[idx] * grad_phi_mag;
                // Propagation term (positive w_p → expansion): −w_p·g·|∇φ|
                let prop = self.propagation_weight * fields.g[idx] * grad_phi_mag;
                // Advection term (attracts the front toward edges): +w_a·∇g·∇φ,
                // upwind-discretised for stability.
                let advection = self.advection_weight * scratch.extra[idx];

                self.dt * (curv - prop + advection)
            },
        );

        let mask = helpers::binary_mask(&phi);
        let tensor = Tensor::<f32, B>::from_slice_on(dims, &mask, &device);

        Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
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
        image: &ritk_image::Image<f32, B, 3>,
        initial_phi: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let dims = helpers::checked_dims(image.shape(), initial_phi.shape())?;

        let img_vals = image.data_slice()?;
        let phi_init = initial_phi.data_slice()?;
        let img_wide: Vec<f64> = img_vals.iter().map(|&v| v as f64).collect();
        let phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();

        let fields = helpers::edge_stopping_fields(&img_wide, dims, self.sigma.get(), self.edge_k);

        let phi = helpers::evolve_to_convergence::<helpers::RootMeanSquare, _, _>(
            phi,
            dims,
            self.dt,
            self.max_iterations,
            self.tolerance,
            |phi, scratch| {
                helpers::upwind_advection_into(
                    phi,
                    dims,
                    &fields.gz,
                    &fields.gy,
                    &fields.gx,
                    &mut scratch.extra,
                );
            },
            |idx, grad_phi_mag, scratch| {
                let curv =
                    self.curvature_weight * fields.g[idx] * scratch.kappa[idx] * grad_phi_mag;
                let prop = self.propagation_weight * fields.g[idx] * grad_phi_mag;
                let advection = self.advection_weight * scratch.extra[idx];

                self.dt * (curv - prop + advection)
            },
        );

        crate::native_output::from_values(image, helpers::binary_mask(&phi), backend)
    }
}

impl Default for GeodesicActiveContourSegmentation {
    fn default() -> Self {
        Self::new()
    }
}

// ── Test-only wrappers ─────────────────────────────────────────────────────────────────────────
//
// The existing tests call `compute_edge_stopping` with f32 data. This thin
// wrapper delegates to the shared f64 helper and converts back to f32,
// preserving the test-facing signatures without modifying any test function.

#[cfg(test)]
fn compute_edge_stopping(grad_mag: &[f32], k: f64) -> Vec<f32> {
    let grad_wide: Vec<f64> = grad_mag.iter().map(|&v| v as f64).collect();
    helpers::math::compute_edge_stopping(&grad_wide, k)
        .iter()
        .map(|&v| v as f32)
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_geodesic_active_contour.rs"]
mod tests;
