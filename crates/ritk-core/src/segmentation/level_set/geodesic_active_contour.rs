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
                    (phi_gz[i] * phi_gz[i] + phi_gy[i] * phi_gy[i] + phi_gx[i] * phi_gx[i])
                        .sqrt();

                // Curvature term (positive κ for convex → contracts): w_c·g·κ·|∇φ|
                let curv =
                    self.curvature_weight * g[i] * kappa[i] * grad_phi_mag;

                // Propagation term (positive w_p → expansion): −w_p·g·|∇φ|
                let prop = self.propagation_weight * g[i] * grad_phi_mag;

                // Advection term (attracts toward edges): −w_a·∇g·∇φ
                let advection = self.advection_weight
                    * (g_grad_z[i] * phi_gz[i]
                        + g_grad_y[i] * phi_gy[i]
                        + g_grad_x[i] * phi_gx[i]);

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
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
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
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn make_image_with_metadata(
        data: Vec<f32>,
        dims: [usize; 3],
        origin: [f64; 3],
        spacing: [f64; 3],
    ) -> Image<B, 3> {
        let device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    fn get_values(image: &Image<B, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// Create a signed distance–like initial φ: negative inside a sphere of
    /// radius `r` centred at (`cz`,`cy`,`cx`), positive outside.
    fn sphere_phi(dims: [usize; 3], center: [f64; 3], r: f64) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let mut phi = vec![0.0_f32; nz * ny * nx];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dz = iz as f64 - center[0];
                    let dy = iy as f64 - center[1];
                    let dx = ix as f64 - center[2];
                    let dist = (dz * dz + dy * dy + dx * dx).sqrt();
                    phi[iz * ny * nx + iy * nx + ix] = (dist - r) as f32;
                }
            }
        }
        phi
    }

    // ── Test 1: Step-edge image — contour expands to edge ──────────────────────

    #[test]
    fn test_step_edge_contour_expands_to_edge() {
        // 16×16×16 image: foreground sphere of radius 6 at center with
        // intensity 200, background 0. Initial φ is a small sphere of
        // radius 2 inside the foreground.
        //
        // Pure balloon expansion (curvature=0, advection=0) with edge-modulated
        // speed: g ≈ 1 inside homogeneous region, g ≪ 1 near edge. The contour
        // should expand from the initial sphere toward the foreground boundary.
        let dims = [16, 16, 16];
        let [nz, ny, nx] = dims;
        let center = [8.0, 8.0, 8.0];
        let fg_radius = 6.0;

        let mut img_data = vec![0.0_f32; nz * ny * nx];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dz = iz as f64 - center[0];
                    let dy = iy as f64 - center[1];
                    let dx = ix as f64 - center[2];
                    if (dz * dz + dy * dy + dx * dx).sqrt() <= fg_radius {
                        img_data[iz * ny * nx + iy * nx + ix] = 200.0;
                    }
                }
            }
        }

        let image = make_image(img_data, dims);
        let init_phi = sphere_phi(dims, center, 2.0);
        let phi_image = make_image(init_phi, dims);

        // Pure balloon: no curvature (avoids shrinkage at the small initial
        // sphere where κ is large), no advection. dt=0.05 keeps the explicit
        // Euler scheme stable.
        let mut gac = GeodesicActiveContourSegmentation::new();
        gac.propagation_weight = 3.0;
        gac.curvature_weight = 0.0;
        gac.advection_weight = 0.0;
        gac.edge_k = 50.0;
        gac.sigma = 0.5;
        gac.dt = 0.05;
        gac.max_iterations = 500;

        let result = gac.apply(&image, &phi_image).unwrap();
        let mask = get_values(&result);

        // Count segmented voxels.
        let seg_count: usize = mask.iter().filter(|&&v| v == 1.0).count();

        // Count actual foreground voxels.
        let fg_count: usize = {
            let mut c = 0;
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let dz = iz as f64 - center[0];
                        let dy = iy as f64 - center[1];
                        let dx = ix as f64 - center[2];
                        if (dz * dz + dy * dy + dx * dx).sqrt() <= fg_radius {
                            c += 1;
                        }
                    }
                }
            }
            c
        };

        // Initial sphere (radius 2) has fewer voxels than the foreground.
        // After balloon expansion, the segmented region must be substantially
        // larger than the initial contour.
        let init_count: usize = sphere_phi(dims, center, 2.0)
            .iter()
            .filter(|&&v| v < 0.0)
            .count();

        assert!(
            seg_count > init_count * 2,
            "segmented region ({}) must be substantially larger than initial contour ({})",
            seg_count,
            init_count
        );

        // Compute overlap between segmented region and foreground sphere.
        let mut overlap = 0usize;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let i = iz * ny * nx + iy * nx + ix;
                    let dz = iz as f64 - center[0];
                    let dy = iy as f64 - center[1];
                    let dx = ix as f64 - center[2];
                    let in_fg = (dz * dz + dy * dy + dx * dx).sqrt() <= fg_radius;
                    if mask[i] == 1.0 && in_fg {
                        overlap += 1;
                    }
                }
            }
        }

        // With pure balloon on a step-edge image, the contour expands through
        // the homogeneous interior (g ≈ 1) and slows at the edge (g ≪ 1).
        // It will leak past the edge over many iterations. We verify:
        //  (a) expansion happened (checked above)
        //  (b) most of the foreground is covered (recall)
        let recall = overlap as f64 / fg_count.max(1) as f64;
        assert!(
            recall > 0.5,
            "recall w.r.t. foreground must exceed 0.5, got {:.4} \
             (overlap={}, fg_count={}, seg_count={})",
            recall,
            overlap,
            fg_count,
            seg_count
        );
    }

    // ── Test 2: Uniform image — no edges, uniform expansion ────────────────────

    #[test]
    fn test_uniform_image_no_edges() {
        // Uniform image: g ≡ 1, ∇g ≡ 0. With positive propagation, the
        // contour should expand. With enough iterations, most voxels become
        // inside (φ < 0).
        let dims = [10, 10, 10];
        let n: usize = dims.iter().product();
        let img_data = vec![100.0_f32; n];
        let image = make_image(img_data, dims);

        let init_phi = sphere_phi(dims, [5.0, 5.0, 5.0], 2.0);
        let phi_image = make_image(init_phi, dims);

        let mut gac = GeodesicActiveContourSegmentation::new();
        gac.propagation_weight = 2.0;
        gac.curvature_weight = 0.0;
        gac.advection_weight = 0.0;
        gac.dt = 0.1;
        gac.max_iterations = 300;

        let result = gac.apply(&image, &phi_image).unwrap();
        let mask = get_values(&result);

        let seg_count: usize = mask.iter().filter(|&&v| v == 1.0).count();
        let init_count: usize = sphere_phi(dims, [5.0, 5.0, 5.0], 2.0)
            .iter()
            .filter(|&&v| v < 0.0)
            .count();

        // With uniform g=1 and positive propagation, the region must expand.
        assert!(
            seg_count > init_count,
            "with positive propagation on uniform image, segmented region ({}) \
             must exceed initial ({})",
            seg_count,
            init_count
        );
    }

    // ── Test 3: Output is strictly binary ──────────────────────────────────────

    #[test]
    fn test_output_is_binary() {
        let dims = [8, 8, 8];
        let n: usize = dims.iter().product();
        // Random-ish image data.
        let img_data: Vec<f32> = (0..n).map(|i| ((i * 37 + 13) % 256) as f32).collect();
        let image = make_image(img_data, dims);

        let init_phi = sphere_phi(dims, [4.0, 4.0, 4.0], 3.0);
        let phi_image = make_image(init_phi, dims);

        let mut gac = GeodesicActiveContourSegmentation::new();
        gac.max_iterations = 20;

        let result = gac.apply(&image, &phi_image).unwrap();
        let mask = get_values(&result);

        for (i, &v) in mask.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "output at voxel {} must be 0.0 or 1.0, got {}",
                i,
                v
            );
        }
    }

    // ── Test 4: Spatial metadata preserved ─────────────────────────────────────

    #[test]
    fn test_metadata_preserved() {
        let dims = [4, 4, 4];
        let n: usize = dims.iter().product();
        let origin = [1.5, -2.0, 3.7];
        let spacing = [0.5, 1.0, 2.0];

        let image = make_image_with_metadata(vec![50.0_f32; n], dims, origin, spacing);
        let phi_image = make_image_with_metadata(
            sphere_phi(dims, [2.0, 2.0, 2.0], 1.5),
            dims,
            origin,
            spacing,
        );

        let mut gac = GeodesicActiveContourSegmentation::new();
        gac.max_iterations = 5;

        let result = gac.apply(&image, &phi_image).unwrap();

        assert_eq!(result.origin(), image.origin(), "origin must be preserved");
        assert_eq!(
            result.spacing(),
            image.spacing(),
            "spacing must be preserved"
        );
        assert_eq!(
            result.direction(),
            image.direction(),
            "direction must be preserved"
        );
        assert_eq!(result.shape(), dims, "shape must be preserved");
    }

    // ── Test 5: Shape mismatch error ───────────────────────────────────────────

    #[test]
    fn test_shape_mismatch_returns_error() {
        let image = make_image(vec![0.0_f32; 27], [3, 3, 3]);
        let phi_image = make_image(vec![0.0_f32; 8], [2, 2, 2]);

        let gac = GeodesicActiveContourSegmentation::new();
        let result = gac.apply(&image, &phi_image);
        assert!(result.is_err(), "shape mismatch must produce an error");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("must match"),
            "error message must mention shape mismatch, got: {}",
            err_msg
        );
    }

    // ── Test 6: Edge stopping function correctness ─────────────────────────────

    #[test]
    fn test_edge_stopping_values() {
        // g(0) = 1, g(k) = 0.5, lim g(∞) → 0.
        let k = 2.0;
        let grad = vec![0.0_f32, 2.0, 100.0];
        let g = compute_edge_stopping(&grad, k);

        // g(0) = 1/(1 + 0) = 1.0
        assert!((g[0] - 1.0).abs() < 1e-6, "g(0) must be 1.0, got {}", g[0]);
        // g(k) = 1/(1 + 1) = 0.5
        assert!((g[1] - 0.5).abs() < 1e-6, "g(k) must be 0.5, got {}", g[1]);
        // g(100) ≈ 1/(1 + 2500) ≈ 0.0004
        assert!(g[2] < 0.01, "g(large) must be near 0, got {}", g[2]);
    }

    // ── Test 7: Gaussian kernel sums to 1 ──────────────────────────────────────

    #[test]
    fn test_gaussian_kernel_normalised() {
        let kernel = build_gaussian_kernel_1d(2.0, 6);
        let sum: f32 = kernel.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Gaussian kernel must sum to 1.0, got {}",
            sum
        );
        // Kernel must be symmetric.
        let len = kernel.len();
        for i in 0..len / 2 {
            assert!(
                (kernel[i] - kernel[len - 1 - i]).abs() < 1e-6,
                "kernel must be symmetric: k[{}]={} vs k[{}]={}",
                i,
                kernel[i],
                len - 1 - i,
                kernel[len - 1 - i]
            );
        }
    }

    // ── Test 8: Default construction ───────────────────────────────────────────

    #[test]
    fn test_default_matches_new() {
        let a = GeodesicActiveContourSegmentation::new();
        let b = GeodesicActiveContourSegmentation::default();
        assert_eq!(a.propagation_weight, b.propagation_weight);
        assert_eq!(a.curvature_weight, b.curvature_weight);
        assert_eq!(a.advection_weight, b.advection_weight);
        assert_eq!(a.edge_k, b.edge_k);
        assert_eq!(a.sigma, b.sigma);
        assert_eq!(a.dt, b.dt);
        assert_eq!(a.max_iterations, b.max_iterations);
        assert_eq!(a.tolerance, b.tolerance);
    }
}
