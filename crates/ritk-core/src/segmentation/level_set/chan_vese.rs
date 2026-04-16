//! Chan-Vese level set segmentation (Active Contours Without Edges).
//!
//! # Mathematical Specification
//!
//! Implements the Chan & Vese (2001) model for segmenting an image into two
//! regions without relying on gradient (edge) information. The energy functional:
//!
//! ```text
//! E(φ, c₁, c₂) = μ·Length(C) + ν·Area(inside(C))
//!               + λ₁ ∫ |u₀ - c₁|² H(φ) dx
//!               + λ₂ ∫ |u₀ - c₂|² (1 - H(φ)) dx
//! ```
//!
//! where:
//! - `φ` is the level set function (C = {φ = 0} is the contour)
//! - `c₁` = mean intensity inside C (where φ > 0 after Heaviside)
//! - `c₂` = mean intensity outside C
//! - `μ` = curvature (length) penalty weight
//! - `ν` = area penalty weight
//! - `λ₁`, `λ₂` = data fidelity weights for inside/outside regions
//!
//! ## PDE Evolution (Euler-Lagrange)
//!
//! ```text
//! ∂φ/∂t = δ_ε(φ) [ μ · div(∇φ/|∇φ|) - ν - λ₁(u₀ - c₁)² + λ₂(u₀ - c₂)² ]
//! ```
//!
//! ## Regularised Heaviside and Dirac
//!
//! ```text
//! H_ε(z) = 0.5 · (1 + (2/π) · arctan(z/ε))
//! δ_ε(z) = (ε/π) / (ε² + z²)
//! ```
//!
//! ## Mean Intensity Updates
//!
//! ```text
//! c₁ = ∫ u₀ · H_ε(φ) dx  /  ∫ H_ε(φ) dx
//! c₂ = ∫ u₀ · (1 - H_ε(φ)) dx  /  ∫ (1 - H_ε(φ)) dx
//! ```
//!
//! ## Curvature
//!
//! ```text
//! κ = div(∇φ/|∇φ|)
//! ```
//!
//! computed via second-order central finite differences with clamped boundaries.
//!
//! ## Initialization
//!
//! Checkerboard signed distance function:
//!
//! ```text
//! φ₀(x,y,z) = -cos(πx/5) · cos(πy/5) · cos(πz/5)
//! ```
//!
//! where x, y, z are voxel indices. Negative regions seed the interior.
//!
//! ## Convergence
//!
//! Iteration stops when `max|φ^{n+1} - φ^n| / dt < tolerance` or
//! `max_iterations` is reached.
//!
//! # Complexity
//!
//! - Per iteration: O(N) where N = total voxels (two passes: mean update + PDE step)
//! - Total: O(max_iterations · N)
//! - Memory: O(N) for φ, curvature buffer, and scratch arrays
//!
//! # References
//!
//! - Chan, T. F. & Vese, L. A. (2001). "Active Contours Without Edges."
//!   *IEEE Transactions on Image Processing*, 10(2), 266–277.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Chan-Vese level set segmentation filter.
///
/// Segments a 3D image into foreground and background by evolving a level set
/// function under the Chan-Vese energy functional. No edge information is
/// required; the model is driven purely by region statistics.
#[derive(Debug, Clone)]
pub struct ChanVeseSegmentation {
    /// Length (curvature) penalty weight μ. Controls boundary smoothness.
    pub mu: f64,
    /// Area penalty weight ν. Positive values penalise large interior regions.
    pub nu: f64,
    /// Data fidelity weight for the inside region.
    pub lambda1: f64,
    /// Data fidelity weight for the outside region.
    pub lambda2: f64,
    /// Regularisation width ε for Heaviside and Dirac approximations.
    pub epsilon: f64,
    /// Euler forward time step Δt.
    pub dt: f64,
    /// Maximum number of PDE evolution iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on max|Δφ|/dt.
    pub tolerance: f64,
}

impl ChanVeseSegmentation {
    /// Construct with default parameters.
    ///
    /// | Parameter       | Default |
    /// |-----------------|---------|
    /// | `mu`            | 0.25    |
    /// | `nu`            | 0.0     |
    /// | `lambda1`       | 1.0     |
    /// | `lambda2`       | 1.0     |
    /// | `epsilon`       | 1.0     |
    /// | `dt`            | 0.1     |
    /// | `max_iterations`| 200     |
    /// | `tolerance`     | 1e-3    |
    pub fn new() -> Self {
        Self {
            mu: 0.25,
            nu: 0.0,
            lambda1: 1.0,
            lambda2: 1.0,
            epsilon: 1.0,
            dt: 0.1,
            max_iterations: 200,
            tolerance: 1e-3,
        }
    }

    /// Segment `image` into a binary mask via Chan-Vese level set evolution.
    ///
    /// Returns an `Image<B, 3>` with values 1.0 (inside, where φ ≥ 0 at
    /// convergence) and 0.0 (outside). Spatial metadata (origin, spacing,
    /// direction) is preserved from `image`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let device = image.data().device();

        let td = image.data().clone().into_data();
        let img: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("ChanVeseSegmentation requires f32 data: {:?}", e))?
            .to_vec();

        let mask = self.evolve(&img, dims);

        let tensor = Tensor::<B, 3>::from_data(TensorData::new(mask, Shape::new(dims)), &device);

        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

impl Default for ChanVeseSegmentation {
    fn default() -> Self {
        Self::new()
    }
}

// ── Core algorithm ─────────────────────────────────────────────────────────────

impl ChanVeseSegmentation {
    /// Run the PDE evolution on a flat f32 slice with shape `[nz, ny, nx]`.
    /// Returns a binary `Vec<f32>` (1.0 inside, 0.0 outside).
    fn evolve(&self, img: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        debug_assert_eq!(img.len(), n);

        if n == 0 {
            return Vec::new();
        }

        // Initialise φ with checkerboard signed distance function.
        let mut phi = vec![0.0_f64; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iz * ny * nx + iy * nx + ix;
                    let cz = (std::f64::consts::PI * iz as f64 / 5.0).cos();
                    let cy = (std::f64::consts::PI * iy as f64 / 5.0).cos();
                    let cx = (std::f64::consts::PI * ix as f64 / 5.0).cos();
                    phi[idx] = -(cz * cy * cx);
                }
            }
        }

        let img_f64: Vec<f64> = img.iter().map(|&v| v as f64).collect();
        let eps = self.epsilon;

        // Scratch buffers (reused across iterations).
        let mut kappa = vec![0.0_f64; n];

        for _iter in 0..self.max_iterations {
            // ── 1. Compute region means c1, c2 ────────────────────────────
            let (c1, c2) = compute_region_means(&img_f64, &phi, eps);

            // ── 2. Compute curvature κ = div(∇φ/|∇φ|) ───────────────────
            compute_curvature(&phi, dims, &mut kappa);

            // ── 3. Evolve φ ──────────────────────────────────────────────
            let mut max_dphi = 0.0_f64;

            for i in 0..n {
                let dirac = regularised_dirac(phi[i], eps);

                let diff1 = img_f64[i] - c1;
                let diff2 = img_f64[i] - c2;

                let force = self.mu * kappa[i] - self.nu - self.lambda1 * diff1 * diff1
                    + self.lambda2 * diff2 * diff2;

                let dphi = self.dt * dirac * force;
                phi[i] += dphi;

                let abs_dphi = dphi.abs();
                if abs_dphi > max_dphi {
                    max_dphi = abs_dphi;
                }
            }

            // ── 4. Convergence check ─────────────────────────────────────
            if max_dphi / self.dt < self.tolerance {
                break;
            }
        }

        // ── Threshold φ → binary mask ────────────────────────────────────
        phi.iter()
            .map(|&v| if v >= 0.0 { 1.0_f32 } else { 0.0_f32 })
            .collect()
    }
}

// ── Regularised Heaviside & Dirac ──────────────────────────────────────────────

/// Regularised Heaviside: H_ε(z) = 0.5 · (1 + (2/π) · arctan(z/ε)).
#[inline]
fn regularised_heaviside(z: f64, eps: f64) -> f64 {
    0.5 * (1.0 + (2.0 / std::f64::consts::PI) * (z / eps).atan())
}

/// Regularised Dirac delta: δ_ε(z) = (ε/π) / (ε² + z²).
#[inline]
fn regularised_dirac(z: f64, eps: f64) -> f64 {
    (eps / std::f64::consts::PI) / (eps * eps + z * z)
}

// ── Region mean computation ────────────────────────────────────────────────────

/// Compute c₁ (mean intensity inside) and c₂ (mean intensity outside).
///
/// ```text
/// c₁ = Σ u₀·H_ε(φ) / Σ H_ε(φ)
/// c₂ = Σ u₀·(1 - H_ε(φ)) / Σ (1 - H_ε(φ))
/// ```
///
/// If either denominator is zero (degenerate partition), the corresponding
/// mean is set to 0.0 to avoid division by zero.
fn compute_region_means(img: &[f64], phi: &[f64], eps: f64) -> (f64, f64) {
    let mut sum_h = 0.0_f64;
    let mut sum_uh = 0.0_f64;
    let mut sum_1mh = 0.0_f64;
    let mut sum_u1mh = 0.0_f64;

    for i in 0..img.len() {
        let h = regularised_heaviside(phi[i], eps);
        sum_h += h;
        sum_uh += img[i] * h;
        let omh = 1.0 - h;
        sum_1mh += omh;
        sum_u1mh += img[i] * omh;
    }

    let c1 = if sum_h > 1e-15 { sum_uh / sum_h } else { 0.0 };
    let c2 = if sum_1mh > 1e-15 {
        sum_u1mh / sum_1mh
    } else {
        0.0
    };

    (c1, c2)
}

// ── Curvature computation ──────────────────────────────────────────────────────

/// Compute mean curvature κ = div(∇φ/|∇φ|) for each voxel via central finite
/// differences with clamped (Neumann) boundary conditions.
///
/// The computation expands the divergence into second-derivative terms following
/// the standard discretisation of the curvature of a level set function on a
/// regular grid. A small constant (1e-10) is added to |∇φ| in the denominator
/// to prevent division by zero in flat regions.
fn compute_curvature(phi: &[f64], dims: [usize; 3], kappa: &mut [f64]) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    debug_assert_eq!(phi.len(), n);
    debug_assert_eq!(kappa.len(), n);

    let idx = |z: usize, y: usize, x: usize| -> usize { z * ny * nx + y * nx + x };

    // Clamped index helpers.
    let cz = |z: isize| -> usize { z.clamp(0, nz as isize - 1) as usize };
    let cy = |y: isize| -> usize { y.clamp(0, ny as isize - 1) as usize };
    let cx = |x: isize| -> usize { x.clamp(0, nx as isize - 1) as usize };

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = idx(iz, iy, ix);

                // Central first derivatives.
                let phi_xp = phi[idx(iz, iy, cx(ix as isize + 1))];
                let phi_xm = phi[idx(iz, iy, cx(ix as isize - 1))];
                let phi_yp = phi[idx(iz, cy(iy as isize + 1), ix)];
                let phi_ym = phi[idx(iz, cy(iy as isize - 1), ix)];
                let phi_zp = phi[idx(cz(iz as isize + 1), iy, ix)];
                let phi_zm = phi[idx(cz(iz as isize - 1), iy, ix)];

                let phi_x = (phi_xp - phi_xm) * 0.5;
                let phi_y = (phi_yp - phi_ym) * 0.5;
                let phi_z = (phi_zp - phi_zm) * 0.5;

                // Central second derivatives.
                let phi_xx = phi_xp - 2.0 * phi[i] + phi_xm;
                let phi_yy = phi_yp - 2.0 * phi[i] + phi_ym;
                let phi_zz = phi_zp - 2.0 * phi[i] + phi_zm;

                // Cross derivatives.
                let phi_xy = (phi[idx(iz, cy(iy as isize + 1), cx(ix as isize + 1))]
                    - phi[idx(iz, cy(iy as isize + 1), cx(ix as isize - 1))]
                    - phi[idx(iz, cy(iy as isize - 1), cx(ix as isize + 1))]
                    + phi[idx(iz, cy(iy as isize - 1), cx(ix as isize - 1))])
                    * 0.25;

                let phi_xz = (phi[idx(cz(iz as isize + 1), iy, cx(ix as isize + 1))]
                    - phi[idx(cz(iz as isize + 1), iy, cx(ix as isize - 1))]
                    - phi[idx(cz(iz as isize - 1), iy, cx(ix as isize + 1))]
                    + phi[idx(cz(iz as isize - 1), iy, cx(ix as isize - 1))])
                    * 0.25;

                let phi_yz = (phi[idx(cz(iz as isize + 1), cy(iy as isize + 1), ix)]
                    - phi[idx(cz(iz as isize + 1), cy(iy as isize - 1), ix)]
                    - phi[idx(cz(iz as isize - 1), cy(iy as isize + 1), ix)]
                    + phi[idx(cz(iz as isize - 1), cy(iy as isize - 1), ix)])
                    * 0.25;

                let grad_sq = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z;
                let grad_mag = (grad_sq + 1e-10).sqrt();
                let grad_sq_safe = grad_sq + 1e-10;

                // κ = (φ_xx(φ_y²+φ_z²) + φ_yy(φ_x²+φ_z²) + φ_zz(φ_x²+φ_y²)
                //     - 2φ_x φ_y φ_xy - 2φ_x φ_z φ_xz - 2φ_y φ_z φ_yz)
                //   / |∇φ|³
                let numerator = phi_xx * (phi_y * phi_y + phi_z * phi_z)
                    + phi_yy * (phi_x * phi_x + phi_z * phi_z)
                    + phi_zz * (phi_x * phi_x + phi_y * phi_y)
                    - 2.0 * phi_x * phi_y * phi_xy
                    - 2.0 * phi_x * phi_z * phi_xz
                    - 2.0 * phi_y * phi_z * phi_yz;

                kappa[i] = numerator / (grad_sq_safe * grad_mag);
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

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

    // ── Test 1: Bimodal sphere recovery ────────────────────────────────────────

    #[test]
    fn test_bimodal_sphere_segmentation() {
        // 16×16×16 image with a sphere of radius 5 at center (8,8,8).
        // Foreground intensity = 200, background intensity = 50.
        // Chan-Vese should approximately recover the sphere interior.
        let (nz, ny, nx) = (16, 16, 16);
        let n = nz * ny * nx;
        let mut data = vec![50.0_f32; n];
        let center = [8.0_f64, 8.0, 8.0];
        let radius = 5.0_f64;
        let radius_sq = radius * radius;

        let mut sphere_count = 0usize;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dz = iz as f64 - center[0];
                    let dy = iy as f64 - center[1];
                    let dx = ix as f64 - center[2];
                    if dz * dz + dy * dy + dx * dx <= radius_sq {
                        data[iz * ny * nx + iy * nx + ix] = 200.0;
                        sphere_count += 1;
                    }
                }
            }
        }

        let image = make_image(data, [nz, ny, nx]);
        let mut cv = ChanVeseSegmentation::new();
        cv.max_iterations = 300;
        cv.dt = 0.1;
        let result = cv.apply(&image).unwrap();
        let vals = get_values(&result);

        // Count segmented foreground voxels.
        let seg_count: usize = vals.iter().filter(|&&v| v == 1.0).count();

        // The segmented region should overlap substantially with the true sphere.
        // Compute Dice coefficient: 2·|A∩B| / (|A|+|B|).
        let mut intersection = 0usize;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iz * ny * nx + iy * nx + ix;
                    let dz = iz as f64 - center[0];
                    let dy = iy as f64 - center[1];
                    let dx = ix as f64 - center[2];
                    let in_sphere = dz * dz + dy * dy + dx * dx <= radius_sq;
                    let in_seg = vals[idx] == 1.0;
                    if in_sphere && in_seg {
                        intersection += 1;
                    }
                }
            }
        }

        let dice = if sphere_count + seg_count > 0 {
            2.0 * intersection as f64 / (sphere_count + seg_count) as f64
        } else {
            0.0
        };

        assert!(
            dice > 0.5,
            "Dice coefficient {:.4} too low; Chan-Vese should recover sphere (sphere_count={}, seg_count={}, intersection={})",
            dice,
            sphere_count,
            seg_count,
            intersection
        );
    }

    // ── Test 2: Uniform image converges to homogeneous label ───────────────────

    #[test]
    fn test_uniform_image_homogeneous_output() {
        // Uniform image: no edges. The output should be either all 0.0 or all 1.0
        // since there is no intensity contrast to drive the contour.
        let dims = [8, 8, 8];
        let n: usize = dims.iter().product();
        let data = vec![100.0_f32; n];
        let image = make_image(data, dims);

        let mut cv = ChanVeseSegmentation::new();
        cv.max_iterations = 300;
        let result = cv.apply(&image).unwrap();
        let vals = get_values(&result);

        let ones: usize = vals.iter().filter(|&&v| v == 1.0).count();
        let zeros: usize = vals.iter().filter(|&&v| v == 0.0).count();

        // Must be entirely one label or the other (or nearly so).
        // With uniform intensity, lambda1 and lambda2 terms are symmetric,
        // so nu=0 means the checkerboard initialisation may settle either way.
        // Accept if at least 90% of voxels share one label.
        let majority = ones.max(zeros);
        let ratio = majority as f64 / n as f64;
        assert!(
            ratio >= 0.90,
            "Uniform image should converge near-homogeneously; majority ratio = {:.4} (ones={}, zeros={})",
            ratio,
            ones,
            zeros
        );
    }

    // ── Test 3: Output is strictly binary ──────────────────────────────────────

    #[test]
    fn test_output_is_strictly_binary() {
        let dims = [10, 10, 10];
        let n: usize = dims.iter().product();
        // Gradient image to exercise non-trivial evolution.
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let ix = i % 10;
                if ix < 5 {
                    20.0
                } else {
                    180.0
                }
            })
            .collect();
        let image = make_image(data, dims);

        let result = ChanVeseSegmentation::new().apply(&image).unwrap();
        let vals = get_values(&result);

        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "Output voxel {} must be 0.0 or 1.0, got {}",
                i,
                v
            );
        }
    }

    // ── Test 4: Spatial metadata preserved ─────────────────────────────────────

    #[test]
    fn test_spatial_metadata_preserved() {
        let origin = [1.5, -2.0, 3.7];
        let spacing = [0.5, 0.8, 1.2];
        let dims = [6, 6, 6];
        let n: usize = dims.iter().product();
        let data = vec![100.0_f32; n];
        let image = make_image_with_metadata(data, dims, origin, spacing);

        let result = ChanVeseSegmentation::new().apply(&image).unwrap();

        assert_eq!(result.origin(), image.origin(), "Origin must be preserved");
        assert_eq!(
            result.spacing(),
            image.spacing(),
            "Spacing must be preserved"
        );
        assert_eq!(
            result.direction(),
            image.direction(),
            "Direction must be preserved"
        );
        assert_eq!(result.shape(), dims, "Shape must be preserved");
    }

    // ── Test 5: Regularised Heaviside properties ───────────────────────────────

    #[test]
    fn test_regularised_heaviside_properties() {
        let eps = 1.0;
        // H_ε(0) = 0.5 (symmetry)
        let h0 = regularised_heaviside(0.0, eps);
        assert!(
            (h0 - 0.5).abs() < 1e-12,
            "H_ε(0) must equal 0.5, got {}",
            h0
        );
        // H_ε(z) → 1 as z → +∞
        let h_large = regularised_heaviside(1e6, eps);
        assert!(
            (h_large - 1.0).abs() < 1e-6,
            "H_ε(large) must approach 1.0, got {}",
            h_large
        );
        // H_ε(z) → 0 as z → -∞
        let h_neg = regularised_heaviside(-1e6, eps);
        assert!(
            h_neg.abs() < 1e-6,
            "H_ε(-large) must approach 0.0, got {}",
            h_neg
        );
        // Monotonically increasing: H_ε(-1) < H_ε(0) < H_ε(1)
        assert!(regularised_heaviside(-1.0, eps) < h0);
        assert!(h0 < regularised_heaviside(1.0, eps));
    }

    // ── Test 6: Regularised Dirac properties ───────────────────────────────────

    #[test]
    fn test_regularised_dirac_properties() {
        let eps = 1.0;
        // δ_ε(0) = ε/(π·ε²) = 1/(π·ε)
        let d0 = regularised_dirac(0.0, eps);
        let expected = 1.0 / (std::f64::consts::PI * eps);
        assert!(
            (d0 - expected).abs() < 1e-12,
            "δ_ε(0) must equal 1/(πε), got {} vs expected {}",
            d0,
            expected
        );
        // δ_ε is symmetric: δ_ε(-z) = δ_ε(z)
        let d_pos = regularised_dirac(2.5, eps);
        let d_neg = regularised_dirac(-2.5, eps);
        assert!(
            (d_pos - d_neg).abs() < 1e-15,
            "Dirac must be symmetric: {} vs {}",
            d_pos,
            d_neg
        );
        // δ_ε(z) > 0 for all z
        assert!(d0 > 0.0);
        assert!(d_pos > 0.0);
        // Peak at z=0: δ_ε(0) ≥ δ_ε(z) for all z
        assert!(d0 >= d_pos);
    }

    // ── Test 7: Curvature of a sphere ──────────────────────────────────────────

    #[test]
    fn test_curvature_of_sphere_phi() {
        // For φ(x,y,z) = sqrt(x²+y²+z²) - R, the mean curvature at the
        // zero level set is κ = 2/R (sum of principal curvatures in 3D).
        // We check that the computed curvature at the center-adjacent voxels
        // is positive (convex surface) for a sphere-like φ.
        let n = 11;
        let center = 5.0_f64;
        let radius = 3.5_f64;
        let total = n * n * n;
        let dims = [n, n, n];
        let mut phi = vec![0.0_f64; total];
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let dz = iz as f64 - center;
                    let dy = iy as f64 - center;
                    let dx = ix as f64 - center;
                    let r = (dz * dz + dy * dy + dx * dx).sqrt();
                    phi[iz * n * n + iy * n + ix] = r - radius;
                }
            }
        }

        let mut kappa = vec![0.0_f64; total];
        compute_curvature(&phi, dims, &mut kappa);

        // At a point on the sphere surface (e.g. (5,5,8), distance ~3),
        // curvature should be positive.
        let test_idx = 5 * n * n + 5 * n + 8; // (5,5,8), distance = 3 from center
        assert!(
            kappa[test_idx] > 0.0,
            "Curvature of sphere φ at near-surface point must be positive, got {}",
            kappa[test_idx]
        );

        // At center (5,5,5), φ = -3.5 (well inside), curvature is less
        // meaningful but the discretisation should still yield a finite value.
        let center_idx = 5 * n * n + 5 * n + 5;
        assert!(
            kappa[center_idx].is_finite(),
            "Curvature at center must be finite, got {}",
            kappa[center_idx]
        );
    }

    // ── Test 8: Two-region slab with distinct intensities ──────────────────────

    #[test]
    fn test_two_region_slab() {
        // 1×10×10 image: left half intensity 20, right half intensity 200.
        // Chan-Vese should segment along the intensity boundary.
        let (nz, ny, nx) = (1, 10, 10);
        let n = nz * ny * nx;
        let mut data = vec![0.0_f32; n];
        for iy in 0..ny {
            for ix in 0..nx {
                data[iy * nx + ix] = if ix < 5 { 20.0 } else { 200.0 };
            }
        }

        let image = make_image(data.clone(), [nz, ny, nx]);
        let mut cv = ChanVeseSegmentation::new();
        cv.max_iterations = 500;
        let result = cv.apply(&image).unwrap();
        let vals = get_values(&result);

        // Count how many voxels in the high-intensity region (ix>=5) are
        // segmented differently from the low-intensity region (ix<5).
        let mut left_ones = 0usize;
        let mut right_ones = 0usize;
        for iy in 0..ny {
            for ix in 0..nx {
                let v = vals[iy * nx + ix];
                if ix < 5 {
                    if v == 1.0 {
                        left_ones += 1;
                    }
                } else if v == 1.0 {
                    right_ones += 1;
                }
            }
        }

        // The two halves should have different majority labels.
        let left_majority_one = left_ones > 25; // >50% of 50 voxels
        let right_majority_one = right_ones > 25;
        assert!(
            left_majority_one != right_majority_one,
            "Two-region slab should have distinct segmentation labels for each half \
             (left_ones={}, right_ones={})",
            left_ones,
            right_ones
        );
    }

    // ── Test 9: Default construction ───────────────────────────────────────────

    #[test]
    fn test_default_matches_new() {
        let d = ChanVeseSegmentation::default();
        let n = ChanVeseSegmentation::new();
        assert_eq!(d.mu, n.mu);
        assert_eq!(d.nu, n.nu);
        assert_eq!(d.lambda1, n.lambda1);
        assert_eq!(d.lambda2, n.lambda2);
        assert_eq!(d.epsilon, n.epsilon);
        assert_eq!(d.dt, n.dt);
        assert_eq!(d.max_iterations, n.max_iterations);
        assert_eq!(d.tolerance, n.tolerance);
    }
}
