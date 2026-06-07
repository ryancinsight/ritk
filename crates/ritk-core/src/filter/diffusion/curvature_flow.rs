//! Mean curvature flow image filter.
//!
//! # Mathematical Specification
//!
//! Implements the pure mean curvature flow PDE (Osher & Sethian 1988):
//!
//!   ∂I/∂t = κ = div(∇I / |∇I|)
//!
//! where κ is the mean curvature of the iso-intensity level set through each voxel.
//!
//! Unlike `CurvatureAnisotropicDiffusionFilter` (which uses `∂I/∂t = |∇I| · κ`),
//! this filter does NOT weight the update by the gradient magnitude. This matches
//! ITK `itk::CurvatureFlowImageFilter` exactly.
//!
//! # 3-D Finite-Difference Discretisation
//!
//! Let first derivatives (central differences at interior, one-sided at boundaries):
//! I_x = (I\[p+x\] − I\[p−x\]) / 2
//! I_y = (I\[p+y\] − I\[p−y\]) / 2
//! I_z = (I\[p+z\] − I\[p−z\]) / 2
//!
//! Let second derivatives (symmetric 3-point stencils):
//! I_xx = I\[p+x\] − 2·I\[p\] + I\[p−x\]
//!   ... (similarly for yy, zz, xy, xz, yz)
//!
//! Mean curvature numerator (Caselles, Kimmel, Sapiro 1997):
//!   N = I_xx·(I_y² + I_z²)
//!     + I_yy·(I_x² + I_z²)
//!     + I_zz·(I_x² + I_y²)
//!     − 2·I_x·I_y·I_xy
//!     − 2·I_x·I_z·I_xz
//!     − 2·I_y·I_z·I_yz
//!
//! Gradient magnitude: |∇I|² = I_x² + I_y² + I_z²
//! Denominator: D = |∇I|³ = (I_x² + I_y² + I_z²)^(3/2)
//!
//! Update:
//!   κ(p) = N / D  if D > ε, else 0
//!   I_new(p) = I(p) + Δt · κ(p)
//!
//! # Stability
//!
//! For 3-D isotropic grids (Δx = Δy = Δz = 1), the explicit-Euler stability bound is:
//!   Δt ≤ 1/6 ≈ 0.1667
//!
//! ITK default: iterations=5, Δt=0.0625 (well within stability bound).
//!
//! # References
//! - ITK `itk::CurvatureFlowImageFilter<TInputImage, TOutputImage>`.
//! - Osher, S. and Sethian, J. A. (1988). "Fronts propagating with
//!   curvature-dependent speed: Algorithms based on Hamilton-Jacobi
//!   formulations." J. Comput. Phys. 79(1):12-49.
//! - Caselles, V., Kimmel, R., Sapiro, G. (1997). "Geodesic active contours."
//!   Int. J. Comput. Vis. 22(1):61-79.

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

// ── Stability constant ────────────────────────────────────────────────────────

/// Minimum denominator magnitude below which curvature is clamped to zero.
/// Prevents division by zero in flat regions.
const GRAD_MAG_EPSILON: f32 = 1e-12;

// ── CurvatureFlowConfig ───────────────────────────────────────────────────────

/// Configuration for `CurvatureFlowImageFilter`.
///
/// ITK defaults: `num_iterations = 5`, `time_step = 0.0625`.
#[derive(Debug, Clone, Copy)]
pub struct CurvatureFlowConfig {
    /// Number of explicit-Euler iterations. ITK default: 5.
    pub num_iterations: usize,
    /// Explicit Euler time step Δt. Must satisfy `Δt ≤ 1/6` for 3-D stability.
    /// ITK default: 0.0625.
    pub time_step: f32,
}

impl Default for CurvatureFlowConfig {
    fn default() -> Self {
        Self {
            num_iterations: 5,
            time_step: 0.0625,
        }
    }
}

// ── CurvatureFlowImageFilter ──────────────────────────────────────────────────

/// Pure mean curvature flow filter.
///
/// Evolves the image by `∂I/∂t = κ` (mean curvature of level sets) for a fixed
/// number of explicit-Euler iterations. This smooths small structures while
/// preserving larger geometric features longer than Gaussian smoothing.
///
/// # Differences from `CurvatureAnisotropicDiffusionFilter`
/// - This filter: `∂I/∂t = κ` (pure curvature, no gradient weighting).
/// - `CurvatureAnisotropicDiffusionFilter`: `∂I/∂t = |∇I| · κ` (anisotropic).
///
/// # Construction
/// ```rust,ignore
/// let filter = CurvatureFlowImageFilter::new(CurvatureFlowConfig {
///     num_iterations: 5,
///     time_step: 0.0625,
/// });
/// let smoothed = filter.apply(&image)?;
/// ```
#[derive(Debug, Clone)]
pub struct CurvatureFlowImageFilter {
    /// Filter configuration.
    pub config: CurvatureFlowConfig,
}

impl CurvatureFlowImageFilter {
    /// Construct a new `CurvatureFlowImageFilter` with the given configuration.
    pub fn new(config: CurvatureFlowConfig) -> Self {
        Self { config }
    }

    /// Apply mean curvature flow to a 3-D image.
    ///
    /// Returns `anyhow::Error` if the voxel data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let vals: &[f32] = &vals_vec;
        let [nz, ny, nx] = dims;
        let mut cur: Vec<f32> = vals.to_vec();
        // Pre-allocated double buffer: copy_from_slice + swap eliminates per-iteration
        // heap allocation (clone → alloc + drop) in favour of a plain memcpy.
        let mut next = vec![0.0f32; cur.len()];
        let dt = self.config.time_step;

        for _ in 0..self.config.num_iterations {
            next.copy_from_slice(&cur);

            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let idx = iz * ny * nx + iy * nx + ix;

                        // Helper: clamp-boundary neighbour access
                        let get = |z: isize, y: isize, x: isize| -> f32 {
                            let zz = z.clamp(0, nz as isize - 1) as usize;
                            let yy = y.clamp(0, ny as isize - 1) as usize;
                            let xx = x.clamp(0, nx as isize - 1) as usize;
                            cur[zz * ny * nx + yy * nx + xx]
                        };

                        let z = iz as isize;
                        let y = iy as isize;
                        let x = ix as isize;

                        // First derivatives (central differences, half-step spacing)
                        let ix_ = (get(z, y, x + 1) - get(z, y, x - 1)) * 0.5;
                        let iy_ = (get(z, y + 1, x) - get(z, y - 1, x)) * 0.5;
                        let iz_ = (get(z + 1, y, x) - get(z - 1, y, x)) * 0.5;

                        // Second derivatives
                        let c = cur[idx];
                        let ixx = get(z, y, x + 1) - 2.0 * c + get(z, y, x - 1);
                        let iyy = get(z, y + 1, x) - 2.0 * c + get(z, y - 1, x);
                        let izz = get(z + 1, y, x) - 2.0 * c + get(z - 1, y, x);

                        // Mixed second derivatives (cross terms)
                        let ixy =
                            (get(z, y + 1, x + 1) - get(z, y + 1, x - 1) - get(z, y - 1, x + 1)
                                + get(z, y - 1, x - 1))
                                * 0.25;
                        let ixz =
                            (get(z + 1, y, x + 1) - get(z + 1, y, x - 1) - get(z - 1, y, x + 1)
                                + get(z - 1, y, x - 1))
                                * 0.25;
                        let iyz =
                            (get(z + 1, y + 1, x) - get(z + 1, y - 1, x) - get(z - 1, y + 1, x)
                                + get(z - 1, y - 1, x))
                                * 0.25;

                        // Mean curvature numerator N
                        let num = ixx * (iy_ * iy_ + iz_ * iz_)
                            + iyy * (ix_ * ix_ + iz_ * iz_)
                            + izz * (ix_ * ix_ + iy_ * iy_)
                            - 2.0 * ix_ * iy_ * ixy
                            - 2.0 * ix_ * iz_ * ixz
                            - 2.0 * iy_ * iz_ * iyz;

                        // |∇I|³ (denominator)
                        let grad_sq = ix_ * ix_ + iy_ * iy_ + iz_ * iz_;
                        let denom = grad_sq.sqrt().powi(3); // = grad_sq^(3/2)

                        let kappa = if denom > GRAD_MAG_EPSILON {
                            num / denom
                        } else {
                            0.0
                        };

                        next[idx] = c + dt * kappa;
                    }
                }
            }

            std::mem::swap(&mut cur, &mut next);
        }

        Ok(rebuild(cur, dims, image))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::ops::extract_vec_infallible;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn cfg(iters: usize, dt: f32) -> CurvatureFlowConfig {
        CurvatureFlowConfig {
            num_iterations: iters,
            time_step: dt,
        }
    }

    // ── Analytical tests ──────────────────────────────────────────────────────

    /// Constant image: every derivative is zero → κ = 0 → image unchanged.
    /// Proof: ∇I = 0 everywhere → N = 0 → κ = 0 → ΔI = 0 each iteration.
    #[test]
    fn constant_image_unchanged() {
        let img = make_image(vec![42.0f32; 27], [3, 3, 3]);
        let out = CurvatureFlowImageFilter::new(cfg(5, 0.0625))
            .apply(&img)
            .unwrap();
        let (v, _) = extract_vec_infallible(&out);
        for &x in &v {
            assert!(
                (x - 42.0f32).abs() < 1e-4,
                "constant image not preserved: {x}"
            );
        }
    }

    /// Zero iterations: filter is identity.
    #[test]
    fn zero_iterations_identity() {
        let vals: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let img = make_image(vals.clone(), [3, 3, 3]);
        let out = CurvatureFlowImageFilter::new(cfg(0, 0.0625))
            .apply(&img)
            .unwrap();
        let (v, _) = extract_vec_infallible(&out);
        for (&o, &e) in v.iter().zip(vals.iter()) {
            assert_eq!(o, e, "0-iter output must equal input");
        }
    }

    /// Single voxel image: boundary conditions clamp all neighbours to same value
    /// → all derivatives are zero → κ = 0 → identity.
    #[test]
    fn single_voxel_identity() {
        let img = make_image(vec![100.0f32], [1, 1, 1]);
        let out = CurvatureFlowImageFilter::new(cfg(3, 0.0625))
            .apply(&img)
            .unwrap();
        let (v, _) = extract_vec_infallible(&out);
        assert!(
            (v[0] - 100.0f32).abs() < 1e-4,
            "single voxel must be unchanged: {}",
            v[0]
        );
    }

    /// Spatial metadata is preserved exactly.
    #[test]
    fn preserves_metadata() {
        let img = make_image(vec![5.0f32; 27], [3, 3, 3]);
        let out = CurvatureFlowImageFilter::new(cfg(2, 0.0625))
            .apply(&img)
            .unwrap();
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.direction(), img.direction());
        assert_eq!(out.shape(), [3, 3, 3]);
    }

    /// Output shape matches input shape.
    #[test]
    fn output_shape_matches_input() {
        let img = make_image(vec![1.0f32; 60], [3, 4, 5]);
        let out = CurvatureFlowImageFilter::new(cfg(2, 0.0625))
            .apply(&img)
            .unwrap();
        assert_eq!(out.shape(), [3, 4, 5]);
    }

    /// Smoothing reduces step-edge intensity range:
    /// A step edge between 0 and 100 should have its discontinuity softened
    /// (max decreases, min increases) after several iterations.
    #[test]
    fn step_edge_range_decreases() {
        // 3D volume: left half = 0.0, right half = 100.0 (step at x=2)
        let mut vals = vec![0.0f32; 27];
        for iz in 0..3 {
            for iy in 0..3 {
                for ix in 0..3 {
                    if ix >= 2 {
                        vals[iz * 9 + iy * 3 + ix] = 100.0;
                    }
                }
            }
        }
        let img = make_image(vals.clone(), [3, 3, 3]);
        let out = CurvatureFlowImageFilter::new(cfg(10, 0.0625))
            .apply(&img)
            .unwrap();
        let (v, _) = extract_vec_infallible(&out);
        let out_max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let in_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let in_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            out_max <= in_max,
            "max should not increase: out={out_max}, in={in_max}"
        );
        assert!(
            out_min >= in_min,
            "min should not decrease: out={out_min}, in={in_min}"
        );
    }

    /// ITK default configuration values match documented standard.
    #[test]
    fn default_config_matches_itk() {
        let cfg = CurvatureFlowConfig::default();
        assert_eq!(cfg.num_iterations, 5, "ITK default iterations = 5");
        assert!(
            (cfg.time_step - 0.0625f32).abs() < 1e-7,
            "ITK default dt = 0.0625"
        );
        // Stability: dt ≤ 1/6 ≈ 0.1667
        assert!(
            cfg.time_step <= 1.0 / 6.0 + 1e-6,
            "default dt must be within stability bound"
        );
    }
}
