//! Curvature anisotropic diffusion filter.
//!
//! # Mathematical Specification
//!
//! Implements mean curvature motion of image level sets (Alvarez et al. 1992):
//!
//!   ∂I/∂t = |∇I| · div(∇I / |∇I|) = |∇I| · κ
//!
//! where κ is the mean curvature of the iso-intensity level set through each voxel.
//!
//! Unlike Perona-Malik (which modulates flux by scalar gradient magnitude),
//! curvature diffusion evolves each level set by its own mean curvature,
//! smoothing structures along level sets while preserving their geometry.
//!
//! # 3-D Finite-Difference Discretisation
//!
//! Let I_x, I_y, I_z denote central-difference first derivatives and I_xx, I_yy, I_zz,
//! I_xy, I_xz, I_yz denote second derivatives (symmetric 3-point stencils at interior
//! voxels, one-sided at boundaries).
//!
//! The curvature-weighted magnitude term is:
//!
//!   C(p) = I_xx·(I_y² + I_z²)
//!         + I_yy·(I_x² + I_z²)
//!         + I_zz·(I_x² + I_y²)
//!         − 2·I_x·I_y·I_xy
//!         − 2·I_x·I_z·I_xz
//!         − 2·I_y·I_z·I_yz
//!
//! Explicit Euler update at each voxel p:
//!
//!   I_new(p) = I(p) + Δt · C(p) / (I_x² + I_y² + I_z² + ε)
//!
//! where ε = 1e-10 prevents division by zero in flat regions.
//!
//! # Stability
//!
//! Stability condition for explicit Euler in 3-D: Δt ≤ 1/6 (unit spacing).
//! Default Δt = 1/16 provides a safety factor of ~2.67.
//! Neumann (zero-flux) boundary conditions are enforced by one-sided differences.
//!
//! # Invariants
//!
//! - Constant and linear images: C(p) = 0 everywhere → image unchanged.
//! - Mean intensity is approximately conserved (zero-mean update in the continuum).
//!
//! # References
//! - Alvarez, L., Lions, P.-L. & Morel, J.-M. (1992). Image selective smoothing
//!   and edge detection by nonlinear diffusion II. *SIAM J. Numer. Anal.* 29(3):845–866.
//! - Weickert, J. (1998). *Anisotropic Diffusion in Image Processing*. Teubner.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for the curvature anisotropic diffusion filter.
#[derive(Debug, Clone)]
pub struct CurvatureConfig {
    /// Number of explicit Euler time steps to perform.
    pub num_iterations: usize,
    /// Time step Δt.  Must satisfy Δt ≤ 1/6 for unit spacing.
    /// Default: 0.0625 = 1/16.
    pub time_step: f32,
}

impl Default for CurvatureConfig {
    fn default() -> Self {
        Self {
            num_iterations: 20,
            time_step: 1.0 / 16.0,
        }
    }
}

/// Curvature anisotropic diffusion filter (mean curvature motion of level sets).
///
/// Smooths images by evolving each iso-intensity level set according to its own
/// mean curvature.  Geometry of edges is preserved while noise is removed.
#[derive(Debug, Clone)]
pub struct CurvatureAnisotropicDiffusionFilter {
    /// Algorithm configuration.
    pub config: CurvatureConfig,
}

impl CurvatureAnisotropicDiffusionFilter {
    /// Create a filter with the given configuration.
    pub fn new(config: CurvatureConfig) -> Self {
        Self { config }
    }

    /// Apply the filter to `image`, returning a smoothed copy.
    ///
    /// # Errors
    /// Returns an error if the image tensor cannot be converted to `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| {
                anyhow::anyhow!(
                    "CurvatureAnisotropicDiffusionFilter requires f32 image data: {:?}",
                    e
                )
            })?
            .to_vec();

        let dims = image.shape();
        let spacing = [
            image.spacing()[0] as f32,
            image.spacing()[1] as f32,
            image.spacing()[2] as f32,
        ];

        let result = curvature_diffuse(&vals, dims, spacing, &self.config);

        let device = image.data().device();
        let td2 = TensorData::new(result, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td2, &device);
        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Numerical floor to prevent division by zero in flat regions.
const EPSILON: f32 = 1e-10;

/// Run explicit Euler curvature diffusion for the requested number of iterations.
///
/// Neumann (zero-flux) boundary conditions: one-sided differences at boundaries.
fn curvature_diffuse(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f32; 3],
    config: &CurvatureConfig,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let [sz, sy, sx] = spacing;

    let mut cur = data.to_vec();
    let mut next = vec![0.0f32; n];

    // Precompute reciprocals to avoid repeated division in the inner loop.
    let _inv_2sz = 1.0 / (2.0 * sz);
    let _inv_2sy = 1.0 / (2.0 * sy);
    let _inv_2sx = 1.0 / (2.0 * sx);
    let inv_sz2 = 1.0 / (sz * sz);
    let inv_sy2 = 1.0 / (sy * sy);
    let inv_sx2 = 1.0 / (sx * sx);

    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };

    for _ in 0..config.num_iterations {
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let c = idx(iz, iy, ix);

                    // ── First derivatives (central, one-sided at boundary) ──
                    let iz_m = if iz > 0 { iz - 1 } else { 0 };
                    let iz_p = if iz + 1 < nz { iz + 1 } else { nz - 1 };
                    let iy_m = if iy > 0 { iy - 1 } else { 0 };
                    let iy_p = if iy + 1 < ny { iy + 1 } else { ny - 1 };
                    let ix_m = if ix > 0 { ix - 1 } else { 0 };
                    let ix_p = if ix + 1 < nx { ix + 1 } else { nx - 1 };

                    // Effective half-step sizes for one-sided boundaries.
                    let dz_span = (iz_p - iz_m) as f32;
                    let dy_span = (iy_p - iy_m) as f32;
                    let dx_span = (ix_p - ix_m) as f32;

                    let dz = if dz_span > 0.0 {
                        (cur[idx(iz_p, iy, ix)] - cur[idx(iz_m, iy, ix)]) / (dz_span * sz)
                    } else {
                        0.0
                    };
                    let dy = if dy_span > 0.0 {
                        (cur[idx(iz, iy_p, ix)] - cur[idx(iz, iy_m, ix)]) / (dy_span * sy)
                    } else {
                        0.0
                    };
                    let dx = if dx_span > 0.0 {
                        (cur[idx(iz, iy, ix_p)] - cur[idx(iz, iy, ix_m)]) / (dx_span * sx)
                    } else {
                        0.0
                    };

                    // ── Second derivatives ─────────────────────────────────
                    // Diagonal: symmetric 3-point stencil or one-sided at edge.
                    let (az, bz, cz_) = diag_stencil(iz, nz);
                    let i_zz = (cur[idx(az, iy, ix)] - 2.0 * cur[idx(bz, iy, ix)]
                        + cur[idx(cz_, iy, ix)])
                        * inv_sz2;

                    let (ay, by_, cy) = diag_stencil(iy, ny);
                    let i_yy = (cur[idx(iz, ay, ix)] - 2.0 * cur[idx(iz, by_, ix)]
                        + cur[idx(iz, cy, ix)])
                        * inv_sy2;

                    let (ax, bx, cx) = diag_stencil(ix, nx);
                    let i_xx = (cur[idx(iz, iy, ax)] - 2.0 * cur[idx(iz, iy, bx)]
                        + cur[idx(iz, iy, cx)])
                        * inv_sx2;

                    // Cross: bilinear stencil, one-sided at edges.
                    let i_zy = if nz > 1 && ny > 1 {
                        let dz2 = (iz_p - iz_m) as f32 * sz;
                        let dy2 = (iy_p - iy_m) as f32 * sy;
                        (cur[idx(iz_p, iy_p, ix)]
                            - cur[idx(iz_p, iy_m, ix)]
                            - cur[idx(iz_m, iy_p, ix)]
                            + cur[idx(iz_m, iy_m, ix)])
                            / (dz2 * dy2 + EPSILON)
                    } else {
                        0.0
                    };

                    let i_zx = if nz > 1 && nx > 1 {
                        let dz2 = (iz_p - iz_m) as f32 * sz;
                        let dx2 = (ix_p - ix_m) as f32 * sx;
                        (cur[idx(iz_p, iy, ix_p)]
                            - cur[idx(iz_p, iy, ix_m)]
                            - cur[idx(iz_m, iy, ix_p)]
                            + cur[idx(iz_m, iy, ix_m)])
                            / (dz2 * dx2 + EPSILON)
                    } else {
                        0.0
                    };

                    let i_yx = if ny > 1 && nx > 1 {
                        let dy2 = (iy_p - iy_m) as f32 * sy;
                        let dx2 = (ix_p - ix_m) as f32 * sx;
                        (cur[idx(iz, iy_p, ix_p)]
                            - cur[idx(iz, iy_p, ix_m)]
                            - cur[idx(iz, iy_m, ix_p)]
                            + cur[idx(iz, iy_m, ix_m)])
                            / (dy2 * dx2 + EPSILON)
                    } else {
                        0.0
                    };

                    // ── Curvature update ───────────────────────────────────
                    // C(p) = I_xx(I_y² + I_z²) + I_yy(I_x² + I_z²) + I_zz(I_x² + I_y²)
                    //        - 2·I_x·I_y·I_xy - 2·I_x·I_z·I_xz - 2·I_y·I_z·I_yz
                    let grad_sq = dx * dx + dy * dy + dz * dz;

                    let curv_num = i_xx * (dy * dy + dz * dz)
                        + i_yy * (dx * dx + dz * dz)
                        + i_zz * (dx * dx + dy * dy)
                        - 2.0 * dx * dy * i_yx
                        - 2.0 * dx * dz * i_zx
                        - 2.0 * dy * dz * i_zy;

                    next[c] = cur[c] + config.time_step * curv_num / (grad_sq + EPSILON);
                }
            }
        }
        cur.copy_from_slice(&next);
    }

    cur
}

/// Returns three indices `(a, b, c)` for the symmetric 3-point stencil at
/// position `i` in a dimension of size `n`.
///
/// Interior: `(i-1, i, i+1)` — standard symmetric stencil.
/// Left boundary (`i == 0`): `(0, 1, 2)` — forward 2nd-difference.
/// Right boundary (`i == n-1`): `(n-3, n-2, n-1)` — backward 2nd-difference.
#[inline(always)]
fn diag_stencil(i: usize, n: usize) -> (usize, usize, usize) {
    if n <= 1 {
        return (0, 0, 0);
    }
    if i == 0 {
        (0, 1, 2.min(n - 1))
    } else if i + 1 >= n {
        (n.saturating_sub(3), n - 2, n - 1)
    } else {
        (i - 1, i, i + 1)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        use crate::spatial::{Direction, Point, Spacing};
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn image_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── Test 1: constant image must be unchanged ───────────────────────────────

    #[test]
    fn test_constant_image_unchanged() {
        let dims = [8, 8, 8];
        let vals = vec![5.0f32; 8 * 8 * 8];
        let img = make_image(vals.clone(), dims);

        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig::default());
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);

        let max_diff = out_vals
            .iter()
            .zip(vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "constant image must be unchanged; max diff = {max_diff}"
        );
    }

    // ── Test 2: linear field must be unchanged ─────────────────────────────────
    // A linear ramp I(x,y,z) = ax+by+cz has zero curvature everywhere.

    #[test]
    fn test_linear_field_unchanged() {
        let [nz, ny, nx] = [10usize, 10, 10];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|i| {
                let ix = (i % nx) as f32;
                let iy = ((i / nx) % ny) as f32;
                let iz = (i / (ny * nx)) as f32;
                0.3 * ix + 0.5 * iy + 0.7 * iz
            })
            .collect();
        let img = make_image(vals.clone(), [nz, ny, nx]);

        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: 10,
            time_step: 1.0 / 16.0,
        });
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);

        // Interior voxels only (boundaries use one-sided stencils which introduce small errors)
        let mut max_interior_diff = 0.0f32;
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let i = iz * ny * nx + iy * nx + ix;
                    let diff = (out_vals[i] - vals[i]).abs();
                    max_interior_diff = max_interior_diff.max(diff);
                }
            }
        }
        assert!(
            max_interior_diff < 1e-3,
            "linear field interior should be unchanged; max diff = {max_interior_diff}"
        );
    }

    // ── Test 3: mean conservation ──────────────────────────────────────────────

    #[test]
    fn test_mean_conservation() {
        let dims = [12, 12, 12];
        let n = 12 * 12 * 12;
        // Sinusoidal image with non-trivial curvature.
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let ix = (i % 12) as f32 / 12.0;
                let iy = ((i / 12) % 12) as f32 / 12.0;
                let iz = (i / 144) as f32 / 12.0;
                (std::f32::consts::PI * ix).sin()
                    * (std::f32::consts::PI * iy).cos()
                    * (std::f32::consts::PI * iz).sin()
                    + 5.0
            })
            .collect();
        let mean_in: f32 = vals.iter().sum::<f32>() / n as f32;

        let img = make_image(vals, dims);
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: 20,
            time_step: 1.0 / 16.0,
        });
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);
        let mean_out: f32 = out_vals.iter().sum::<f32>() / n as f32;

        let rel_err = ((mean_out - mean_in) / mean_in).abs();
        assert!(
            rel_err < 1e-2,
            "mean should be approximately conserved; rel err = {rel_err}"
        );
    }

    // ── Test 4: spherical blob smoothed (gradient reduced) ────────────────────

    #[test]
    fn test_spherical_blob_smoothed() {
        let [nz, ny, nx] = [24usize, 24, 24];
        let n = nz * ny * nx;
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let ix = (i % nx) as f32 - 12.0;
                let iy = ((i / nx) % ny) as f32 - 12.0;
                let iz = (i / (ny * nx)) as f32 - 12.0;
                let r = (ix * ix + iy * iy + iz * iz).sqrt();
                if r < 6.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        // Compute max gradient magnitude before filtering.
        let max_grad_in = max_gradient_magnitude(&vals, [nz, ny, nx]);

        let img = make_image(vals, [nz, ny, nx]);
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: 10,
            time_step: 1.0 / 16.0,
        });
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);
        let max_grad_out = max_gradient_magnitude(&out_vals, [nz, ny, nx]);

        assert!(
            max_grad_out < max_grad_in,
            "spherical blob should be smoothed; max grad before={max_grad_in:.4} after={max_grad_out:.4}"
        );
    }

    // ── Test 5: stability — outputs finite and within intensity range ──────────

    #[test]
    fn test_stability_small_timestep() {
        let [nz, ny, nx] = [10usize, 10, 10];
        let n = nz * ny * nx;
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let ix = i % nx;
                let iy = (i / nx) % ny;
                let iz = i / (ny * nx);
                ((ix + iy * 3 + iz * 7) % 17) as f32 * 10.0
            })
            .collect();
        let v_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let v_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let img = make_image(vals, [nz, ny, nx]);
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: 50,
            time_step: 1.0 / 16.0,
        });
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);

        for &v in &out_vals {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
        // Allow a small overshoot margin due to finite-difference approximation.
        let margin = (v_max - v_min) * 0.05;
        for &v in &out_vals {
            assert!(
                v >= v_min - margin && v <= v_max + margin,
                "output value {v} outside input range [{}, {}] (+margin)",
                v_min - margin,
                v_max + margin
            );
        }
    }

    // ── Helper: max gradient magnitude via central differences ─────────────────

    fn max_gradient_magnitude(data: &[f32], dims: [usize; 3]) -> f32 {
        let [nz, ny, nx] = dims;
        let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
        let mut max_g = 0.0f32;
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let gz = (data[idx(iz + 1, iy, ix)] - data[idx(iz - 1, iy, ix)]) * 0.5;
                    let gy = (data[idx(iz, iy + 1, ix)] - data[idx(iz, iy - 1, ix)]) * 0.5;
                    let gx = (data[idx(iz, iy, ix + 1)] - data[idx(iz, iy, ix - 1)]) * 0.5;
                    let g = (gz * gz + gy * gy + gx * gx).sqrt();
                    max_g = max_g.max(g);
                }
            }
        }
        max_g
    }
}
