//! Laplacian level set segmentation for 3-D medical images.
//!
//! # Mathematical Specification
//!
//! Segments structures where the image Laplacian is negative (local bright
//! maxima, thin bright structures).  The contour expands toward bright regions
//! driven by the normalised Laplacian speed function.
//!
//! ## Speed Function
//!
//! Let `I` be the Gaussian pre-smoothed input image.  The second-order
//! central-difference Laplacian with clamped boundaries is:
//!
//! ```text
//! L(I)(x) = d2I/dz2 + d2I/dy2 + d2I/dx2
//! ```
//!
//! Normalised to bound values in `(-1, 1)`:
//!
//! ```text
//! F(x) = L(I)(x) / (1.0 + |L(I)(x)|)
//! ```
//!
//! ## PDE
//!
//! Forward-Euler level set evolution:
//!
//! ```text
//! dphi/dt = [w_p * F(x) + w_c * kappa] * |grad phi|
//! ```
//!
//! where:
//! - `kappa = div(grad phi / |grad phi|)` is mean curvature,
//! - `w_p` (`propagation_weight`) controls Laplacian-driven expansion/contraction,
//! - `w_c` (`curvature_weight`) regularises the contour via curvature flow.
//!
//! For a 3-D Gaussian blob `I(r) = exp(-r^2/(2*sigma^2))` the Laplacian at
//! the centre is `L = -3/sigma^2`, giving `F < 0`.  With default `w_p = 1.0`
//! the propagation term `w_p * F < 0`, so the speed is negative and the
//! contour expands toward bright maxima.
//!
//! ## Output
//!
//! Binary mask: `1.0` where `phi < 0`, `0.0` elsewhere.
//!
//! ## Convergence
//!
//! Stops when `max |delta phi| / dt < tolerance` or
//! `iterations >= max_iterations`.
//!
//! # References
//!
//! - Sethian, J. A. (1999). *Level Set Methods and Fast Marching Methods*.
//!   Cambridge University Press.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

use super::helpers;

/// Laplacian level set segmentation.
///
/// Evolves an initial level set `phi` using a speed function derived from the
/// normalised image Laplacian.  Regions where `L(I) < 0` (local bright
/// maxima) produce negative propagation speed, driving the contour to expand
/// toward bright structures.
///
/// | Parameter            | Default |
/// |----------------------|---------|
/// | `propagation_weight` | 1.0     |
/// | `curvature_weight`   | 0.2     |
/// | `sigma`              | 1.0     |
/// | `dt`                 | 0.05    |
/// | `max_iterations`     | 200     |
/// | `tolerance`          | 1e-3    |
#[derive(Debug, Clone)]
pub struct LaplacianLevelSet {
    /// Weight `w_p` on the Laplacian propagation term.
    pub propagation_weight: f64,
    /// Weight `w_c` on the curvature regularisation term.
    pub curvature_weight: f64,
    /// Standard deviation for Gaussian pre-smoothing of the input image.
    pub sigma: f64,
    /// Forward-Euler time step `dt`.
    pub dt: f64,
    /// Maximum number of PDE iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on `max |delta phi| / dt`.
    pub tolerance: f64,
}

impl LaplacianLevelSet {
    /// Construct with default parameters.
    pub fn new() -> Self {
        Self {
            propagation_weight: 1.0,
            curvature_weight: 0.2,
            sigma: 1.0,
            dt: 0.05,
            max_iterations: 200,
            tolerance: 1e-3,
        }
    }

    /// Apply Laplacian level set segmentation to a 3-D image.
    ///
    /// # Errors
    ///
    /// Returns `Err` if shapes differ or data cannot be read as `f32`.
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
        let [nz, ny, nx] = dims;
        let n: usize = nz * ny * nx;

        let img_td = image.data().clone().into_data();
        let img_f32: Vec<f32> = img_td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("LaplacianLevelSet requires f32 image data: {:?}", e))?
            .to_vec();
        let img_f64: Vec<f64> = img_f32.iter().map(|&v| v as f64).collect();

        let phi_td = initial_phi.data().clone().into_data();
        let phi_f32: Vec<f32> = phi_td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("LaplacianLevelSet requires f32 phi data: {:?}", e))?
            .to_vec();
        let mut phi: Vec<f64> = phi_f32.iter().map(|&v| v as f64).collect();

        // Optional Gaussian pre-smoothing of the input image.
        let smoothed = if self.sigma > 0.0 {
            helpers::gaussian_smooth_3d(&img_f64, dims, self.sigma)
        } else {
            img_f64.clone()
        };

        // L(I)[i] = d2I/dz2 + d2I/dy2 + d2I/dx2  (central diffs, clamped BC).
        let mut laplacian = vec![0.0_f64; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let i = iz * ny * nx + iy * nx + ix;
                    let zz = iz as isize;
                    let yy = iy as isize;
                    let xx = ix as isize;

                    let d2z = smoothed[helpers::idx_clamped(zz + 1, yy, xx, nz, ny, nx)]
                        - 2.0 * smoothed[i]
                        + smoothed[helpers::idx_clamped(zz - 1, yy, xx, nz, ny, nx)];
                    let d2y = smoothed[helpers::idx_clamped(zz, yy + 1, xx, nz, ny, nx)]
                        - 2.0 * smoothed[i]
                        + smoothed[helpers::idx_clamped(zz, yy - 1, xx, nz, ny, nx)];
                    let d2x = smoothed[helpers::idx_clamped(zz, yy, xx + 1, nz, ny, nx)]
                        - 2.0 * smoothed[i]
                        + smoothed[helpers::idx_clamped(zz, yy, xx - 1, nz, ny, nx)];

                    laplacian[i] = d2z + d2y + d2x;
                }
            }
        }

        // F[i] = L[i] / (1.0 + |L[i]|) maps L into (-1, +1).
        let speed_field: Vec<f64> = laplacian
            .iter()
            .map(|&l| l / (1.0 + l.abs()))
            .collect();

        // PDE scratch buffer for mean curvature kappa.
        let mut kappa = vec![0.0_f64; n];

        // dphi/dt = [w_p * F(x) + w_c * kappa] * |grad phi|
        for _iter in 0..self.max_iterations {
            helpers::compute_curvature_into(&phi, dims, &mut kappa);
            let (phi_z, phi_y, phi_x) = helpers::compute_field_gradient(&phi, dims);

            let mut max_change = 0.0_f64;

            for i in 0..n {
                let grad_phi_mag = (phi_z[i] * phi_z[i]
                    + phi_y[i] * phi_y[i]
                    + phi_x[i] * phi_x[i])
                    .sqrt();

                let speed = self.propagation_weight * speed_field[i]
                    + self.curvature_weight * kappa[i];
                let dphi = self.dt * speed * grad_phi_mag;
                phi[i] += dphi;

                let change = dphi.abs() / self.dt;
                if change > max_change {
                    max_change = change;
                }
            }

            if max_change < self.tolerance {
                break;
            }
        }

        // Binary mask: phi < 0 => 1.0 (foreground), else => 0.0 (background).
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

impl Default for LaplacianLevelSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};

    type B = burn_ndarray::NdArray<f32>;

    fn make_image(dims: [usize; 3], val: f32) -> Image<B, 3> {
        let n: usize = dims.iter().product();
        let device = Default::default();
        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(vec![val; n], Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn make_image_with_metadata(
        dims: [usize; 3],
        val: f32,
        origin: [f64; 3],
        spacing: [f64; 3],
    ) -> Image<B, 3> {
        let n: usize = dims.iter().product();
        let device = Default::default();
        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(vec![val; n], Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    fn sphere_phi(dims: [usize; 3], center: [f64; 3], radius: f64) -> Image<B, 3> {
        let n: usize = dims.iter().product();
        let [nz, ny, nx] = dims;
        let mut data = vec![0.0_f32; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dist = ((iz as f64 - center[0]).powi(2)
                        + (iy as f64 - center[1]).powi(2)
                        + (ix as f64 - center[2]).powi(2))
                    .sqrt();
                    data[iz * ny * nx + iy * nx + ix] = (dist - radius) as f32;
                }
            }
        }
        let device = Default::default();
        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn count_foreground(image: &Image<B, 3>) -> usize {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .filter(|&&v| v == 1.0)
            .count()
    }

    fn count_phi_inside(phi: &Image<B, 3>) -> usize {
        image_values(phi).iter().filter(|&&v| v < 0.0).count()
    }

    fn image_values(image: &Image<B, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// All output voxels must be exactly 0.0 or 1.0.
    #[test]
    fn test_output_is_binary() {
        let dims = [8, 8, 8];
        let image = make_image(dims, 128.0);
        let phi = sphere_phi(dims, [4.0, 4.0, 4.0], 2.0);
        let result = LaplacianLevelSet::new().apply(&image, &phi).unwrap();
        let vals = image_values(&result);
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "Output voxel {} has non-binary value {}",
                i,
                v
            );
        }
    }

    /// Spatial metadata must be copied unchanged from image to output.
    #[test]
    fn test_metadata_preserved() {
        let dims = [6, 6, 6];
        let origin = [2.0, 3.0, 4.0];
        let spacing = [0.5, 0.5, 1.0];
        let image = make_image_with_metadata(dims, 128.0, origin, spacing);
        let phi = sphere_phi(dims, [3.0, 3.0, 3.0], 2.0);
        let result = LaplacianLevelSet::new().apply(&image, &phi).unwrap();
        assert_eq!(result.origin(), &Point::new(origin));
        assert_eq!(result.spacing(), &Spacing::new(spacing));
        assert_eq!(result.direction(), image.direction());
        assert_eq!(result.shape(), dims);
    }

    /// Mismatched image and phi shapes must return an Err.
    #[test]
    fn test_shape_mismatch_returns_error() {
        let image = make_image([5, 5, 5], 128.0);
        let phi = sphere_phi([3, 3, 3], [1.0, 1.0, 1.0], 1.0);
        let result = LaplacianLevelSet::new().apply(&image, &phi);
        assert!(result.is_err());
    }

    /// Default::default() must produce fields identical to new().
    #[test]
    fn test_default_matches_new() {
        let d = LaplacianLevelSet::default();
        let n = LaplacianLevelSet::new();
        assert_eq!(d.propagation_weight, n.propagation_weight);
        assert_eq!(d.curvature_weight, n.curvature_weight);
        assert_eq!(d.sigma, n.sigma);
        assert_eq!(d.dt, n.dt);
        assert_eq!(d.max_iterations, n.max_iterations);
        assert_eq!(d.tolerance, n.tolerance);
    }

    /// Contour must expand when the image has a bright local maximum (negative Laplacian).
    ///
    /// Analytical invariant for a 3-D Gaussian blob I(r) = exp(-r^2/(2*sigma^2)),
    /// sigma = 3.0, at the centre r = 0:
    ///
    ///   L = -3/sigma^2 = -3/9 = -0.333
    ///   F = L/(1+|L|) = -0.25
    ///   kappa = 2/R = 2/2 = 1.0  (sphere SDF, R=2)
    ///   speed = w_p*F + w_c*kappa = 1.0*(-0.25) + 0.1*1.0 = -0.15 < 0  => expansion
    ///
    /// Note: a uniform image (all same value) has zero Laplacian everywhere and
    /// therefore zero propagation speed; only a spatially varying bright region
    /// (negative Laplacian at maxima) drives expansion in this model.
    #[test]
    fn test_laplacian_expands_in_bright_region() {
        let dims = [15, 15, 15];
        let center = [7.0, 7.0, 7.0];
        let [nz, ny, nx] = dims;
        let n: usize = nz * ny * nx;
        // Gaussian blob: I(r) = exp(-r^2 / (2*3^2)).
        // Discrete Laplacian at centre: L approx -3/9 = -0.333, F approx -0.25.
        let sigma_img: f64 = 3.0;
        let mut img_data = vec![0.0_f32; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let r2 = (iz as f64 - center[0]).powi(2)
                        + (iy as f64 - center[1]).powi(2)
                        + (ix as f64 - center[2]).powi(2);
                    img_data[iz * ny * nx + iy * nx + ix] =
                        f64::exp(-r2 / (2.0 * sigma_img * sigma_img)) as f32;
                }
            }
        }
        let device = Default::default();
        let img_tensor =
            Tensor::<B, 3>::from_data(TensorData::new(img_data, Shape::new(dims)), &device);
        let image = Image::new(
            img_tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );
        let phi = sphere_phi(dims, center, 2.0);
        let initial_inside = count_phi_inside(&phi);
        let mut ls = LaplacianLevelSet::new();
        ls.propagation_weight = 1.0;
        ls.curvature_weight = 0.1;
        ls.sigma = 0.5;
        ls.max_iterations = 200;
        let result = ls.apply(&image, &phi).unwrap();
        let final_fg = count_foreground(&result);
        assert!(
            final_fg > initial_inside,
            "Laplacian level set must expand in bright region: final {} > initial {}",
            final_fg,
            initial_inside
        );
    }

    /// With max_iterations = 0 the PDE loop never executes.
    /// Output must equal threshold of the initial phi.
    #[test]
    fn test_zero_iterations_returns_initial_phi_thresholded() {
        let dims = [7, 7, 7];
        let center = [3.0, 3.0, 3.0];
        let image = make_image(dims, 1.0);
        let phi = sphere_phi(dims, center, 2.0);
        let expected_inside = count_phi_inside(&phi);
        let mut ls = LaplacianLevelSet::new();
        ls.max_iterations = 0;
        let result = ls.apply(&image, &phi).unwrap();
        let actual_fg = count_foreground(&result);
        assert_eq!(
            actual_fg,
            expected_inside,
            "Zero iterations: fg count must equal initial phi threshold: {} == {}",
            actual_fg,
            expected_inside
        );
    }
}
