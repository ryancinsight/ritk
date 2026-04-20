//! Shape Detection level set segmentation for 3-D medical images.
//!
//! # Mathematical Specification
//!
//! This module implements a front-propagation level set that is driven by an
//! edge-stopping function derived from the image gradient magnitude. The
//! evolution follows the shape-detection model of Malladi, Sethian, and Vemuri
//! (1995).
//!
//! Let `I` be the input image, `φ` the level set function, and
//! `g(|∇I|) = 1 / (1 + (|∇I| / k)^2)` the edge-stopping function.
//!
//! The discretised evolution is:
//!
//! ```text
//! ∂φ/∂t = g · (w_c · κ - w_p) · |∇φ| - w_a · ∇g · ∇φ
//! ```
//!
//! where:
//! - `κ = div(∇φ / |∇φ|)` is mean curvature,
//! - `w_p > 0` drives outward propagation in homogeneous regions,
//! - `w_c > 0` regularises the contour by curvature,
//! - `w_a > 0` attracts the front toward image edges.
//!
//! The implementation uses:
//! - clamped boundary conditions,
//! - central finite differences,
//! - shared numerical helpers from `helpers.rs`,
//! - `f64` for the PDE evolution pipeline.
//!
//! The final output is a binary mask obtained by thresholding `φ < 0`.
//!
//! # References
//!
//! - Malladi, R., Sethian, J. A., & Vemuri, B. C. (1995).
//!   "Shape Modeling with Front Propagation: A Level Set Approach."
//!   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 17(2).

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

use super::helpers;

/// Shape Detection level set segmentation.
///
/// Evolves an initial level set function toward object boundaries using
/// edge-stopping, curvature regularisation, and propagation in a single
/// PDE.
#[derive(Debug, Clone)]
pub struct ShapeDetectionSegmentation {
    /// Weight on the curvature regularisation term.
    pub curvature_weight: f64,
    /// Weight on the propagation term.
    pub propagation_weight: f64,
    /// Weight on the edge attraction term.
    pub advection_weight: f64,
    /// Edge stopping sensitivity parameter `k`.
    pub edge_k: f64,
    /// Gaussian pre-smoothing standard deviation for the input image.
    pub sigma: f64,
    /// Euler forward time step.
    pub dt: f64,
    /// Maximum number of PDE iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on `max |Δφ| / dt`.
    pub tolerance: f64,
}

impl ShapeDetectionSegmentation {
    /// Construct with default parameters.
    ///
    /// | Parameter            | Default |
    /// |----------------------|---------|
    /// | `curvature_weight`   | 1.0     |
    /// | `propagation_weight` | 1.0     |
    /// | `advection_weight`   | 1.0     |
    /// | `edge_k`             | 1.0     |
    /// | `sigma`              | 1.0     |
    /// | `dt`                 | 0.05    |
    /// | `max_iterations`     | 200     |
    /// | `tolerance`          | 1e-3    |
    pub fn new() -> Self {
        Self {
            curvature_weight: 1.0,
            propagation_weight: 1.0,
            advection_weight: 1.0,
            edge_k: 1.0,
            sigma: 1.0,
            dt: 0.05,
            max_iterations: 200,
            tolerance: 1e-3,
        }
    }

    /// Apply shape-detection segmentation to a 3-D image with an initial level set.
    ///
    /// `initial_phi` defines the starting contour; the output mask is `1.0` where
    /// the converged level set is negative and `0.0` elsewhere.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be read as `f32` or the shapes do
    /// not match.
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
        let img_f32: Vec<f32> = img_td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("ShapeDetection requires f32 image data: {:?}", e))?
            .to_vec();

        let phi_td = initial_phi.data().clone().into_data();
        let phi_f32: Vec<f32> = phi_td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("ShapeDetection requires f32 phi data: {:?}", e))?
            .to_vec();

        let img_f64: Vec<f64> = img_f32.iter().map(|&v| v as f64).collect();
        let mut phi: Vec<f64> = phi_f32.iter().map(|&v| v as f64).collect();

        let smoothed = if self.sigma > 0.0 {
            helpers::gaussian_smooth_3d(&img_f64, dims, self.sigma)
        } else {
            img_f64.clone()
        };

        let grad_mag = helpers::compute_gradient_magnitude(&smoothed, dims);
        let g = helpers::compute_edge_stopping(&grad_mag, self.edge_k);
        let (g_z, g_y, g_x) = helpers::compute_field_gradient(&g, dims);

        let n = phi.len();
        let mut kappa = vec![0.0_f64; n];
        let mut phi_new = vec![0.0_f64; n];

        for _iter in 0..self.max_iterations {
            helpers::compute_curvature_into(&phi, dims, &mut kappa);
            let (phi_z, phi_y, phi_x) = helpers::compute_field_gradient(&phi, dims);

            let mut max_change = 0.0_f64;

            for i in 0..n {
                let grad_phi_mag =
                    (phi_z[i] * phi_z[i] + phi_y[i] * phi_y[i] + phi_x[i] * phi_x[i]).sqrt();

                let curvature = self.curvature_weight * g[i] * kappa[i] * grad_phi_mag;
                let propagation = self.propagation_weight * g[i] * grad_phi_mag;
                let advection = self.advection_weight
                    * (g_z[i] * phi_z[i] + g_y[i] * phi_y[i] + g_x[i] * phi_x[i]);

                let dphi = self.dt * (curvature - propagation - advection);
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

impl Default for ShapeDetectionSegmentation {
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
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
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

    #[test]
    fn test_shape_detection_expands_in_homogeneous_region() {
        let dims = [15, 15, 15];
        let center = [7.0, 7.0, 7.0];
        let image = make_image(dims, 40.0);
        let phi = sphere_phi(dims, center, 3.0);
        let initial_inside = count_phi_inside(&phi);

        let mut ls = ShapeDetectionSegmentation::new();
        ls.propagation_weight = 1.0;
        ls.curvature_weight = 0.1;
        ls.advection_weight = 0.0;
        ls.edge_k = 1.0;
        ls.max_iterations = 200;

        let result = ls.apply(&image, &phi).unwrap();
        let final_fg = count_foreground(&result);

        assert!(
            final_fg > initial_inside,
            "Shape detection must expand in homogeneous regions: final {} > initial {}",
            final_fg,
            initial_inside
        );
    }

    #[test]
    fn test_shape_detection_stops_at_edges() {
        let dims = [17, 17, 17];
        let center = [8.0, 8.0, 8.0];
        let image = make_image(dims, 0.0);
        let mut img_data = image_values(&image);
        let [nz, ny, nx] = dims;

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dist = ((iz as f64 - center[0]).powi(2)
                        + (iy as f64 - center[1]).powi(2)
                        + (ix as f64 - center[2]).powi(2))
                    .sqrt();
                    if (dist - 5.0).abs() < 0.5 {
                        img_data[iz * ny * nx + iy * nx + ix] = 255.0;
                    }
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

        let mut ls = ShapeDetectionSegmentation::new();
        ls.propagation_weight = 1.0;
        ls.curvature_weight = 0.2;
        ls.advection_weight = 0.5;
        ls.edge_k = 5.0;
        ls.sigma = 1.0;
        ls.max_iterations = 200;

        let result = ls.apply(&image, &phi).unwrap();
        let final_fg = count_foreground(&result);

        assert!(
            final_fg >= initial_inside,
            "Edge stopping must prevent collapse past the initial contour"
        );
    }

    #[test]
    fn test_output_is_binary() {
        let dims = [8, 8, 8];
        let image = make_image(dims, 128.0);
        let phi = sphere_phi(dims, [4.0, 4.0, 4.0], 2.0);

        let result = ShapeDetectionSegmentation::new()
            .apply(&image, &phi)
            .unwrap();
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

    #[test]
    fn test_metadata_preserved() {
        let dims = [6, 6, 6];
        let origin = [2.0, 3.0, 4.0];
        let spacing = [0.5, 0.5, 1.0];
        let image = make_image_with_metadata(dims, 128.0, origin, spacing);
        let phi = sphere_phi(dims, [3.0, 3.0, 3.0], 2.0);

        let result = ShapeDetectionSegmentation::new()
            .apply(&image, &phi)
            .unwrap();

        assert_eq!(result.origin(), &Point::new(origin));
        assert_eq!(result.spacing(), &Spacing::new(spacing));
        assert_eq!(result.direction(), image.direction());
        assert_eq!(result.shape(), dims);
    }

    #[test]
    fn test_shape_mismatch_returns_error() {
        let image = make_image([5, 5, 5], 128.0);
        let phi = sphere_phi([3, 3, 3], [1.0, 1.0, 1.0], 1.0);
        let result = ShapeDetectionSegmentation::new().apply(&image, &phi);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_matches_new() {
        let d = ShapeDetectionSegmentation::default();
        let n = ShapeDetectionSegmentation::new();

        assert_eq!(d.curvature_weight, n.curvature_weight);
        assert_eq!(d.propagation_weight, n.propagation_weight);
        assert_eq!(d.advection_weight, n.advection_weight);
        assert_eq!(d.edge_k, n.edge_k);
        assert_eq!(d.sigma, n.sigma);
        assert_eq!(d.dt, n.dt);
        assert_eq!(d.max_iterations, n.max_iterations);
        assert_eq!(d.tolerance, n.tolerance);
    }
}
