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

use super::helpers;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_filter::edge::GaussianSigma;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

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
    pub sigma: GaussianSigma,
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
            sigma: GaussianSigma::new_unchecked(1.0),
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

        let (img_vals, _) = extract_vec(image)?;
        let (phi_init, _) = extract_vec(initial_phi)?;
        let img_wide: Vec<f64> = img_vals.iter().map(|&v| v as f64).collect();
        let mut phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();

        let smoothed = helpers::smooth_or_borrow(&img_wide, dims, self.sigma.get());

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
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

impl Default for ShapeDetectionSegmentation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests_shape_detection.rs"]
mod tests;
