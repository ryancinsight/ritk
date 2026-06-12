//! Threshold Level Set segmentation for 3-D medical images.
//!
//! Evolves a level set function with a speed function driven by intensity
//! thresholds: the contour expands where the image intensity is within
//! [lower_threshold, upper_threshold] and contracts elsewhere.
//!
//! PDE: d_phi/dt = |grad_phi| * (w_c * kappa - w_p * T(I))
//!   where T(I) = +1 if lower <= I <= upper, else -1.
//!
//! Reference: Whitaker, R.T. (1998). "A Level-Set Approach to 3D
//! Reconstruction from Range Data." IJCV.

use super::helpers;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;
/// Threshold Level Set segmentation parameters.
///
/// The contour expands where image intensity lies within
/// `[lower_threshold, upper_threshold]` and contracts elsewhere.
///
/// # PDE
///
/// d_phi/dt = |grad_phi| * (curvature_weight * kappa - propagation_weight * T(I(x)))
///
/// where:
/// - kappa = mean curvature = div(grad_phi / |grad_phi|)
/// - T(I) = +1.0 if lower_threshold <= I <= upper_threshold, else -1.0
///
/// With propagation_weight > 0 and T = +1 (inside threshold range),
/// the term -propagation_weight * T is negative, so phi decreases
/// and the contour (phi < 0 region) expands.
///
/// # Convergence
///
/// Iteration terminates when max |dphi| / dt < tolerance, or
/// iteration == max_iterations.
#[derive(Debug, Clone)]
pub struct ThresholdLevelSet {
    /// Lower bound of the intensity threshold range.
    pub lower_threshold: f64,
    /// Upper bound of the intensity threshold range.
    pub upper_threshold: f64,
    /// Weight on the propagation (balloon) term. Positive expands inside range.
    pub propagation_weight: f64,
    /// Weight on the curvature regularisation term.
    pub curvature_weight: f64,
    /// Euler forward time step.
    pub dt: f64,
    /// Maximum number of PDE iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on max |dphi|/dt.
    pub tolerance: f64,
}

impl ThresholdLevelSet {
    /// Construct with the given intensity threshold range and default PDE parameters.
    ///
    /// | Parameter            | Default |
    /// |----------------------|---------|
    /// | `propagation_weight` | 1.0     |
    /// | `curvature_weight`   | 0.2     |
    /// | `dt`                 | 0.05    |
    /// | `max_iterations`     | 200     |
    /// | `tolerance`          | 1e-3    |
    pub fn new(lower: f64, upper: f64) -> Self {
        Self {
            lower_threshold: lower,
            upper_threshold: upper,
            propagation_weight: 1.0,
            curvature_weight: 0.2,
            dt: 0.05,
            max_iterations: 200,
            tolerance: 1e-3,
        }
    }

    /// Apply threshold level set segmentation to a 3-D image.
    ///
    /// # Arguments
    /// - `image`: input scalar 3-D image.
    /// - `initial_phi`: initial level set function (same shape as `image`).
    ///   phi < 0 inside the initial contour, phi > 0 outside.
    ///
    /// # Returns
    /// Binary mask image: 1.0 where phi < 0 (inside), 0.0 elsewhere.
    /// Metadata (origin, spacing, direction) is preserved from `image`.
    ///
    /// # Errors
    /// Returns `Err` if shapes mismatch or tensor data cannot be read as f32.
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

        // Extract f32 tensor data and convert to f64 for PDE pipeline.
        let (img_vals, _) = extract_vec(image)?;
        let (phi_init, _) = extract_vec(initial_phi)?;
        let img_wide: Vec<f64> = img_vals.iter().map(|&v| v as f64).collect();
        let mut phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();

        // Precompute threshold speed field T(I).
        let threshold_speed: Vec<f64> = img_wide
            .iter()
            .map(|&v| {
                if self.lower_threshold <= v && v <= self.upper_threshold {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        // Scratch buffer.
        let mut kappa = vec![0.0_f64; n];

        // PDE evolution loop.
        for _iter in 0..self.max_iterations {
            helpers::compute_curvature_into(&phi, dims, &mut kappa);

            let mut max_change: f64 = 0.0;

            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let i = iz * ny * nx + iy * nx + ix;
                        let zz = iz as isize;
                        let yy = iy as isize;
                        let xx = ix as isize;

                        // Central-difference gradient of phi.
                        let dphi_z = (phi[helpers::idx_clamped(zz + 1, yy, xx, nz, ny, nx)]
                            - phi[helpers::idx_clamped(zz - 1, yy, xx, nz, ny, nx)])
                            * 0.5;
                        let dphi_y = (phi[helpers::idx_clamped(zz, yy + 1, xx, nz, ny, nx)]
                            - phi[helpers::idx_clamped(zz, yy - 1, xx, nz, ny, nx)])
                            * 0.5;
                        let dphi_x = (phi[helpers::idx_clamped(zz, yy, xx + 1, nz, ny, nx)]
                            - phi[helpers::idx_clamped(zz, yy, xx - 1, nz, ny, nx)])
                            * 0.5;
                        let grad_phi_mag =
                            (dphi_z * dphi_z + dphi_y * dphi_y + dphi_x * dphi_x).sqrt();

                        let speed = self.curvature_weight * kappa[i]
                            - self.propagation_weight * threshold_speed[i];
                        let dphi = self.dt * grad_phi_mag * speed;
                        phi[i] += dphi;

                        let change = dphi.abs() / self.dt;
                        if change > max_change {
                            max_change = change;
                        }
                    }
                }
            }

            if max_change < self.tolerance {
                break;
            }
        }

        // Binary mask: phi < 0 => 1.0, else 0.0.
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

impl Default for ThresholdLevelSet {
    fn default() -> Self {
        Self::new(0.0, 255.0)
    }
}

#[cfg(test)]
#[path = "tests_threshold_level_set.rs"]
mod tests;
