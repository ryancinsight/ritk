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
use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
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
        let img_td = image.data().clone().into_data();
        let img_f32: Vec<f32> = img_td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("ThresholdLevelSet requires f32 image data: {:?}", e))?
            .to_vec();
        let img_f64: Vec<f64> = img_f32.iter().map(|&v| v as f64).collect();

        let phi_td = initial_phi.data().clone().into_data();
        let phi_f32: Vec<f32> = phi_td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("ThresholdLevelSet requires f32 phi data: {:?}", e))?
            .to_vec();
        let mut phi: Vec<f64> = phi_f32.iter().map(|&v| v as f64).collect();

        // Precompute threshold speed field T(I).
        let threshold_speed: Vec<f64> = img_f64
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
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

impl Default for ThresholdLevelSet {
    fn default() -> Self {
        Self::new(0.0, 255.0)
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
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(vec![val; n], Shape::new(dims)),
            &device,
        );
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
        let tensor = Tensor::<B, 3>::from_data(
            TensorData::new(vec![val; n], Shape::new(dims)),
            &device,
        );
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

    fn make_bimodal_image(
        dims: [usize; 3],
        center: [f64; 3],
        radius: f64,
        inside_val: f32,
        outside_val: f32,
    ) -> Image<B, 3> {
        let n: usize = dims.iter().product();
        let [nz, ny, nx] = dims;
        let mut data = vec![outside_val; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dist = ((iz as f64 - center[0]).powi(2)
                        + (iy as f64 - center[1]).powi(2)
                        + (ix as f64 - center[2]).powi(2))
                    .sqrt();
                    if dist <= radius {
                        data[iz * ny * nx + iy * nx + ix] = inside_val;
                    }
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

    fn get_values(image: &Image<B, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn count_foreground(image: &Image<B, 3>) -> usize {
        get_values(image).iter().filter(|&&v| v == 1.0).count()
    }

    fn count_phi_inside(phi: &Image<B, 3>) -> usize {
        get_values(phi).iter().filter(|&&v| v < 0.0).count()
    }

    #[test]
    fn test_threshold_expands_within_range() {
        let dims = [15, 15, 15];
        let center = [7.0, 7.0, 7.0];
        let image = make_bimodal_image(dims, center, 4.0, 150.0, 0.0);
        let phi = sphere_phi(dims, center, 3.0);
        let initial_inside = count_phi_inside(&phi);

        let mut ls = ThresholdLevelSet::new(100.0, 200.0);
        ls.propagation_weight = 1.0;
        ls.curvature_weight = 0.1;
        ls.max_iterations = 300;

        let result = ls.apply(&image, &phi).unwrap();
        let final_fg = count_foreground(&result);

        assert!(
            final_fg > initial_inside,
            "Contour must expand within threshold range: final {} > initial {}",
            final_fg,
            initial_inside
        );
    }

    #[test]
    fn test_threshold_contracts_outside_range() {
        let dims = [11, 11, 11];
        let center = [5.0, 5.0, 5.0];
        // Both intensities (50.0 and 300.0) are OUTSIDE threshold range [100, 250].
        let image = make_bimodal_image(dims, center, 3.0, 50.0, 300.0);
        let phi = sphere_phi(dims, center, 3.0);
        let initial_inside = count_phi_inside(&phi);

        let ls = ThresholdLevelSet::new(100.0, 250.0);
        let result = ls.apply(&image, &phi).unwrap();
        let final_fg = count_foreground(&result);

        assert!(
            final_fg < initial_inside,
            "Contour must contract outside threshold range: final {} < initial {}",
            final_fg,
            initial_inside
        );
    }

    #[test]
    fn test_output_is_binary() {
        let dims = [8, 8, 8];
        let image = make_image(dims, 128.0);
        let phi = sphere_phi(dims, [4.0, 4.0, 4.0], 2.0);

        let result = ThresholdLevelSet::new(100.0, 200.0).apply(&image, &phi).unwrap();
        let vals = get_values(&result);

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

        let result = ThresholdLevelSet::new(100.0, 200.0).apply(&image, &phi).unwrap();

        assert_eq!(result.origin(), &Point::new(origin), "Origin must be preserved");
        assert_eq!(result.spacing(), &Spacing::new(spacing), "Spacing must be preserved");
        assert_eq!(result.direction(), image.direction(), "Direction must be preserved");
        assert_eq!(result.shape(), dims, "Shape must be preserved");
    }

    #[test]
    fn test_shape_mismatch_returns_error() {
        let image = make_image([5, 5, 5], 128.0);
        let phi = sphere_phi([3, 3, 3], [1.0, 1.0, 1.0], 1.0);
        let result = ThresholdLevelSet::new(100.0, 200.0).apply(&image, &phi);
        assert!(result.is_err(), "Shape mismatch must return Err");
    }

    #[test]
    fn test_default_matches_new() {
        let d = ThresholdLevelSet::default();
        let n = ThresholdLevelSet::new(0.0, 255.0);

        assert_eq!(d.lower_threshold, n.lower_threshold);
        assert_eq!(d.upper_threshold, n.upper_threshold);
        assert_eq!(d.propagation_weight, n.propagation_weight);
        assert_eq!(d.curvature_weight, n.curvature_weight);
        assert_eq!(d.dt, n.dt);
        assert_eq!(d.max_iterations, n.max_iterations);
        assert_eq!(d.tolerance, n.tolerance);
    }

    #[test]
    fn test_uniform_in_range_expands() {
        let dims = [11, 11, 11];
        let center = [5.0, 5.0, 5.0];
        let image = make_image(dims, 128.0);
        let phi = sphere_phi(dims, center, 3.0);
        let initial_inside = count_phi_inside(&phi);

        let mut ls = ThresholdLevelSet::new(100.0, 200.0);
        ls.propagation_weight = 1.0;
        ls.curvature_weight = 0.1;
        ls.max_iterations = 200;

        let result = ls.apply(&image, &phi).unwrap();
        let final_fg = count_foreground(&result);

        assert!(
            final_fg > initial_inside,
            "Uniform in-range image must cause expansion: final {} > initial {}",
            final_fg,
            initial_inside
        );
    }
}
