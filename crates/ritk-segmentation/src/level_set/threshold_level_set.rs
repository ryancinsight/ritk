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
use ritk_image::tensor::{Backend, Tensor};
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
        image: &Image<f32, B, 3>,
        initial_phi: &Image<f32, B, 3>,
    ) -> anyhow::Result<Image<f32, B, 3>> {
        let dims = helpers::checked_dims(image.shape(), initial_phi.shape())?;

        let device = B::default();

        // Extract f32 tensor data and convert to f64 for PDE pipeline.
        let (img_vals, _) = extract_vec(image)?;
        let (phi_init, _) = extract_vec(initial_phi)?;
        let img_wide: Vec<f64> = img_vals.iter().map(|&v| v as f64).collect();
        let phi: Vec<f64> = phi_init.iter().map(|&v| v as f64).collect();

        let speed_field = self.sign_folded_speed(&img_wide);

        let phi = helpers::evolve_to_convergence::<helpers::MaxAbsRate, _, _>(
            phi,
            dims,
            self.dt,
            self.max_iterations,
            self.tolerance,
            |_phi, _scratch| {},
            |idx, grad_phi_mag, scratch| {
                // dphi = dt * |grad phi| * (w_p * S + w_c * kappa),
                // with S = -T so the bracket reproduces w_c*kappa - w_p*T.
                let speed = self.propagation_weight * speed_field[idx]
                    + self.curvature_weight * scratch.kappa[idx];
                self.dt * grad_phi_mag * speed
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

    /// Apply threshold level set segmentation to Coeus-native images.
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

        let speed_field = self.sign_folded_speed(&img_wide);

        let phi = helpers::evolve_to_convergence::<helpers::MaxAbsRate, _, _>(
            phi,
            dims,
            self.dt,
            self.max_iterations,
            self.tolerance,
            |_phi, _scratch| {},
            |idx, grad_phi_mag, scratch| {
                let speed = self.propagation_weight * speed_field[idx]
                    + self.curvature_weight * scratch.kappa[idx];
                self.dt * grad_phi_mag * speed
            },
        );

        crate::native_output::from_values(image, helpers::binary_mask(&phi), backend)
    }
}

impl Default for ThresholdLevelSet {
    fn default() -> Self {
        Self::new(0.0, 255.0)
    }
}

impl ThresholdLevelSet {
    /// The threshold sign field `S = -T` for the shared evolution engine.
    ///
    /// `T = +1` inside `[lower, upper]`, `-1` outside; the PDE carries
    /// `- w_p * T`, and the engine's bracket is `w_p * S + w_c * kappa`.
    /// Folding the minus sign into `S` up front is exact under IEEE-754:
    /// `w_p * (-T) = -(w_p * T)`, and `y + (-(x))` rounds identically to
    /// `y - x`, so every increment matches the pre-engine arithmetic
    /// bit-for-bit.
    fn sign_folded_speed(&self, img_wide: &[f64]) -> Vec<f64> {
        img_wide
            .iter()
            .map(|&v| {
                if self.lower_threshold <= v && v <= self.upper_threshold {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "tests_threshold_level_set.rs"]
mod tests;
