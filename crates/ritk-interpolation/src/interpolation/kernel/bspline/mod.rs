//! B-Spline interpolation implementation.
//!
//! This module provides cubic B-Spline interpolation for smooth sampling
//! of image values at continuous coordinates.

use super::BoundsPolicy;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use ritk_core::interpolation::Interpolator;

mod flat;
mod legacy;
mod prefilter;

#[cfg(test)]
mod tests;

/// Cubic B-Spline basis function.
///
/// The cubic B-Spline kernel is defined as:
/// - (2/3) - |x|^2 + (1/2)|x|^3 for |x| < 1
/// - (1/6)(2 - |x|)^3 for 1 <= |x| < 2
/// - 0 otherwise
///
/// Inlined version of cubic B-spline basis function for performance.
/// Uses multiplication instead of powi for better optimization.
#[inline(always)]
pub(crate) fn cubic_bspline(x: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x < 1.0 {
        (2.0 / 3.0) - abs_x * abs_x + 0.5 * abs_x * abs_x * abs_x
    } else if abs_x < 2.0 {
        let two_minus_x = 2.0 - abs_x;
        (1.0 / 6.0) * two_minus_x * two_minus_x * two_minus_x
    } else {
        0.0
    }
}

/// Cubic B-Spline interpolator.
///
/// Provides smooth interpolation using cubic B-Spline basis functions.
///
/// The image is first prefiltered into B-spline coefficients (see
/// [`prefilter`]), so reconstruction interpolates the samples exactly at the grid
/// points rather than smoothing them.
///
/// When [`BoundsPolicy::Extend`] (the default), support taps outside the volume
/// use whole-sample **mirror** boundary conditions — matching ITK's B-spline
/// interpolator and the mirror boundary used by the coefficient prefilter.
/// When [`BoundsPolicy::ZeroPad`], query coordinates outside the valid voxel
/// range `[0, dim-1]` return `0.0` immediately and out-of-bounds support taps
/// contribute zero, matching \[`LinearInterpolator`\] and
/// \[`NearestNeighborInterpolator`\] in zero-pad mode.
#[derive(Debug, Clone, Copy)]
pub struct BSplineInterpolator {
    /// Boundary handling policy. Default: `Extend`.
    pub bounds_policy: BoundsPolicy,
}

impl BSplineInterpolator {
    /// Create a new B-Spline interpolator with edge-renormalization (default).
    pub fn new() -> Self {
        Self {
            bounds_policy: BoundsPolicy::Extend,
        }
    }

    /// Create a B-Spline interpolator that returns `0.0` for out-of-bounds query coordinates.
    pub fn new_zero_pad() -> Self {
        Self {
            bounds_policy: BoundsPolicy::ZeroPad,
        }
    }

    /// Builder-style setter for the bounds policy.
    pub fn with_bounds_policy(mut self, policy: BoundsPolicy) -> Self {
        self.bounds_policy = policy;
        self
    }
}

impl Default for BSplineInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Interpolator<B> for BSplineInterpolator {
    fn interpolate<const D: usize>(
        &self,
        data: &Tensor<B, D>,
        indices: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let device = indices.device();
        let [n_points, rank] = indices.dims();
        assert_eq!(rank, D, "Indices rank must match data dimensionality");
        assert!(
            D == 2 || D == 3,
            "B-Spline interpolation only supports 2D and 3D"
        );

        let shape = data.shape();
        let dims: Vec<usize> = shape.dims;

        // Pre-extract the volume as a flat f32 slice, then recover the separable
        // B-spline coefficients once (O(volume) prefilter). Sampling the
        // coefficients — rather than the raw samples — is what makes this true
        // interpolation (exact at grid points) instead of a smoothing
        // approximation; it is also O(1) per query point.
        let volume_data = data.to_data();
        let volume_slice: &[f32] = volume_data
            .as_slice::<f32>()
            .expect("Volume data must be f32");
        let coeffs = prefilter::compute_coefficients(volume_slice, &dims);

        // Get all index data at once
        let indices_data = indices.to_data();
        let indices_slice: &[f32] = indices_data.as_slice::<f32>().expect("Indices must be f32");

        if n_points == 0 {
            return Tensor::zeros([0], &device);
        }

        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let coords_start = i * D;
            let value = match D {
                3 => flat::interpolate_point_3d_flat(
                    &coeffs,
                    &indices_slice[coords_start..coords_start + D],
                    &dims,
                    self.bounds_policy.as_out_of_bounds_mode(),
                ),
                2 => flat::interpolate_point_2d_flat(
                    &coeffs,
                    &indices_slice[coords_start..coords_start + D],
                    &dims,
                    self.bounds_policy.as_out_of_bounds_mode(),
                ),
                _ => unreachable!("D is asserted to be 2 or 3 above"),
            };
            results.push(value);
        }

        Tensor::<B, 1>::from_data(TensorData::new(results, [n_points]), &device)
    }
}
