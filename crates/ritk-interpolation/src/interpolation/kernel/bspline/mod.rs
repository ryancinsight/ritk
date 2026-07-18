//! B-Spline interpolation implementation.
//!
//! This module provides cubic B-Spline interpolation for smooth sampling
//! of image values at continuous coordinates.

use super::BoundsPolicy;
use coeus_core::{Backend, CpuAddressableStorage};
use coeus_tensor::Tensor;
use ritk_core::interpolation::Interpolator;

mod flat;
mod prefilter;

#[cfg(test)]
mod tests;

/// Recover the separable **cubic** B-spline interpolation coefficients of a flat
/// row-major volume of shape `dims`, with whole-sample mirror boundary
/// conditions. This is the standalone decomposition (prefilter) step, matching
/// ITK's `BSplineDecompositionImageFilter` at spline order 3. A degenerate
/// (size-1) axis is skipped (its coefficients equal the samples).
pub fn bspline_decomposition_coefficients(volume: &[f32], dims: &[usize]) -> Vec<f32> {
    prefilter::compute_coefficients(volume, dims)
}

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
/// `prefilter`) so reconstruction interpolates the samples exactly at the grid
/// points rather than smoothing them.
///
/// When [`BoundsPolicy::Extend`] (the default), support taps outside the volume
/// use whole-sample **mirror** boundary conditions — matching ITK's B-spline
/// interpolator and the mirror boundary used by the coefficient prefilter.
/// When [`BoundsPolicy::ZeroPad`], query coordinates outside the valid voxel
/// range `[0, dim-1]` return `0.0` immediately and out-of-bounds support taps
/// contribute zero, matching [`crate::LinearInterpolator`] and
/// [`crate::NearestNeighborInterpolator`] in zero-pad mode.
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

impl<B: Backend> Interpolator<B> for BSplineInterpolator
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    fn interpolate(&self, data: &Tensor<f32, B>, indices: Tensor<f32, B>) -> Tensor<f32, B> {
        let shape = data.shape().to_vec();
        let rank = shape.len();
        assert!(
            matches!(rank, 2 | 3),
            "B-Spline interpolation only supports 2D and 3D data"
        );

        let idx_shape = indices.shape();
        assert_eq!(idx_shape.len(), 2, "indices must be a 2D tensor [N, rank]");
        let n_points = idx_shape[0];
        let idx_rank = idx_shape[1];
        assert_eq!(idx_rank, rank, "indices rank must match data rank");

        let data_contig = data.to_contiguous();
        let volume_slice = data_contig.as_slice();
        let coeffs = prefilter::compute_coefficients(volume_slice, &shape);

        let idx_contig = indices.to_contiguous();
        let idx_slice = idx_contig.as_slice();

        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let coords = &idx_slice[i * rank..(i + 1) * rank];
            let value = match rank {
                3 => flat::interpolate_point_3d_flat(
                    &coeffs,
                    coords,
                    &shape,
                    self.bounds_policy.as_out_of_bounds_mode(),
                ),
                2 => flat::interpolate_point_2d_flat(
                    &coeffs,
                    coords,
                    &shape,
                    self.bounds_policy.as_out_of_bounds_mode(),
                ),
                _ => unreachable!(),
            };
            results.push(value);
        }

        Tensor::from_slice([n_points], &results)
    }
}
