//! B-Spline interpolation implementation.
//!
//! This module provides cubic B-Spline interpolation for smooth sampling
//! of image values at continuous coordinates.

use super::trait_::Interpolator;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

mod flat;
mod legacy;

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
pub(super) fn cubic_bspline(x: f32) -> f32 {
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
/// When `zero_pad` is `false` (the default), out-of-bounds neighborhood
/// samples are skipped and the remaining in-bounds weights are renormalized,
/// which produces an edge-continuation effect at volume boundaries.
/// When `zero_pad` is `true`, query coordinates that fall outside the valid
/// voxel range `[0, dim-1]` for any dimension return `0.0` immediately,
/// matching the behavior of \[`LinearInterpolator`\] and
/// \[`NearestNeighborInterpolator`\] in zero-pad mode.
#[derive(Debug, Clone, Copy)]
pub struct BSplineInterpolator {
    /// If `true`, samples outside the volume boundary return `0.0` instead of
    /// the renormalized edge value. Mirrors \[`LinearInterpolator::zero_pad`\]
    /// and \[`NearestNeighborInterpolator::zero_pad`\].
    pub zero_pad: bool,
}

impl BSplineInterpolator {
    /// Create a new B-Spline interpolator with edge-renormalization (default).
    pub fn new() -> Self {
        Self { zero_pad: false }
    }

    /// Create a B-Spline interpolator that returns `0.0` for out-of-bounds query coordinates.
    pub fn new_zero_pad() -> Self {
        Self { zero_pad: true }
    }

    /// Builder-style setter for the `zero_pad` option.
    pub fn with_zero_pad(mut self, zero_pad: bool) -> Self {
        self.zero_pad = zero_pad;
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

        // Pre-extract the volume data as a flat f32 slice — O(1) per point instead of
        // O(volume_size) per neighborhood sample. This is the core Sprint 293 optimization:
        // it replaces 64 (3-D) or 16 (2-D) `data.clone().slice(…)` calls per query point
        // with a single `to_data()` call and pure-Rust scalar indexing.
        let volume_data = data.clone().to_data();
        let volume_slice: &[f32] = volume_data
            .as_slice::<f32>()
            .expect("Volume data must be f32");

        // Get all index data at once
        let indices_data = indices.to_data();
        let indices_slice: &[f32] = indices_data.as_slice::<f32>().expect("Indices must be f32");

        if n_points == 0 {
            return Tensor::zeros([0], &device);
        }

        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let coords_start = i * D;
            let value = if D == 3 {
                flat::interpolate_point_3d_flat(
                    volume_slice,
                    &indices_slice[coords_start..coords_start + D],
                    &dims,
                    self.zero_pad,
                )
            } else {
                flat::interpolate_point_2d_flat(
                    volume_slice,
                    &indices_slice[coords_start..coords_start + D],
                    &dims,
                    self.zero_pad,
                )
            };
            results.push(value);
        }

        Tensor::<B, 1>::from_data(TensorData::new(results, [n_points]), &device)
    }
}
