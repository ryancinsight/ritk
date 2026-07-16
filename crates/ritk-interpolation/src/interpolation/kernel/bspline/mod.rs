//! B-Spline interpolation implementation.

use super::BoundsPolicy;
use ritk_core::interpolation::Interpolator;
use coeus_core::Backend;
use coeus_tensor::Tensor;

mod flat;
mod prefilter;

#[cfg(test)]
mod tests;

pub fn bspline_decomposition_coefficients(volume: &[f32], dims: &[usize]) -> Vec<f32> {
    prefilter::compute_coefficients(volume, dims)
}

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

#[derive(Debug, Clone, Copy)]
pub struct BSplineInterpolator {
    pub bounds_policy: BoundsPolicy,
}

impl BSplineInterpolator {
    pub fn new() -> Self {
        Self {
            bounds_policy: BoundsPolicy::Extend,
        }
    }

    pub fn new_zero_pad() -> Self {
        Self {
            bounds_policy: BoundsPolicy::ZeroPad,
        }
    }

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

impl<B> Interpolator<B> for BSplineInterpolator
where
    B: Backend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    fn interpolate(&self, data: &Tensor<f32, B>, indices: Tensor<f32, B>) -> Tensor<f32, B> {
        let backend = B::default();
        let shape = data.shape();
        let rank = shape.len();
        assert!(matches!(rank, 2 | 3), "B-Spline interpolation only supports 2D and 3D");

        let idx_shape = indices.shape();
        assert_eq!(idx_shape.len(), 2, "indices must be rank 2");
        assert_eq!(idx_shape[1], rank, "indices trailing dim must match tensor rank");

        let coeffs = prefilter::compute_coefficients(data.as_slice(), &shape);
        let idx = indices.as_slice().to_vec();
        let n_points = idx_shape[0];
        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let start = i * rank;
            let value = match rank {
                2 => flat::interpolate_point_2d_flat(
                    &coeffs,
                    &idx[start..start + rank],
                    &shape,
                    self.bounds_policy.as_out_of_bounds_mode(),
                ),
                3 => flat::interpolate_point_3d_flat(
                    &coeffs,
                    &idx[start..start + rank],
                    &shape,
                    self.bounds_policy.as_out_of_bounds_mode(),
                ),
                _ => unreachable!(),
            };
            results.push(value);
        }

        Tensor::<f32, B>::from_slice_on([n_points], &results, &backend)
    }
}
