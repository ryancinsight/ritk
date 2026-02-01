//! B-Spline interpolation implementation.
//!
//! This module provides cubic B-Spline interpolation for smooth sampling
//! of image values at continuous coordinates.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use super::trait_::Interpolator;

/// Cubic B-Spline basis function.
///
/// The cubic B-Spline kernel is defined as:
/// - (2/3) - |x|^2 + (1/2)|x|^3    for |x| < 1
/// - (1/6)(2 - |x|)^3              for 1 <= |x| < 2
/// - 0                             otherwise
fn cubic_bspline(x: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x < 1.0 {
        (2.0 / 3.0) - abs_x.powi(2) + 0.5 * abs_x.powi(3)
    } else if abs_x < 2.0 {
        let two_minus_x = 2.0 - abs_x;
        (1.0 / 6.0) * two_minus_x.powi(3)
    } else {
        0.0
    }
}

/// Cubic B-Spline interpolator.
///
/// Provides smooth interpolation using cubic B-Spline basis functions.
#[derive(Debug, Clone, Copy)]
pub struct BSplineInterpolator;

impl BSplineInterpolator {
    /// Create a new B-Spline interpolator.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BSplineInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Interpolator<B> for BSplineInterpolator {
    fn interpolate<const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let device = indices.device();
        let [n_points, rank] = indices.dims();
        assert_eq!(rank, D, "Indices rank must match data dimensionality");
        assert!(D == 2 || D == 3, "B-Spline interpolation only supports 2D and 3D");

        let shape = data.shape();
        let dims: Vec<usize> = shape.dims.into();

        // Get all index data at once
        let indices_data = indices.to_data();
        let indices_slice: &[f32] = indices_data.as_slice::<f32>().expect("Indices must be f32");

        // Process each point
        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            // Get coordinates for this point
            let coords_start = i * D;
            let coords: Vec<f32> = (0..D)
                .map(|d| indices_slice[coords_start + d])
                .collect();

            let value = if D == 3 {
                interpolate_point_3d(data, &coords, &dims, &device)
            } else {
                interpolate_point_2d(data, &coords, &dims, &device)
            };
            results.push(value);
        }

        if results.is_empty() {
            Tensor::zeros([0], &device)
        } else {
            Tensor::cat(results, 0)
        }
    }
}

/// 3D B-Spline interpolation for a single point.
fn interpolate_point_3d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    coords: &[f32],
    dims: &[usize],
    device: &B::Device,
) -> Tensor<B, 1> {
    let x = coords[0];
    let y = coords[1];
    let z = coords[2];

    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;
    let z0 = z.floor() as isize - 1;

    let mut result = Tensor::zeros([1], device);
    let mut weight_sum = 0.0f32;

    // Sample 4x4x4 neighborhood
    for dz in 0..4 {
        for dy in 0..4 {
            for dx in 0..4 {
                let xi = x0 + dx;
                let yi = y0 + dy;
                let zi = z0 + dz;

                // Compute B-Spline weights
                let wx = cubic_bspline(x - xi as f32);
                let wy = cubic_bspline(y - yi as f32);
                let wz = cubic_bspline(z - zi as f32);
                let weight = wx * wy * wz;

                // Check bounds and sample
                if xi >= 0 && xi < dims[0] as isize
                    && yi >= 0 && yi < dims[1] as isize
                    && zi >= 0 && zi < dims[2] as isize
                {
                    let sample = data.clone().slice([
                        xi as usize..xi as usize + 1,
                        yi as usize..yi as usize + 1,
                        zi as usize..zi as usize + 1,
                    ]);
                    let sample_scalar = sample.reshape([1]);
                    result = result.add(sample_scalar.mul_scalar(weight));
                    weight_sum += weight;
                }
            }
        }
    }

    // Normalize by weight sum
    if weight_sum > 0.0 {
        result = result.div_scalar(weight_sum);
    }

    result
}

/// 2D B-Spline interpolation for a single point.
fn interpolate_point_2d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    coords: &[f32],
    dims: &[usize],
    device: &B::Device,
) -> Tensor<B, 1> {
    let x = coords[0];
    let y = coords[1];

    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;

    let mut result = Tensor::zeros([1], device);
    let mut weight_sum = 0.0f32;

    // Sample 4x4 neighborhood
    for dy in 0..4 {
        for dx in 0..4 {
            let xi = x0 + dx;
            let yi = y0 + dy;

            // Compute B-Spline weights
            let wx = cubic_bspline(x - xi as f32);
            let wy = cubic_bspline(y - yi as f32);
            let weight = wx * wy;

            // Check bounds and sample
            if xi >= 0 && xi < dims[0] as isize
                && yi >= 0 && yi < dims[1] as isize
            {
                let sample = data.clone().slice([
                    xi as usize..xi as usize + 1,
                    yi as usize..yi as usize + 1,
                ]);
                let sample_scalar = sample.reshape([1]);
                result = result.add(sample_scalar.mul_scalar(weight));
                weight_sum += weight;
            }
        }
    }

    // Normalize by weight sum
    if weight_sum > 0.0 {
        result = result.div_scalar(weight_sum);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, ElementConversion};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_bspline_3d() {
        let device = Default::default();

        // Create a simple 3D volume
        let data = Tensor::<TestBackend, 3>::from_floats(
            [[
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ]],
            &device,
        );

        let interpolator = BSplineInterpolator::new();

        // Test at exact grid point
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar().elem::<f32>();
        assert!((val - 1.0).abs() < 0.1, "Expected ~1.0, got {}", val);

        // Test at interpolated point
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar().elem::<f32>();
        // Value should be between min and max
        assert!(val >= 0.0 && val <= 8.0, "Interpolated value {} out of range", val);
    }

    #[test]
    fn test_bspline_2d() {
        let device = Default::default();

        // Create a simple 2D image
        let data = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            &device,
        );

        let interpolator = BSplineInterpolator::new();

        // Test at exact grid point
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar().elem::<f32>();
        assert!((val - 1.0).abs() < 0.1, "Expected ~1.0, got {}", val);
    }

    #[test]
    fn test_bspline_basis() {
        // Test B-Spline basis properties
        assert!((cubic_bspline(0.0) - 2.0 / 3.0).abs() < 1e-6);
        assert!(cubic_bspline(1.0) > 0.0);
        assert_eq!(cubic_bspline(2.0), 0.0);
        assert_eq!(cubic_bspline(-2.0), 0.0);
        assert_eq!(cubic_bspline(3.0), 0.0);

        // Symmetry
        assert!((cubic_bspline(0.5) - cubic_bspline(-0.5)).abs() < 1e-6);
    }
}
