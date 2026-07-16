//! Scale transform implementation.
//!
//! This module provides a scale transform (scaling around a center).

use ritk_core::transform::Transform;
use coeus_core::CpuAddressableStorage;
use coeus_core::Backend;
use coeus_tensor::Tensor;

/// Scale Transform.
///
/// Represents a scaling transformation with a fixed center:
/// T(x) = S * (x - c) + c
///
/// where:
/// * S is a D-dimensional scale vector (diagonal matrix)
/// * c is a D-dimensional fixed center of scaling
#[derive(Clone)]
pub struct ScaleTransform<B: Backend, const D: usize> {
    scale: Tensor<f32, B>, // [D] scale factors
    center: Tensor<f32, B>,       // [D] fixed center
}

impl<B: Backend, const D: usize> ScaleTransform<B, D> {
    /// Create a new scale transform.
    ///
    /// # Arguments
    /// * `scale` - Tensor of shape `[D]` containing the scale factors
    /// * `center` - Tensor of shape `[D]` containing the fixed center
    pub fn new(scale: Tensor<f32, B>, center: Tensor<f32, B>) -> Self {
        Self { scale, center }
    }

    /// Create an identity scale transform (scale = 1.0).
    ///
    /// # Arguments
    /// * `center` - Optional center of scaling. If None, uses origin (0,0...0).
    /// * `device` - Device to create tensors on.
    pub fn identity(center: Option<Tensor<f32, B>>, device: &B) -> Self {
        let scale = Tensor::<f32, B>::ones_on([D], device);
        let center = center.unwrap_or_else(|| Tensor::<f32, B>::zeros_on([D], device));
        Self::new(scale, center)
    }

    /// Get the scale factors.
    pub fn scale(&self) -> Tensor<f32, B> {
        self.scale.clone()
    }

    /// Get the center of scaling.
    pub fn center(&self) -> Tensor<f32, B> {
        self.center.clone()
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for ScaleTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        let device = B::default();
        let points = points.to_contiguous();
        let scale = self.scale.to_contiguous();
        let center = self.center.to_contiguous();
        let batch = points.shape()[0];
        let point_data = points.as_slice();
        let scale_data = scale.as_slice();
        let center_data = center.as_slice();
        let mut output = vec![0.0f32; batch * D];

        for row in 0..batch {
            for dim in 0..D {
                let x = point_data[row * D + dim];
                let c = center_data[dim];
                output[row * D + dim] = scale_data[dim] * (x - c) + c;
            }
        }

        Tensor::<f32, B>::from_slice_on([batch, D], &output, &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    type B = SequentialBackend;

    /// Tolerance for scale transform exact-value arithmetic.
    /// Scale factors are exact floats; error bound is 4 × f32::EPSILON.
    const SCALE_TRANSFORM_TOL: f32 = f32::EPSILON * 4.0;

    #[test]
    fn test_scale_transform() {
        let device = Default::default();
        let scale = Tensor::<f32, B>::from_floats([2.0, 0.5, 1.0], &device); // Scale X by 2, Y by 0.5, Z by 1
        let center = Tensor::<f32, B>::zeros([3], &device);

        let transform = ScaleTransform::<B, 3>::new(scale, center);

        let points = Tensor::<f32, B>::from_floats([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], &device);

        let transformed = transform.transform_points(points);
        let data = transformed.into_data();
        let slice = data.as_slice::<f32>().unwrap();

        // Point 1: [1*2, 2*0.5, 3*1] = [2, 1, 3]
        assert!((slice[0] - 2.0).abs() < SCALE_TRANSFORM_TOL);
        assert!((slice[1] - 1.0).abs() < SCALE_TRANSFORM_TOL);
        assert!((slice[2] - 3.0).abs() < SCALE_TRANSFORM_TOL);

        // Point 2: [2*2, 4*0.5, 6*1] = [4, 2, 6]
        assert!((slice[3] - 4.0).abs() < SCALE_TRANSFORM_TOL);
        assert!((slice[4] - 2.0).abs() < SCALE_TRANSFORM_TOL);
        assert!((slice[5] - 6.0).abs() < SCALE_TRANSFORM_TOL);
    }

    #[test]
    fn test_scale_transform_with_center() {
        let device = Default::default();
        let scale = Tensor::<f32, B>::from_floats([2.0, 2.0], &device);
        let center = Tensor::<f32, B>::from_floats([1.0, 1.0], &device);

        let transform = ScaleTransform::<B, 2>::new(scale, center);

        // Point at center should not move
        let points = Tensor::<f32, B>::from_floats([[1.0, 1.0]], &device);
        let transformed = transform.transform_points(points);
        let data = transformed.into_data();
        let slice = data.as_slice::<f32>().unwrap();

        assert!((slice[0] - 1.0).abs() < SCALE_TRANSFORM_TOL);
        assert!((slice[1] - 1.0).abs() < SCALE_TRANSFORM_TOL);

        // Point at (2, 2). Relative to center (1, 1) is (1, 1).
        // Scale by 2 -> (2, 2). Add center -> (3, 3).
        let points = Tensor::<f32, B>::from_floats([[2.0, 2.0]], &device);
        let transformed = transform.transform_points(points);
        let data = transformed.into_data();
        let slice = data.as_slice::<f32>().unwrap();

        assert!((slice[0] - 3.0).abs() < SCALE_TRANSFORM_TOL);
        assert!((slice[1] - 3.0).abs() < SCALE_TRANSFORM_TOL);
    }
}
