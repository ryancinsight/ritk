//! Scale transform implementation.
//!
//! This module provides a scale transform (scaling around a center).

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::module::{Module, Param};
use super::trait_::Transform;

/// Scale Transform.
///
/// Represents a scaling transformation with a fixed center:
/// T(x) = S * (x - c) + c
///
/// where:
/// * S is a D-dimensional scale vector (diagonal matrix)
/// * c is a D-dimensional fixed center of scaling
#[derive(Module, Debug)]
pub struct ScaleTransform<B: Backend, const D: usize> {
    scale: Param<Tensor<B, 1>>, // [D] scale factors
    center: Tensor<B, 1>,       // [D] fixed center
}

impl<B: Backend, const D: usize> ScaleTransform<B, D> {
    /// Create a new scale transform.
    ///
    /// # Arguments
    /// * `scale` - Tensor of shape `[D]` containing the scale factors
    /// * `center` - Tensor of shape `[D]` containing the fixed center
    pub fn new(scale: Tensor<B, 1>, center: Tensor<B, 1>) -> Self {
        Self {
            scale: Param::from_tensor(scale),
            center,
        }
    }

    /// Create an identity scale transform (scale = 1.0).
    ///
    /// # Arguments
    /// * `center` - Optional center of scaling. If None, uses origin (0,0...0).
    /// * `device` - Device to create tensors on.
    pub fn identity(center: Option<Tensor<B, 1>>, device: &B::Device) -> Self {
        let scale = Tensor::<B, 1>::ones([D], device);
        let center = center.unwrap_or_else(|| Tensor::<B, 1>::zeros([D], device));
        Self::new(scale, center)
    }

    /// Get the scale factors.
    pub fn scale(&self) -> Tensor<B, 1> {
        self.scale.val()
    }

    /// Get the center of scaling.
    pub fn center(&self) -> Tensor<B, 1> {
        self.center.clone()
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for ScaleTransform<B, D> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        // points: [Batch, D]
        // scale (s): [D]
        // center (c): [D]
        //
        // T(x) = s * (x - c) + c
        // Element-wise multiplication since s is diagonal

        let c = self.center.clone().reshape([1, D]);
        let s = self.scale.val().reshape([1, D]);

        let centered = points - c.clone();
        
        // Element-wise multiplication broadcast
        let scaled = centered * s;
        
        scaled + c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_scale_transform() {
        let device = Default::default();
        let scale = Tensor::<B, 1>::from_floats([2.0, 0.5, 1.0], &device); // Scale X by 2, Y by 0.5, Z by 1
        let center = Tensor::<B, 1>::zeros([3], &device);
        
        let transform = ScaleTransform::<B, 3>::new(scale, center);
        
        let points = Tensor::<B, 2>::from_floats([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0]
        ], &device);
        
        let transformed = transform.transform_points(points);
        let data = transformed.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        
        // Point 1: [1*2, 2*0.5, 3*1] = [2, 1, 3]
        assert!((slice[0] - 2.0).abs() < 1e-6);
        assert!((slice[1] - 1.0).abs() < 1e-6);
        assert!((slice[2] - 3.0).abs() < 1e-6);
        
        // Point 2: [2*2, 4*0.5, 6*1] = [4, 2, 6]
        assert!((slice[3] - 4.0).abs() < 1e-6);
        assert!((slice[4] - 2.0).abs() < 1e-6);
        assert!((slice[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_transform_with_center() {
        let device = Default::default();
        let scale = Tensor::<B, 1>::from_floats([2.0, 2.0], &device);
        let center = Tensor::<B, 1>::from_floats([1.0, 1.0], &device);
        
        let transform = ScaleTransform::<B, 2>::new(scale, center);
        
        // Point at center should not move
        let points = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);
        let transformed = transform.transform_points(points);
        let data = transformed.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        
        assert!((slice[0] - 1.0).abs() < 1e-6);
        assert!((slice[1] - 1.0).abs() < 1e-6);
        
        // Point at (2, 2). Relative to center (1, 1) is (1, 1).
        // Scale by 2 -> (2, 2). Add center -> (3, 3).
        let points = Tensor::<B, 2>::from_floats([[2.0, 2.0]], &device);
        let transformed = transform.transform_points(points);
        let data = transformed.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        
        assert!((slice[0] - 3.0).abs() < 1e-6);
        assert!((slice[1] - 3.0).abs() < 1e-6);
    }
}
