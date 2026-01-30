//! Affine transform implementation.
//!
//! This module provides an affine transform (linear transformation + translation).

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::module::{Module, Param};
use super::trait_::Transform;

/// Affine Transform (Linear transformation + Translation).
///
/// Represents a general affine transformation with a fixed center:
/// T(x) = A(x - c) + c + t
///
/// where:
/// * A is a D×D matrix (linear transformation: rotation, scale, shear)
/// * t is a D-dimensional translation vector
/// * c is a D-dimensional fixed center of rotation/scaling
#[derive(Module, Debug)]
pub struct AffineTransform<B: Backend, const D: usize> {
    matrix: Param<Tensor<B, 2>>, // [D, D] linear transformation matrix
    translation: Param<Tensor<B, 1>>, // [D] translation vector
    center: Tensor<B, 1>, // [D] fixed center
}

impl<B: Backend, const D: usize> AffineTransform<B, D> {
    /// Create a new affine transform.
    ///
    /// # Arguments
    /// * `matrix` - Tensor of shape `[D, D]` containing the linear transformation matrix
    /// * `translation` - Tensor of shape `[D]` containing the translation vector
    /// * `center` - Tensor of shape `[D]` containing the fixed center
    pub fn new(matrix: Tensor<B, 2>, translation: Tensor<B, 1>, center: Tensor<B, 1>) -> Self {
        Self {
            matrix: Param::from_tensor(matrix),
            translation: Param::from_tensor(translation),
            center,
        }
    }

    /// Create an identity affine transform.
    ///
    /// # Arguments
    /// * `center` - Optional center of rotation. If None, uses origin (0,0...0).
    /// * `device` - Device to create tensors on.
    pub fn identity(center: Option<Tensor<B, 1>>, device: &B::Device) -> Self {
        let mut matrix_data = vec![0.0f32; D * D];
        for i in 0..D {
            matrix_data[i * (D + 1)] = 1.0;
        }
        let data = burn::tensor::TensorData::from(matrix_data.as_slice());
        let matrix = Tensor::<B, 1>::from_data(data, device).reshape([D, D]);
        
        let translation = Tensor::<B, 1>::zeros([D], device);
        let center = center.unwrap_or_else(|| Tensor::<B, 1>::zeros([D], device));
        
        Self::new(matrix, translation, center)
    }

    /// Get the transformation matrix.
    pub fn matrix(&self) -> Tensor<B, 2> {
        self.matrix.val()
    }

    /// Get the translation vector.
    pub fn translation(&self) -> Tensor<B, 1> {
        self.translation.val()
    }

    /// Get the center of rotation.
    pub fn center(&self) -> Tensor<B, 1> {
        self.center.clone()
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for AffineTransform<B, D> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        // points: [Batch, D]
        // matrix (A): [D, D]
        // translation (t): [D]
        // center (c): [D]
        //
        // T(x) = A(x - c) + c + t
        //
        // In row vector notation (standard for Burn/PyTorch inputs [N, D]):
        // y = (x - c) @ A^T + c + t

        let c = self.center.clone().reshape([1, D]);
        let t = self.translation.val().reshape([1, D]);
        let a = self.matrix.val();

        let centered = points - c.clone();
        
        // Matmul: [N, D] x [D, D]^T -> [N, D] x [D, D] (if transposed)
        // Correct is: (x-c) * A^T
        let rotated = centered.matmul(a.transpose());
        
        rotated + c + t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_affine_transform_identity() {
        let device = Default::default();
        let transform = AffineTransform::<TestBackend, 3>::identity(None, &device);

        let points = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &device,
        );

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();

        // Identity transform should not change points
        let slice = data.as_slice::<f32>().unwrap();
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1], 2.0);
        assert_eq!(slice[2], 3.0);
        assert_eq!(slice[3], 4.0);
        assert_eq!(slice[4], 5.0);
        assert_eq!(slice[5], 6.0);
    }

    #[test]
    fn test_affine_transform_translation_with_center() {
        let device = Default::default();
        
        // Matrix: Identity
        let matrix = Tensor::<TestBackend, 2>::eye(2, &device);
        
        // Translation: [1.0, 1.0]
        let translation = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);
        
        // Center: [10.0, 10.0]
        let center = Tensor::<TestBackend, 1>::from_floats([10.0, 10.0], &device);
        
        let transform = AffineTransform::<TestBackend, 2>::new(matrix, translation, center);
        
        // Point at center: [10, 10]
        // T(c) = A(c-c) + c + t = 0 + c + t = c + t
        // Expected: [11, 11]
        let points = Tensor::<TestBackend, 2>::from_floats([[10.0, 10.0]], &device);
        
        let transformed = transform.transform_points(points);
        let data = transformed.to_data();
        let slice = data.as_slice::<f32>().unwrap();
        
        assert_eq!(slice[0], 11.0);
        assert_eq!(slice[1], 11.0);
    }
    
    #[test]
    fn test_affine_transform_scale_with_center() {
        let device = Default::default();
        
        // Matrix: Scale by 2.0
        // [2, 0]
        // [0, 2]
        let matrix = Tensor::<TestBackend, 2>::eye(2, &device) * 2.0;
        
        // Translation: 0
        let translation = Tensor::<TestBackend, 1>::zeros([2], &device);
        
        // Center: [1.0, 1.0]
        let center = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);
        
        let transform = AffineTransform::<TestBackend, 2>::new(matrix, translation, center);
        
        // Point: [2.0, 1.0] (1 unit right of center)
        // T(x) = A(x - c) + c
        // x - c = [1, 0]
        // A(x-c) = [2, 0]
        // + c = [3, 1]
        let points = Tensor::<TestBackend, 2>::from_floats([[2.0, 1.0]], &device);
        
        let transformed = transform.transform_points(points);
        let data = transformed.to_data();
        let slice = data.as_slice::<f32>().unwrap();
        
        assert!((slice[0] - 3.0).abs() < 1e-6);
        assert!((slice[1] - 1.0).abs() < 1e-6);
    }
}
