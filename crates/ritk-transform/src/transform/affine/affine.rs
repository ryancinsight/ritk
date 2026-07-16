//! Affine transform implementation.
//!
//! This module provides an affine transform (linear transformation + translation).

use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{Resampleable, Transform};
use coeus_core::CpuAddressableStorage;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;

/// Affine Transform (Linear transformation + Translation).
///
/// Represents a general affine transformation with a fixed center:
/// T(x) = A(x - c) + c + t
///
/// where:
/// * A is a D×D matrix (linear transformation: rotation, scale, shear)
/// * t is a D-dimensional translation vector
/// * c is a D-dimensional fixed center of rotation/scaling
#[derive(Clone)]
pub struct AffineTransform<B: Backend, const D: usize> {
    matrix: Tensor<f32, B>,      // [D, D] linear transformation matrix
    translation: Tensor<f32, B>, // [D] translation vector
    center: Tensor<f32, B>,             // [D] fixed center
}

impl<B: Backend, const D: usize> AffineTransform<B, D> {
    /// Create a new affine transform.
    ///
    /// # Arguments
    /// * `matrix` - Tensor of shape `[D, D]` containing the linear transformation matrix
    /// * `translation` - Tensor of shape `[D]` containing the translation vector
    /// * `center` - Tensor of shape `[D]` containing the fixed center
    pub fn new(matrix: Tensor<f32, B>, translation: Tensor<f32, B>, center: Tensor<f32, B>) -> Self {
        // The linear part must be [D, D]. A common mistake is to seed an affine
        // from `RigidTransform::matrix()`, which returns the [D+1, D+1]
        // homogeneous form `[R, t'; 0, 1]`; `transform_points` would then
        // mis-multiply `[N, D] @ [D+1, D+1]` and panic deep inside the backend.
        // Use `RigidTransform::build_rotation_matrix()` (the [D, D] rotation) to
        // seed instead. Fail here, loudly and actionably.
        let &[rows, cols] = matrix.shape() else {
            panic!("AffineTransform::new expects a 2-D matrix");
        };
        assert!(
            rows == D && cols == D,
            "AffineTransform::new expects a [D, D] linear matrix (D = {D}), got [{rows}, {cols}]; \
             if seeding from a RigidTransform use `build_rotation_matrix()` (the [D, D] rotation), \
             not `matrix()` (the [D+1, D+1] homogeneous form)"
        );
        Self {
            matrix,
            translation,
            center,
        }
    }

    /// Create an identity affine transform.
    ///
    /// # Arguments
    /// * `center` - Optional center of rotation. If None, uses origin (0,0...0).
    /// * `device` - Device to create tensors on.
    pub fn identity(center: Option<Tensor<f32, B>>, device: &B) -> Self {
        let mut matrix_data = vec![0.0f32; D * D];
        for i in 0..D {
            matrix_data[i * (D + 1)] = 1.0;
        }
        let matrix = Tensor::<f32, B>::from_slice_on([D, D], &matrix_data, device);
        let translation = Tensor::<f32, B>::zeros_on([D], device);
        let center = center.unwrap_or_else(|| Tensor::<f32, B>::zeros_on([D], device));

        Self::new(matrix, translation, center)
    }

    /// Get the transformation matrix.
    pub fn matrix(&self) -> Tensor<f32, B> {
        self.matrix.clone()
    }

    /// Get the translation vector.
    pub fn translation(&self) -> Tensor<f32, B> {
        self.translation.clone()
    }

    /// Get the center of rotation.
    pub fn center(&self) -> Tensor<f32, B> {
        self.center.clone()
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for AffineTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        let device = B::default();
        let points = points.to_contiguous();
        let matrix = self.matrix.to_contiguous();
        let translation = self.translation.to_contiguous();
        let center = self.center.to_contiguous();
        let batch = points.shape()[0];
        let point_data = points.as_slice();
        let matrix_data = matrix.as_slice();
        let translation_data = translation.as_slice();
        let center_data = center.as_slice();
        let mut output = vec![0.0f32; batch * D];

        for row in 0..batch {
            for out_dim in 0..D {
                let mut acc = center_data[out_dim] + translation_data[out_dim];
                for in_dim in 0..D {
                    let centered = point_data[row * D + in_dim] - center_data[in_dim];
                    acc += centered * matrix_data[out_dim * D + in_dim];
                }
                output[row * D + out_dim] = acc;
            }
        }

        Tensor::<f32, B>::from_slice_on([batch, D], &output, &device)
    }
}

impl<B: Backend, const D: usize> Resampleable<B, D> for AffineTransform<B, D> {
    fn resample(
        &self,
        _shape: [usize; D],
        _origin: Point<D>,
        _spacing: Spacing<D>,
        _direction: Direction<D>,
    ) -> Self {
        self.clone()
    }
}

#[cfg(test)]
#[path = "tests_affine.rs"]
mod tests;
