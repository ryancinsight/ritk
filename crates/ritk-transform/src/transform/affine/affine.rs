//! Affine transform implementation.
//!
//! This module provides an affine transform (linear transformation + translation).

use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{Resampleable, Transform};
use ritk_wgpu_compat::apply_row_chunks;

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
    matrix: Param<Tensor<B, 2>>,      // [D, D] linear transformation matrix
    translation: Param<Tensor<B, 1>>, // [D] translation vector
    center: Tensor<B, 1>,             // [D] fixed center
}

impl<B: Backend, const D: usize> AffineTransform<B, D> {
    /// Create a new affine transform.
    ///
    /// # Arguments
    /// * `matrix` - Tensor of shape `[D, D]` containing the linear transformation matrix
    /// * `translation` - Tensor of shape `[D]` containing the translation vector
    /// * `center` - Tensor of shape `[D]` containing the fixed center
    pub fn new(matrix: Tensor<B, 2>, translation: Tensor<B, 1>, center: Tensor<B, 1>) -> Self {
        // The linear part must be [D, D]. A common mistake is to seed an affine
        // from `RigidTransform::matrix()`, which returns the [D+1, D+1]
        // homogeneous form `[R, t'; 0, 1]`; `transform_points` would then
        // mis-multiply `[N, D] @ [D+1, D+1]` and panic deep inside the backend.
        // Use `RigidTransform::build_rotation_matrix()` (the [D, D] rotation) to
        // seed instead. Fail here, loudly and actionably.
        let [rows, cols] = matrix.dims();
        assert!(
            rows == D && cols == D,
            "AffineTransform::new expects a [D, D] linear matrix (D = {D}), got [{rows}, {cols}]; \
             if seeding from a RigidTransform use `build_rotation_matrix()` (the [D, D] rotation), \
             not `matrix()` (the [D+1, D+1] homogeneous form)"
        );
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
        let a_t = a.transpose();

        apply_row_chunks(points, ritk_wgpu_compat::WGPU_CHUNK_SIZE, |chunk_points| {
            let centered = chunk_points - c.clone();
            let rotated = centered.matmul(a_t.clone());
            rotated + c.clone() + t.clone()
        })
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
