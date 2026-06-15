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
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    const NEAR_ZERO: f32 = 1e-6;

    #[test]
    fn test_affine_transform_identity() {
        let device = Default::default();
        let transform = AffineTransform::<TestBackend, 3>::identity(None, &device);

        let points =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

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

        assert!((slice[0] - 3.0).abs() < NEAR_ZERO);
        assert!((slice[1] - 1.0).abs() < NEAR_ZERO);
    }

    #[test]
    #[should_panic(expected = "expects a [D, D] linear matrix")]
    fn new_rejects_homogeneous_matrix() {
        // Seeding a 3-D affine with a [D+1, D+1] = [4, 4] homogeneous matrix
        // (the shape `RigidTransform::matrix()` returns) must fail loudly at
        // construction, not with a cryptic backend matmul panic later.
        let device = Default::default();
        let matrix = Tensor::<TestBackend, 2>::eye(4, &device);
        let translation = Tensor::<TestBackend, 1>::zeros([3], &device);
        let center = Tensor::<TestBackend, 1>::zeros([3], &device);
        let _ = AffineTransform::<TestBackend, 3>::new(matrix, translation, center);
    }

    #[test]
    fn affine_seeded_from_rigid_rotation_reproduces_rigid() {
        use crate::transform::affine::rigid::RigidTransform;
        let device = Default::default();
        // A rigid transform with a non-trivial rotation/translation/center.
        let rigid = RigidTransform::<TestBackend, 3>::new(
            Tensor::from_floats([2.0, -1.0, 3.0], &device), // translation
            Tensor::from_floats([0.3, -0.2, 0.1], &device), // Euler angles
            Tensor::from_floats([5.0, 6.0, 7.0], &device),  // center
        );
        // Correct seeding: A = R (the [D, D] rotation), not the homogeneous matrix.
        let affine = AffineTransform::<TestBackend, 3>::new(
            rigid.build_rotation_matrix(),
            rigid.translation(),
            rigid.center(),
        );
        let pts =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [10.0, -4.0, 0.5]], &device);
        let r = rigid.transform_points(pts.clone()).to_data();
        let a = affine.transform_points(pts).to_data();
        let r = r.as_slice::<f32>().unwrap();
        let a = a.as_slice::<f32>().unwrap();
        for (ri, ai) in r.iter().zip(a.iter()) {
            assert!((ri - ai).abs() < 1e-5, "affine {ai} != rigid {ri}");
        }
    }
}
