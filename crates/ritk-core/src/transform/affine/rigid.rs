//! Rigid transform implementation.
//!
//! This module provides a rigid transform (rotation + translation).

use super::super::trait_::{Resampleable, Transform};
use crate::spatial::{Direction, Point, Spacing};
use crate::wgpu_compat::apply_row_chunks;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Rigid Transform (Rotation + Translation).
///
/// Supports 2D (1 angle) and 3D (3 Euler angles: ZYX convention).
/// Includes a fixed center of rotation: T(x) = R(x - c) + c + t
#[derive(Module, Debug)]
pub struct RigidTransform<B: Backend, const D: usize> {
    translation: Param<Tensor<B, 1>>,
    rotation: Param<Tensor<B, 1>>, // [3] for 3D (x, y, z radians), [1] for 2D
    center: Tensor<B, 1>,          // Fixed center of rotation
}

impl<B: Backend, const D: usize> RigidTransform<B, D> {
    /// Create a new rigid transform.
    ///
    /// # Arguments
    /// * `translation` - Tensor of shape `[D]` containing the translation vector
    /// * `rotation` - Tensor of shape `[1]` for 2D (angle in radians) or `[3]` for 3D (Euler angles in radians)
    /// * `center` - Tensor of shape `[D]` containing the fixed center of rotation
    pub fn new(translation: Tensor<B, 1>, rotation: Tensor<B, 1>, center: Tensor<B, 1>) -> Self {
        Self {
            translation: Param::from_tensor(translation),
            rotation: Param::from_tensor(rotation),
            center,
        }
    }

    /// Get the translation vector.
    pub fn translation(&self) -> Tensor<B, 1> {
        self.translation.val().clone()
    }

    /// Get the rotation angles.
    pub fn rotation(&self) -> Tensor<B, 1> {
        self.rotation.val().clone()
    }

    /// Get the center of rotation.
    pub fn center(&self) -> Tensor<B, 1> {
        self.center.clone()
    }

    /// Get the affine transformation matrix.
    ///
    /// Returns a matrix of shape `[D+1, D+1]` representing the affine transform.
    /// For 3D: [R, t'; 0, 1] where t' = t + c - R@c
    pub fn matrix(&self) -> Tensor<B, 2> {
        let r = self.build_rotation_matrix();
        let t = self.translation.val();
        let c = self.center.clone();
        let rc = r.clone().matmul(c.clone().reshape([D, 1])).squeeze(); // R @ c
        let t_prime = t + c - rc; // t + c - R@c

        let mut matrix = Tensor::<B, 2>::zeros([D + 1, D + 1], &self.center.device());
        // Set rotation part
        matrix = matrix.slice_assign([0..D, 0..D], r);
        // Set translation part
        matrix = matrix.slice_assign([0..D, D..D + 1], t_prime.reshape([D, 1]));
        // Set bottom right to 1
        matrix = matrix.slice_assign(
            [D..D + 1, D..D + 1],
            Tensor::<B, 2>::ones([1, 1], &self.center.device()),
        );
        matrix
    }

    /// Create an identity rigid transform (no rotation, no translation).
    ///
    /// # Arguments
    /// * `center` - Optional center of rotation. If None, uses origin (0,0...0).
    /// * `device` - Device to create tensors on.
    pub fn identity(center: Option<Tensor<B, 1>>, device: &B::Device) -> Self {
        let translation = Tensor::<B, 1>::zeros([D], device);
        let rotation = Tensor::<B, 1>::zeros([if D == 3 { 3 } else { 1 }], device);
        let center = center.unwrap_or_else(|| Tensor::<B, 1>::zeros([D], device));
        Self::new(translation, rotation, center)
    }

    /// Build the rotation matrix from Euler angles.
    ///
    /// Extracts angles as host scalars, computes the matrix entries on the CPU,
    /// then uploads the result as a single `[D, D]` tensor. This avoids the
    /// ~40 intermediate tensor allocations that the previous tensor-only
    /// formulation required.
    pub fn build_rotation_matrix(&self) -> Tensor<B, 2> {
        let r = self.rotation.val();
        let dev = r.device();

        if D == 3 {
            // Euler angles: x (alpha), y (beta), z (gamma)
            // R = R_z(gamma) * R_y(beta) * R_x(alpha)
            let r_data = r.into_data();
            let r_slice = r_data
                .as_slice::<f32>()
                .expect("rotation tensor must be contiguous f32");
            let alpha = r_slice[0] as f64;
            let beta = r_slice[1] as f64;
            let gamma = r_slice[2] as f64;

            let (cx, sx) = (alpha.cos(), alpha.sin());
            let (cy, sy) = (beta.cos(), beta.sin());
            let (cz, sz) = (gamma.cos(), gamma.sin());

            // Row 1: R_z * R_y * R_x
            let r11 = cz * cy;
            let r12 = cz * sy * sx - sz * cx;
            let r13 = cz * sy * cx + sz * sx;
            // Row 2
            let r21 = sz * cy;
            let r22 = sz * sy * sx + cz * cx;
            let r23 = sz * sy * cx - cz * sx;
            // Row 3
            let r31 = -sy;
            let r32 = cy * sx;
            let r33 = cy * cx;

            Tensor::<B, 2>::from_floats(
                [
                    [r11 as f32, r12 as f32, r13 as f32],
                    [r21 as f32, r22 as f32, r23 as f32],
                    [r31 as f32, r32 as f32, r33 as f32],
                ],
                &dev,
            )
        } else if D == 2 {
            let r_data = r.into_data();
            let r_slice = r_data
                .as_slice::<f32>()
                .expect("rotation tensor must be contiguous f32");
            let theta = r_slice[0] as f64;
            let c = theta.cos() as f32;
            let s = theta.sin() as f32;

            Tensor::<B, 2>::from_floats([[c, -s], [s, c]], &dev)
        } else if D == 1 || D == 4 {
            // For 1D and 4D, return identity rotation matrix.
            // 4D rotation optimization is not yet supported.
            Tensor::eye(D, &dev)
        } else {
            panic!("RigidTransform only supports 1D, 2D, 3D, and 4D");
        }
    }
}

impl<B: Backend, const D: usize> Resampleable<B, D> for RigidTransform<B, D> {
    fn resample(
        &self,
        _shape: [usize; D],
        _origin: Point<D>,
        _spacing: Spacing<D>,
        _direction: Direction<D>,
    ) -> Self {
        // Rigid transform is independent of grid resolution
        self.clone()
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for RigidTransform<B, D> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        // points: [Batch, D]
        // R: [D, D]
        // t: [D]
        // c: [D]
        // T(x) = R(x - c) + c + t
        //
        // In row vector notation:
        // y = (x - c) @ R^T + c + t

        let r = self.build_rotation_matrix();
        let t = self.translation.val().reshape([1, D]);
        let c = self.center.clone().reshape([1, D]);

        apply_row_chunks(
            points,
            crate::wgpu_compat::WGPU_CHUNK_SIZE,
            |chunk_points| {
                let centered = chunk_points - c.clone();
                let rotated = centered.matmul(r.clone().transpose());
                rotated + c.clone() + t.clone()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rigid_transform_2d() {
        let device = Default::default();
        let translation = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0], &device);
        let rotation = Tensor::<TestBackend, 1>::from_floats([0.0], &device); // No rotation
        let center = Tensor::<TestBackend, 1>::zeros([2], &device);
        let transform = RigidTransform::<TestBackend, 2>::new(translation, rotation, center);

        let points = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0], [1.0, 1.0]], &device);

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();

        // With no rotation, just translation
        assert_eq!(data.as_slice::<f32>().unwrap()[0], 1.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[1], 2.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[2], 2.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[3], 3.0);
    }

    #[test]
    fn test_rigid_transform_3d() {
        let device = Default::default();
        let translation = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let rotation = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0, 0.0], &device); // No rotation
        let center = Tensor::<TestBackend, 1>::zeros([3], &device);
        let transform = RigidTransform::<TestBackend, 3>::new(translation, rotation, center);

        let points =
            Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], &device);

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();

        // With no rotation, just translation
        assert_eq!(data.as_slice::<f32>().unwrap()[0], 1.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[1], 2.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[2], 3.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[3], 2.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[4], 3.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[5], 4.0);
    }

    #[test]
    fn test_rigid_transform_2d_rotation() {
        let device = Default::default();
        let translation = Tensor::<TestBackend, 1>::zeros([2], &device);
        let rotation =
            Tensor::<TestBackend, 1>::from_floats([std::f32::consts::FRAC_PI_2], &device); // 90 degrees
        let center = Tensor::<TestBackend, 1>::zeros([2], &device);
        let transform = RigidTransform::<TestBackend, 2>::new(translation, rotation, center);

        // Point (1, 0)
        let points = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0]], &device);

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();
        let slice = data.as_slice::<f32>().unwrap();

        // Should be (0, 1) (approximately)
        assert!((slice[0] - 0.0).abs() < 1e-6);
        assert!((slice[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rigid_transform_3d_rotation_z() {
        let device = Default::default();
        let translation = Tensor::<TestBackend, 1>::zeros([3], &device);
        // Rotate 90 deg around Z. Euler: x, y, z. So [0, 0, PI/2]
        let rotation =
            Tensor::<TestBackend, 1>::from_floats([0.0, 0.0, std::f32::consts::FRAC_PI_2], &device);
        let center = Tensor::<TestBackend, 1>::zeros([3], &device);
        let transform = RigidTransform::<TestBackend, 3>::new(translation, rotation, center);

        // Point (1, 0, 0) should become (0, 1, 0)
        let points = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0, 0.0]], &device);

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();
        let slice = data.as_slice::<f32>().unwrap();

        assert!((slice[0] - 0.0).abs() < 1e-6);
        assert!((slice[1] - 1.0).abs() < 1e-6);
        assert!((slice[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rigid_transform_1d() {
        let device = Default::default();
        let translation = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let rotation = Tensor::<TestBackend, 1>::zeros([0], &device);
        let center = Tensor::<TestBackend, 1>::zeros([1], &device);
        let transform = RigidTransform::<TestBackend, 1>::new(translation, rotation, center);

        let points = Tensor::<TestBackend, 2>::from_floats([[1.0]], &device);
        let transformed = transform.transform_points(points);
        let val = transformed.into_data().as_slice::<f32>().unwrap()[0];
        assert_eq!(val, 2.0);
    }

    #[test]
    fn test_rigid_transform_4d() {
        let device = Default::default();
        let translation = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0, 1.0, 1.0], &device);
        let rotation = Tensor::<TestBackend, 1>::zeros([6], &device); // Usually 6 for 4D but we ignore it
        let center = Tensor::<TestBackend, 1>::zeros([4], &device);
        let transform = RigidTransform::<TestBackend, 4>::new(translation, rotation, center);

        let points = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0, 0.0]], &device);
        let transformed = transform.transform_points(points);
        let result = transformed.into_data().as_slice::<f32>().unwrap().to_vec();

        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 1.0);
    }
}
