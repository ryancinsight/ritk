//! Rigid transform implementation.
//!
//! This module provides a rigid transform (rotation + translation).

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::module::{Module, Param};
use crate::spatial::{Point, Spacing, Direction};
use super::trait_::{Transform, Resampleable};

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
    fn build_rotation_matrix(&self) -> Tensor<B, 2> {
        let r = self.rotation.val();

        if D == 3 {
            // Euler angles: x (alpha), y (beta), z (gamma)
            // R = R_z(gamma) * R_y(beta) * R_x(alpha)
            let alpha = r.clone().slice([0..1]); // x
            let beta = r.clone().slice([1..2]);  // y
            let gamma = r.clone().slice([2..3]); // z

            let cx = alpha.clone().cos();
            let sx = alpha.sin();
            let cy = beta.clone().cos();
            let sy = beta.sin();
            let cz = gamma.clone().cos();
            let sz = gamma.sin();

            // Row 1
            let r11 = cz.clone().mul(cy.clone());
            let r12 = cz.clone().mul(sy.clone()).mul(sx.clone()).sub(sz.clone().mul(cx.clone()));
            let r13 = cz.clone().mul(sy.clone()).mul(cx.clone()).add(sz.clone().mul(sx.clone()));

            // Row 2
            let r21 = sz.clone().mul(cy.clone());
            let r22 = sz.clone().mul(sy.clone()).mul(sx.clone()).add(cz.clone().mul(cx.clone()));
            let r23 = sz.clone().mul(sy.clone()).mul(cx.clone()).sub(cz.clone().mul(sx.clone()));

            // Row 3
            let r31 = sy.clone().neg();
            let r32 = cy.clone().mul(sx.clone());
            let r33 = cy.clone().mul(cx.clone());

            // Construct matrix [3, 3]
            let row1 = Tensor::cat(vec![r11, r12, r13], 0).reshape([1, 3]);
            let row2 = Tensor::cat(vec![r21, r22, r23], 0).reshape([1, 3]);
            let row3 = Tensor::cat(vec![r31, r32, r33], 0).reshape([1, 3]);

            Tensor::cat(vec![row1, row2, row3], 0)
        } else if D == 2 {
            let theta = r.clone().slice([0..1]);
            let c = theta.clone().cos();
            let s = theta.sin();

            let row1 = Tensor::cat(vec![c.clone(), s.clone().neg()], 0).reshape([1, 2]);
            let row2 = Tensor::cat(vec![s, c], 0).reshape([1, 2]);

            Tensor::cat(vec![row1, row2], 0)
        } else if D == 1 || D == 4 {
            // For 1D and 4D, return identity rotation matrix.
            // 4D rotation optimization is not yet supported.
            Tensor::eye(D, &r.device())
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

        let [n_points, _] = points.dims();
        let r = self.build_rotation_matrix();
        let t = self.translation.val().reshape([1, D]);
        let c = self.center.clone().reshape([1, D]);

        // WGPU has a dispatch limit of 65535. 
        // We chunk the points to avoid hitting this limit for large images.
        const CHUNK_SIZE: usize = 32768; // Safe margin below 65535

        if n_points <= CHUNK_SIZE {
            let centered = points - c.clone();
            let rotated = centered.matmul(r.transpose());
            rotated + c + t
        } else {
            let mut chunks = Vec::new();
            let num_chunks = (n_points + CHUNK_SIZE - 1) / CHUNK_SIZE;
            
            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n_points);
                let chunk_points = points.clone().slice([start..end]);
                
                let centered = chunk_points - c.clone();
                let rotated = centered.matmul(r.clone().transpose());
                let result = rotated + c.clone() + t.clone();
                chunks.push(result);
            }
            
            Tensor::cat(chunks, 0)
        }
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

        let points = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0], [1.0, 1.0]],
            &device,
        );

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

        let points = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            &device,
        );

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
        let rotation = Tensor::<TestBackend, 1>::from_floats([std::f32::consts::FRAC_PI_2], &device); // 90 degrees
        let center = Tensor::<TestBackend, 1>::zeros([2], &device);
        let transform = RigidTransform::<TestBackend, 2>::new(translation, rotation, center);

        // Point (1, 0)
        let points = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 0.0]],
            &device,
        );

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
        let rotation = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0, std::f32::consts::FRAC_PI_2], &device);
        let center = Tensor::<TestBackend, 1>::zeros([3], &device);
        let transform = RigidTransform::<TestBackend, 3>::new(translation, rotation, center);

        // Point (1, 0, 0) should become (0, 1, 0)
        let points = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 0.0, 0.0]],
            &device,
        );

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
