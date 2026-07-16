//! Rigid transform implementation.
//!
//! This module provides a rigid transform (rotation + translation).

use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{Resampleable, Transform};
use coeus_core::CpuAddressableStorage;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;

/// Rigid Transform (Rotation + Translation).
///
/// Supports 2D (1 angle) and 3D (3 Euler angles: ZYX convention).
/// Includes a fixed center of rotation: T(x) = R(x - c) + c + t
#[derive(Clone)]
pub struct RigidTransform<B: Backend, const D: usize> {
    translation: Tensor<f32, B>,
    rotation: Tensor<f32, B>, // [3] for 3D (x, y, z radians), [1] for 2D
    center: Tensor<f32, B>,          // Fixed center of rotation
}

impl<B: Backend, const D: usize> RigidTransform<B, D> {
    /// Create a new rigid transform.
    ///
    /// # Arguments
    /// * `translation` - Tensor of shape `[D]` containing the translation vector
    /// * `rotation` - Tensor of shape `[1]` for 2D (angle in radians) or `[3]` for 3D (Euler angles in radians)
    /// * `center` - Tensor of shape `[D]` containing the fixed center of rotation
    pub fn new(translation: Tensor<f32, B>, rotation: Tensor<f32, B>, center: Tensor<f32, B>) -> Self {
        Self { translation, rotation, center }
    }

    /// Get the translation vector.
    pub fn translation(&self) -> Tensor<f32, B> {
        self.translation.clone()
    }

    /// Get the rotation angles.
    pub fn rotation(&self) -> Tensor<f32, B> {
        self.rotation.clone()
    }

    /// Get the center of rotation.
    pub fn center(&self) -> Tensor<f32, B> {
        self.center.clone()
    }

    /// Get the affine transformation matrix.
    ///
    /// Returns a matrix of shape `[D+1, D+1]` representing the affine transform.
    /// For 3D: [R, t'; 0, 1] where t' = t + c - R@c
    pub fn matrix(&self) -> Tensor<f32, B>
    where
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let device = B::default();
        let r = self.build_rotation_matrix().to_contiguous();
        let t = self.translation.to_contiguous();
        let c = self.center.to_contiguous();
        let r_data = r.as_slice();
        let t_data = t.as_slice();
        let c_data = c.as_slice();
        let mut matrix = vec![0.0f32; (D + 1) * (D + 1)];
        let mut rc = vec![0.0f32; D];

        for row in 0..D {
            for col in 0..D {
                let value = r_data[row * D + col];
                matrix[row * (D + 1) + col] = value;
                rc[row] += value * c_data[col];
            }
        }

        for row in 0..D {
            matrix[row * (D + 1) + D] = t_data[row] + c_data[row] - rc[row];
        }
        matrix[(D + 1) * D + D] = 1.0;

        Tensor::<f32, B>::from_slice_on([D + 1, D + 1], &matrix, &device)
    }

    /// Create an identity rigid transform (no rotation, no translation).
    ///
    /// # Arguments
    /// * `center` - Optional center of rotation. If None, uses origin (0,0...0).
    /// * `device` - Device to create tensors on.
    pub fn identity(center: Option<Tensor<f32, B>>, device: &B) -> Self {
        let translation = Tensor::<f32, B>::zeros_on([D], device);
        let rotation = Tensor::<f32, B>::zeros_on([if D == 3 { 3 } else { 1 }], device);
        let center = center.unwrap_or_else(|| Tensor::<f32, B>::zeros_on([D], device));
        Self::new(translation, rotation, center)
    }

    /// Build the rotation matrix from Euler angles.
    ///
    /// Extracts angles as host scalars, computes the matrix entries on the CPU,
    /// then uploads the result as a single `[D, D]` tensor. This avoids the
    /// ~40 intermediate tensor allocations that the previous tensor-only
    /// formulation required.
    pub fn build_rotation_matrix(&self) -> Tensor<f32, B>
    where
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let dev = B::default();
        let r = self.rotation.to_contiguous();
        let r_data = r.as_slice();

        if D == 3 {
            let alpha = r_data[0];
            let beta = r_data[1];
            let gamma = r_data[2];
            let (sx, cx) = alpha.sin_cos();
            let (sy, cy) = beta.sin_cos();
            let (sz, cz) = gamma.sin_cos();

            let matrix = [
                cz * cy,
                cz * sy * sx - sz * cx,
                cz * sy * cx + sz * sx,
                sz * cy,
                sz * sy * sx + cz * cx,
                sz * sy * cx - cz * sx,
                -sy,
                cy * sx,
                cy * cx,
            ];
            Tensor::<f32, B>::from_slice_on([3, 3], &matrix, &dev)
        } else if D == 2 {
            let theta = r_data[0];
            let (s, c) = theta.sin_cos();
            let matrix = [c, -s, s, c];
            Tensor::<f32, B>::from_slice_on([2, 2], &matrix, &dev)
        } else if D == 1 || D == 4 {
            let mut matrix = vec![0.0f32; D * D];
            for i in 0..D {
                matrix[i * D + i] = 1.0;
            }
            Tensor::<f32, B>::from_slice_on([D, D], &matrix, &dev)
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

impl<B: Backend, const D: usize> Transform<B, D> for RigidTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        let device = B::default();
        let points = points.to_contiguous();
        let rotation = self.build_rotation_matrix().to_contiguous();
        let translation = self.translation.to_contiguous();
        let center = self.center.to_contiguous();
        let batch = points.shape()[0];
        let point_data = points.as_slice();
        let rotation_data = rotation.as_slice();
        let translation_data = translation.as_slice();
        let center_data = center.as_slice();
        let mut output = vec![0.0f32; batch * D];

        for row in 0..batch {
            for out_dim in 0..D {
                let mut acc = center_data[out_dim] + translation_data[out_dim];
                for in_dim in 0..D {
                    let centered = point_data[row * D + in_dim] - center_data[in_dim];
                    acc += centered * rotation_data[out_dim * D + in_dim];
                }
                output[row * D + out_dim] = acc;
            }
        }

        Tensor::<f32, B>::from_slice_on([batch, D], &output, &device)
    }
}

#[cfg(test)]
#[path = "tests_rigid.rs"]
mod tests;
