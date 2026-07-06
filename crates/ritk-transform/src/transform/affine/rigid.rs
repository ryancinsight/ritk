//! Rigid transform implementation.
//!
//! This module provides a rigid transform (rotation + translation).

use ritk_image::burn::module::{Module, Param};
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{Resampleable, Transform};
use ritk_wgpu_compat::apply_row_chunks;

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
            // Clippy single_range_in_vec_init: single-element range array is
            // intentional — Burn `slice` requires an array of ranges.
            #[allow(clippy::single_range_in_vec_init)]
            let alpha = r.clone().slice([0..1]);
            #[allow(clippy::single_range_in_vec_init)]
            let beta = r.clone().slice([1..2]);
            #[allow(clippy::single_range_in_vec_init)]
            let gamma = r.clone().slice([2..3]);

            let cx = alpha.clone().cos();
            let sx = alpha.sin();
            let cy = beta.clone().cos();
            let sy = beta.sin();
            let cz = gamma.clone().cos();
            let sz = gamma.sin();

            // Row 1: R_z * R_y * R_x
            let r11 = cz.clone() * cy.clone();
            let r12 = cz.clone() * sy.clone() * sx.clone() - sz.clone() * cx.clone();
            let r13 = cz.clone() * sy.clone() * cx.clone() + sz.clone() * sx.clone();
            // Row 2
            let r21 = sz.clone() * cy.clone();
            let r22 = sz.clone() * sy.clone() * sx.clone() + cz.clone() * cx.clone();
            let r23 = sz.clone() * sy.clone() * cx.clone() - cz.clone() * sx.clone();
            // Row 3
            let r31 = -sy.clone();
            let r32 = cy.clone() * sx.clone();
            let r33 = cy.clone() * cx.clone();

            let row1 = Tensor::cat(
                vec![
                    r11.reshape([1, 1]),
                    r12.reshape([1, 1]),
                    r13.reshape([1, 1]),
                ],
                1,
            );
            let row2 = Tensor::cat(
                vec![
                    r21.reshape([1, 1]),
                    r22.reshape([1, 1]),
                    r23.reshape([1, 1]),
                ],
                1,
            );
            let row3 = Tensor::cat(
                vec![
                    r31.reshape([1, 1]),
                    r32.reshape([1, 1]),
                    r33.reshape([1, 1]),
                ],
                1,
            );

            Tensor::cat(vec![row1, row2, row3], 0)
        } else if D == 2 {
            #[allow(clippy::single_range_in_vec_init)]
            let theta = r.clone().slice([0..1]);
            let c = theta.clone().cos();
            let s = theta.sin();

            let row1 = Tensor::cat(
                vec![c.clone().reshape([1, 1]), (-s.clone()).reshape([1, 1])],
                1,
            );
            let row2 = Tensor::cat(vec![s.reshape([1, 1]), c.reshape([1, 1])], 1);

            Tensor::cat(vec![row1, row2], 0)
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

        apply_row_chunks(points, ritk_wgpu_compat::WGPU_CHUNK_SIZE, |chunk_points| {
            let centered = chunk_points - c.clone();
            let rotated = centered.matmul(r.clone().transpose());
            rotated + c.clone() + t.clone()
        })
    }
}

#[cfg(test)]
#[path = "tests_rigid.rs"]
mod tests;
