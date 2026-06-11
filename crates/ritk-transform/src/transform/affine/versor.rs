//! Versor Rigid transform implementation.
//!
//! This module provides a versor rigid transform (quaternion rotation + translation).
//! It is robust against Gimbal lock and is suitable for 3D registration.

use ritk_core::transform::Transform;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Versor Rigid Transform (Quaternion Rotation + Translation).
///
/// Supports 3D only.
/// Includes a fixed center of rotation: T(x) = R(x - c) + c + t
#[derive(Module, Debug)]
pub struct VersorRigid3DTransform<B: Backend> {
    translation: Param<Tensor<B, 1>>,
    rotation: Param<Tensor<B, 1>>, // [4] Quaternion (x, y, z, w)
    center: Tensor<B, 1>,          // Fixed center of rotation
}

impl<B: Backend> VersorRigid3DTransform<B> {
    /// Create a new versor rigid transform.
    ///
    /// # Arguments
    /// * `translation` - Tensor of shape `[3]` containing the translation vector
    /// * `rotation` - Tensor of shape `[4]` containing the quaternion (x, y, z, w)
    /// * `center` - Tensor of shape `[3]` containing the fixed center of rotation
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

    /// Get the rotation quaternion.
    pub fn rotation(&self) -> Tensor<B, 1> {
        self.rotation.val().clone()
    }

    /// Get the center of rotation.
    pub fn center(&self) -> Tensor<B, 1> {
        self.center.clone()
    }

    /// Build the rotation matrix from Quaternion.
    ///
    /// Extracts the four quaternion components as host scalars, computes all nine
    /// rotation-matrix entries on the CPU, then uploads the result as a single
    /// `[3, 3]` tensor. This avoids the ~28 intermediate tensor allocations
    /// (clones, element-wise products, scalar constants) that the previous
    /// tensor-only formulation required.
    fn build_rotation_matrix(&self) -> Tensor<B, 2> {
        let q = self.rotation.val();
        let dev = q.device();

        // Extract normalised quaternion components as host scalars.
        let norm_data = q.clone().powf_scalar(2.0).sum().sqrt().into_data();
        let norm_val = norm_data
            .as_slice::<f32>()
            .expect("norm tensor must be contiguous f32")[0];
        let norm = norm_val + 1e-12; // Avoid div by zero
        let q_data = (q / norm).into_data();
        let q_slice = q_data
            .as_slice::<f32>()
            .expect("quaternion tensor must be contiguous f32");
        let x = q_slice[0] as f64;
        let y = q_slice[1] as f64;
        let z = q_slice[2] as f64;
        let w = q_slice[3] as f64;

        // Pre-compute products (each used 1–3 times).
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let xw = x * w;
        let yw = y * w;
        let zw = z * w;

        let r11 = 1.0 - 2.0 * (yy + zz);
        let r12 = 2.0 * (xy - zw);
        let r13 = 2.0 * (xz + yw);
        let r21 = 2.0 * (xy + zw);
        let r22 = 1.0 - 2.0 * (xx + zz);
        let r23 = 2.0 * (yz - xw);
        let r31 = 2.0 * (xz - yw);
        let r32 = 2.0 * (yz + xw);
        let r33 = 1.0 - 2.0 * (xx + yy);

        Tensor::<B, 2>::from_floats(
            [
                [r11 as f32, r12 as f32, r13 as f32],
                [r21 as f32, r22 as f32, r23 as f32],
                [r31 as f32, r32 as f32, r33 as f32],
            ],
            &dev,
        )
    }
}

impl<B: Backend> Transform<B, 3> for VersorRigid3DTransform<B> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        // points: [Batch, 3]
        // R: [3, 3]
        // t: [3]
        // c: [3]
        // T(x) = R(x - c) + c + t

        let r = self.build_rotation_matrix();
        let t = self.translation.val().reshape([1, 3]);
        let c = self.center.clone().reshape([1, 3]);

        let centered = points - c.clone();
        let rotated = centered.matmul(r.transpose());

        rotated + c + t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_versor_transform_3d_identity() {
        let device = Default::default();
        let translation = Tensor::<TestBackend, 1>::zeros([3], &device);
        let rotation = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0, 0.0, 1.0], &device); // Identity quaternion (x=0,y=0,z=0,w=1)
        let center = Tensor::<TestBackend, 1>::zeros([3], &device);
        let transform = VersorRigid3DTransform::<TestBackend>::new(translation, rotation, center);

        let points = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device);

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();
        let vals = data.as_slice::<f32>().unwrap();

        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 2.0);
        assert_eq!(vals[2], 3.0);
    }

    #[test]
    fn test_versor_transform_3d_rotation_x_90() {
        let device = Default::default();
        let translation = Tensor::<TestBackend, 1>::zeros([3], &device);
        // Rotate 90 degrees around X axis
        // q = [sin(45)*1, 0, 0, cos(45)] = [1/√2, 0, 0, 1/√2]
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = Tensor::<TestBackend, 1>::from_floats([s, 0.0, 0.0, s], &device);
        let center = Tensor::<TestBackend, 1>::zeros([3], &device);
        let transform = VersorRigid3DTransform::<TestBackend>::new(translation, rotation, center);

        // Point (0, 1, 0) should rotate to (0, 0, 1)
        let points = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0, 0.0]], &device);

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();
        let vals = data.as_slice::<f32>().unwrap();

        // Check closeness
        assert!((vals[0] - 0.0).abs() < 1e-4);
        assert!((vals[1] - 0.0).abs() < 1e-4);
        assert!((vals[2] - 1.0).abs() < 1e-4);
    }
}
