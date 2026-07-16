//! Versor Rigid transform implementation.
//!
//! This module provides a versor rigid transform (quaternion rotation + translation).
//! It is robust against Gimbal lock and is suitable for 3D registration.

use ritk_core::transform::Transform;
use coeus_core::CpuAddressableStorage;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;

/// Versor Rigid Transform (Quaternion Rotation + Translation).
///
/// Supports 3D only.
/// Includes a fixed center of rotation: T(x) = R(x - c) + c + t
#[derive(Clone)]
pub struct VersorRigid3DTransform<B: Backend> {
    translation: Tensor<f32, B>,
    rotation: Tensor<f32, B>, // [4] Quaternion (x, y, z, w)
    center: Tensor<f32, B>,          // Fixed center of rotation
}

impl<B: Backend> VersorRigid3DTransform<B> {
    /// Create a new versor rigid transform.
    ///
    /// # Arguments
    /// * `translation` - Tensor of shape `[3]` containing the translation vector
    /// * `rotation` - Tensor of shape `[4]` containing the quaternion (x, y, z, w)
    /// * `center` - Tensor of shape `[3]` containing the fixed center of rotation
    pub fn new(translation: Tensor<f32, B>, rotation: Tensor<f32, B>, center: Tensor<f32, B>) -> Self {
        Self { translation, rotation, center }
    }

    /// Get the translation vector.
    pub fn translation(&self) -> Tensor<f32, B> {
        self.translation.clone()
    }

    /// Get the rotation quaternion.
    pub fn rotation(&self) -> Tensor<f32, B> {
        self.rotation.clone()
    }

    /// Get the center of rotation.
    pub fn center(&self) -> Tensor<f32, B> {
        self.center.clone()
    }

    /// Build the rotation matrix from Quaternion.
    ///
    /// Extracts the four quaternion components as host scalars, computes all nine
    /// rotation-matrix entries on the CPU, then uploads the result as a single
    /// `[3, 3]` tensor. This avoids the ~28 intermediate tensor allocations
    /// (clones, element-wise products, scalar constants) that the previous
    /// tensor-only formulation required.
    fn build_rotation_matrix(&self) -> Tensor<f32, B>
    where
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let dev = B::default();
        let q = self.rotation.to_contiguous();
        let q_slice = q.as_slice();
        let norm_val = (q_slice[0] * q_slice[0]
            + q_slice[1] * q_slice[1]
            + q_slice[2] * q_slice[2]
            + q_slice[3] * q_slice[3])
            .sqrt();
        // Quaternion norm guard: prevents divide-by-zero during normalization.
        // Practical threshold well above f32 underflow (~1.2e-38).
        const QUAT_NORM_GUARD: f32 = 1e-12;
        let norm = norm_val + QUAT_NORM_GUARD; // Avoid div by zero
        let x = (q_slice[0] / norm) as f64;
        let y = (q_slice[1] / norm) as f64;
        let z = (q_slice[2] / norm) as f64;
        let w = (q_slice[3] / norm) as f64;

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

        let data = [
            r11 as f32, r12 as f32, r13 as f32, r21 as f32, r22 as f32, r23 as f32, r31 as f32,
            r32 as f32, r33 as f32,
        ];
        Tensor::<f32, B>::from_slice_on([3, 3], &data, &dev)
    }
}

impl<B: Backend> Transform<B, 3> for VersorRigid3DTransform<B>
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
        let mut output = vec![0.0f32; batch * 3];

        for row in 0..batch {
            for out_dim in 0..3 {
                let mut acc = center_data[out_dim] + translation_data[out_dim];
                for in_dim in 0..3 {
                    let centered = point_data[row * 3 + in_dim] - center_data[in_dim];
                    acc += centered * rotation_data[out_dim * 3 + in_dim];
                }
                output[row * 3 + out_dim] = acc;
            }
        }

        Tensor::<f32, B>::from_slice_on([batch, 3], &output, &device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    type TestBackend = SequentialBackend;

    #[test]
    fn identity_quaternion_leaves_point_unchanged() {
        let device = Default::default();
        let translation = Tensor::<f32, TestBackend>::zeros([3], &device);
        let rotation = Tensor::<f32, TestBackend>::from_floats([0.0, 0.0, 0.0, 1.0], &device); // Identity quaternion (x=0,y=0,z=0,w=1)
        let center = Tensor::<f32, TestBackend>::zeros([3], &device);
        let transform = VersorRigid3DTransform::<TestBackend>::new(translation, rotation, center);

        let points = Tensor::<f32, TestBackend>::from_floats([[1.0, 2.0, 3.0]], &device);

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();
        let vals = data.as_slice::<f32>().unwrap();

        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 2.0);
        assert_eq!(vals[2], 3.0);
    }

    #[test]
    fn rotation_x_90deg_maps_y_to_z() {
        let device = Default::default();
        let translation = Tensor::<f32, TestBackend>::zeros([3], &device);
        // Rotate 90 degrees around X axis
        // q = [sin(45)*1, 0, 0, cos(45)] = [1/√2, 0, 0, 1/√2]
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let rotation = Tensor::<f32, TestBackend>::from_floats([s, 0.0, 0.0, s], &device);
        let center = Tensor::<f32, TestBackend>::zeros([3], &device);
        let transform = VersorRigid3DTransform::<TestBackend>::new(translation, rotation, center);

        // Point (0, 1, 0) should rotate to (0, 0, 1)
        let points = Tensor::<f32, TestBackend>::from_floats([[0.0, 1.0, 0.0]], &device);

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();
        let vals = data.as_slice::<f32>().unwrap();

        // Check closeness
        const QUATERNION_ROTATION_TOL: f32 = 1e-4;
        assert!((vals[0] - 0.0).abs() < QUATERNION_ROTATION_TOL);
        assert!((vals[1] - 0.0).abs() < QUATERNION_ROTATION_TOL);
        assert!((vals[2] - 1.0).abs() < QUATERNION_ROTATION_TOL);
    }
}
