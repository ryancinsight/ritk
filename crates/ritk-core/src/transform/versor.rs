//! Versor Rigid transform implementation.
//!
//! This module provides a versor rigid transform (quaternion rotation + translation).
//! It is robust against Gimbal lock and is suitable for 3D registration.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::module::{Module, Param};
use super::trait_::Transform;

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
    fn build_rotation_matrix(&self) -> Tensor<B, 2> {
        // Normalize quaternion to ensure it represents a rotation
        let q = self.rotation.val();
        let norm = q.clone().powf_scalar(2.0).sum().sqrt() + 1e-12; // Avoid div by zero
        let q = q / norm;
        
        let x = q.clone().slice([0..1]);
        let y = q.clone().slice([1..2]);
        let z = q.clone().slice([2..3]);
        let w = q.clone().slice([3..4]);
        
        let xx = x.clone().mul(x.clone());
        let yy = y.clone().mul(y.clone());
        let zz = z.clone().mul(z.clone());
        let xy = x.clone().mul(y.clone());
        let xz = x.clone().mul(z.clone());
        let yz = y.clone().mul(z.clone());
        let xw = x.clone().mul(w.clone());
        let yw = y.clone().mul(w.clone());
        let zw = z.clone().mul(w.clone());
        
        let one = Tensor::<B, 1>::ones([1], &x.device());
        let two = Tensor::<B, 1>::from_floats([2.0], &x.device());
        
        // Row 1
        // 1 - 2(y^2 + z^2)
        let r11 = one.clone() - two.clone() * (yy.clone() + zz.clone());
        // 2(xy - zw)
        let r12 = two.clone() * (xy.clone() - zw.clone());
        // 2(xz + yw)
        let r13 = two.clone() * (xz.clone() + yw.clone());
        
        // Row 2
        // 2(xy + zw)
        let r21 = two.clone() * (xy.clone() + zw.clone());
        // 1 - 2(x^2 + z^2)
        let r22 = one.clone() - two.clone() * (xx.clone() + zz.clone());
        // 2(yz - xw)
        let r23 = two.clone() * (yz.clone() - xw.clone());
        
        // Row 3
        // 2(xz - yw)
        let r31 = two.clone() * (xz.clone() - yw.clone());
        // 2(yz + xw)
        let r32 = two.clone() * (yz.clone() + xw.clone());
        // 1 - 2(x^2 + y^2)
        let r33 = one.clone() - two.clone() * (xx.clone() + yy.clone());
        
        // Construct matrix [3, 3]
        let row1 = Tensor::cat(vec![r11, r12, r13], 0).reshape([1, 3]);
        let row2 = Tensor::cat(vec![r21, r22, r23], 0).reshape([1, 3]);
        let row3 = Tensor::cat(vec![r31, r32, r33], 0).reshape([1, 3]);

        Tensor::cat(vec![row1, row2, row3], 0)
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

        let points = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0]],
            &device,
        );

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
        // q = [sin(45)*1, 0, 0, cos(45)] = [0.7071, 0, 0, 0.7071]
        let rotation = Tensor::<TestBackend, 1>::from_floats([0.70710678, 0.0, 0.0, 0.70710678], &device);
        let center = Tensor::<TestBackend, 1>::zeros([3], &device);
        let transform = VersorRigid3DTransform::<TestBackend>::new(translation, rotation, center);

        // Point (0, 1, 0) should rotate to (0, 0, 1)
        let points = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 1.0, 0.0]],
            &device,
        );

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();
        let vals = data.as_slice::<f32>().unwrap();

        // Check closeness
        assert!((vals[0] - 0.0).abs() < 1e-4);
        assert!((vals[1] - 0.0).abs() < 1e-4);
        assert!((vals[2] - 1.0).abs() < 1e-4);
    }
}
