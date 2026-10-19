//! Trainable three-dimensional quaternion rigid transform.

use super::rigid::matrix_from_rows;
use coeus_autograd::{add, broadcast_to, div, mul, reshape, sqrt, sub, sum, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use ritk_core::transform::Transform;
use ritk_image::tensor::Tensor;

/// Quaternion rotation and translation about a fixed center.
///
/// The quaternion component order is `(x, y, z, w)`.
#[derive(Clone)]
pub struct VersorRigid3DTransform<B: Backend + BackendOps<f32>> {
    translation: Var<f32, B>,
    rotation: Var<f32, B>,
    center: Var<f32, B>,
}

impl<B: Backend + BackendOps<f32> + Default> VersorRigid3DTransform<B>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct a trainable versor transform.
    ///
    /// # Panics
    ///
    /// Panics unless translation and center are `[3]`, rotation is `[4]`, and
    /// the quaternion has a finite nonzero norm.
    #[must_use]
    pub fn new(
        translation: Tensor<f32, B>,
        rotation: Tensor<f32, B>,
        center: Tensor<f32, B>,
    ) -> Self {
        assert_eq!(
            translation.shape(),
            [3],
            "versor translation must have shape [3]"
        );
        assert_eq!(
            rotation.shape(),
            [4],
            "versor quaternion must have shape [4]"
        );
        assert_eq!(center.shape(), [3], "versor center must have shape [3]");
        let squared_norm = rotation
            .to_contiguous()
            .expect("versor transform requires backend evaluation to succeed")
            .as_slice()
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        assert!(
            squared_norm.is_finite() && squared_norm > 0.0,
            "versor quaternion must have a finite nonzero norm"
        );
        Self {
            translation: Var::new(translation, true)
                .expect("versor transform requires backend evaluation to succeed"),
            rotation: Var::new(rotation, true)
                .expect("versor transform requires backend evaluation to succeed"),
            center: Var::new(center, false)
                .expect("versor transform requires backend evaluation to succeed"),
        }
    }

    /// Clone the current translation.
    #[must_use]
    pub fn translation(&self) -> Tensor<f32, B> {
        self.translation.tensor.clone()
    }

    /// Clone the current quaternion.
    #[must_use]
    pub fn rotation(&self) -> Tensor<f32, B> {
        self.rotation.tensor.clone()
    }

    /// Clone the fixed center.
    #[must_use]
    pub fn center(&self) -> Tensor<f32, B> {
        self.center.tensor.clone()
    }

    /// Apply the transform while retaining the Coeus autograd graph.
    #[must_use]
    pub fn transform_variables(&self, points: &Var<f32, B>) -> Var<f32, B> {
        let shape = points.tensor.shape();
        assert!(
            shape.len() == 2 && shape[1] == 3,
            "versor transform requires points shaped [N, 3], got {shape:?}"
        );
        let target = vec![shape[0], 3];
        let center_row = reshape(&self.center, [1, 3])
            .expect("versor transform requires backend evaluation to succeed");
        let translation_row = reshape(&self.translation, [1, 3])
            .expect("versor transform requires backend evaluation to succeed");
        let center = broadcast_to(&center_row, target.clone())
            .expect("versor transform requires backend evaluation to succeed");
        let translation = broadcast_to(&translation_row, target)
            .expect("versor transform requires backend evaluation to succeed");
        let centered =
            sub(points, &center).expect("versor transform requires backend evaluation to succeed");
        let rotation_vars = self.rotation_matrix();
        let rotation_t = coeus_autograd::transpose_2d(&rotation_vars)
            .expect("versor transform requires backend evaluation to succeed");
        let rotated = coeus_autograd::matmul(&centered, &rotation_t)
            .expect("versor transform requires backend evaluation to succeed");
        let centered_output = add(&rotated, &center)
            .expect("versor transform requires backend evaluation to succeed");
        add(&centered_output, &translation)
            .expect("versor transform requires backend evaluation to succeed")
    }

    fn rotation_matrix(&self) -> Var<f32, B> {
        let rotation_sq = mul(&self.rotation, &self.rotation)
            .expect("versor transform requires backend evaluation to succeed");
        let rotation_sum =
            sum(&rotation_sq).expect("versor transform requires backend evaluation to succeed");
        let norm =
            sqrt(&rotation_sum).expect("versor transform requires backend evaluation to succeed");
        let norm_wide = broadcast_to(&norm, vec![4])
            .expect("versor transform requires backend evaluation to succeed");
        let quaternion = div(&self.rotation, &norm_wide)
            .expect("versor transform requires backend evaluation to succeed");
        let component = |index| {
            coeus_autograd::slice(&quaternion, &[(index, index + 1)])
                .expect("versor transform requires backend evaluation to succeed")
        };
        let (x, y, z, w) = (component(0), component(1), component(2), component(3));
        let one = Var::new(
            Tensor::from_slice_on([1], &[1.0], &B::default())
                .expect("versor transform requires backend evaluation to succeed"),
            false,
        )
        .expect("versor transform requires backend evaluation to succeed");
        let two = Var::new(
            Tensor::from_slice_on([1], &[2.0], &B::default())
                .expect("versor transform requires backend evaluation to succeed"),
            false,
        )
        .expect("versor transform requires backend evaluation to succeed");
        let twice = |value: &Var<f32, B>| {
            mul(&two, value).expect("versor transform requires backend evaluation to succeed")
        };

        let xx = mul(&x, &x).expect("versor transform requires backend evaluation to succeed");
        let yy = mul(&y, &y).expect("versor transform requires backend evaluation to succeed");
        let zz = mul(&z, &z).expect("versor transform requires backend evaluation to succeed");
        let xy = mul(&x, &y).expect("versor transform requires backend evaluation to succeed");
        let xz = mul(&x, &z).expect("versor transform requires backend evaluation to succeed");
        let yz = mul(&y, &z).expect("versor transform requires backend evaluation to succeed");
        let xw = mul(&x, &w).expect("versor transform requires backend evaluation to succeed");
        let yw = mul(&y, &w).expect("versor transform requires backend evaluation to succeed");
        let zw = mul(&z, &w).expect("versor transform requires backend evaluation to succeed");

        let r11_inner =
            add(&yy, &zz).expect("versor transform requires backend evaluation to succeed");
        let r11 = sub(&one, &twice(&r11_inner))
            .expect("versor transform requires backend evaluation to succeed");
        let r12_inner =
            sub(&xy, &zw).expect("versor transform requires backend evaluation to succeed");
        let r12 = twice(&r12_inner);
        let r13_inner =
            add(&xz, &yw).expect("versor transform requires backend evaluation to succeed");
        let r13 = twice(&r13_inner);
        let r21_inner =
            add(&xy, &zw).expect("versor transform requires backend evaluation to succeed");
        let r21 = twice(&r21_inner);
        let r22_inner =
            add(&xx, &zz).expect("versor transform requires backend evaluation to succeed");
        let r22 = sub(&one, &twice(&r22_inner))
            .expect("versor transform requires backend evaluation to succeed");
        let r23_inner =
            sub(&yz, &xw).expect("versor transform requires backend evaluation to succeed");
        let r23 = twice(&r23_inner);
        let r31_inner =
            sub(&xz, &yw).expect("versor transform requires backend evaluation to succeed");
        let r31 = twice(&r31_inner);
        let r32_inner =
            add(&yz, &xw).expect("versor transform requires backend evaluation to succeed");
        let r32 = twice(&r32_inner);
        let r33_inner =
            add(&xx, &yy).expect("versor transform requires backend evaluation to succeed");
        let r33 = sub(&one, &twice(&r33_inner))
            .expect("versor transform requires backend evaluation to succeed");

        matrix_from_rows(&[[&r11, &r12, &r13], [&r21, &r22, &r23], [&r31, &r32, &r33]])
    }
}

impl<B: Backend + BackendOps<f32> + Default> Transform<B, 3> for VersorRigid3DTransform<B>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        self.transform_variables(
            &Var::new(points, false)
                .expect("versor transform requires backend evaluation to succeed"),
        )
        .tensor
    }
}

impl<B: Backend + BackendOps<f32> + Default> Module<f32, B> for VersorRigid3DTransform<B>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        vec![self.translation.clone(), self.rotation.clone()]
    }

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<f32, B>> {
        vec![
            coeus_autograd::Parameter::new(self.translation.clone(), "translation"),
            coeus_autograd::Parameter::new(self.rotation.clone(), "rotation"),
        ]
    }

    fn forward(
        &self,
        input: &Var<f32, B>,
    ) -> Result<Var<f32, B>, coeus_nn::ModuleError<<B as coeus_core::ComputeBackend>::Error>> {
        Ok(self.transform_variables(input))
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        assert_eq!(
            parameters.len(),
            2,
            "invariant: versor transform owns translation and rotation parameters"
        );
        self.translation = parameters[0].clone();
        self.rotation = parameters[1].clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn quarter_turn_about_x_maps_y_to_z() {
        let backend = SequentialBackend;
        let half_angle = std::f32::consts::FRAC_1_SQRT_2;
        let transform = VersorRigid3DTransform::<SequentialBackend>::new(
            Tensor::zeros_on([3], &backend).expect("test tensor is valid"),
            Tensor::from_slice_on([4], &[half_angle, 0.0, 0.0, half_angle], &backend)
                .expect("test tensor is valid"),
            Tensor::zeros_on([3], &backend).expect("test tensor is valid"),
        );
        let point = Tensor::from_slice_on([1, 3], &[0.0, 1.0, 0.0], &backend)
            .expect("test tensor is valid");

        let transformed = transform.transform_points(point);

        let values = transformed.as_slice();
        let bound = 8.0 * f32::EPSILON;
        assert!(values[0].abs() <= bound);
        assert!(values[1].abs() <= bound);
        assert!((values[2] - 1.0).abs() <= bound);
    }
}
