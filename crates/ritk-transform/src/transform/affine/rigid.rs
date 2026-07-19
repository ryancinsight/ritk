//! Trainable rigid transform.

use coeus_autograd::{
    add as add_variables, broadcast_to as broadcast_variable, cat as cat_variables, cos,
    matmul as matmul_variables, mul as mul_variables, neg as neg_variable, reshape, sin, slice,
    sub as sub_variables, transpose_2d, Var,
};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::{add, broadcast_to, cat, matmul, sub, BackendOps};
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{Resampleable, Transform};
use ritk_image::tensor::Tensor;

/// Rotation and translation about a fixed center.
///
/// Two-dimensional rotations use one angle. Three-dimensional rotations use
/// `(x, y, z)` Euler angles with `R = Rz * Ry * Rx`.
#[derive(Clone)]
pub struct RigidTransform<B: Backend + BackendOps<f32>, const D: usize> {
    translation: Var<f32, B>,
    rotation: Var<f32, B>,
    center: Var<f32, B>,
}

impl<B: Backend + BackendOps<f32> + Default, const D: usize> RigidTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct a trainable rigid transform.
    ///
    /// # Panics
    ///
    /// Panics unless `D` is in `1..=4`, translation and center are `[D]`, and
    /// rotation is `[1]` for 2-D or `[3]` for 3-D.
    #[must_use]
    pub fn new(
        translation: Tensor<f32, B>,
        rotation: Tensor<f32, B>,
        center: Tensor<f32, B>,
    ) -> Self {
        assert!(
            (1..=4).contains(&D),
            "rigid transform supports dimensions 1 through 4"
        );
        assert_eq!(
            translation.shape(),
            [D],
            "rigid translation must have shape [{D}]"
        );
        assert_eq!(center.shape(), [D], "rigid center must have shape [{D}]");
        let angles = if D == 3 { 3 } else { 1 };
        assert_eq!(
            rotation.shape(),
            [angles],
            "rigid rotation must have shape [{angles}] for dimension {D}"
        );
        Self {
            translation: Var::new(translation, true),
            rotation: Var::new(rotation, true),
            center: Var::new(center, false),
        }
    }

    /// Construct the identity transform.
    #[must_use]
    pub fn identity(center: Option<Tensor<f32, B>>, backend: &B) -> Self {
        Self::new(
            Tensor::zeros_on([D], backend),
            Tensor::zeros_on([if D == 3 { 3 } else { 1 }], backend),
            center.unwrap_or_else(|| Tensor::zeros_on([D], backend)),
        )
    }

    /// Clone the current translation.
    #[must_use]
    pub fn translation(&self) -> Tensor<f32, B> {
        self.translation.tensor.clone()
    }

    /// Clone the current Euler-angle parameter.
    #[must_use]
    pub fn rotation(&self) -> Tensor<f32, B> {
        self.rotation.tensor.clone()
    }

    /// Clone the fixed center.
    #[must_use]
    pub fn center(&self) -> Tensor<f32, B> {
        self.center.tensor.clone()
    }

    /// Materialize the homogeneous `[D + 1, D + 1]` matrix.
    #[must_use]
    pub fn matrix(&self) -> Tensor<f32, B> {
        let backend = B::default();
        let rotation = self.build_rotation_matrix();
        let rotated_center =
            matmul(&rotation, &self.center.tensor.reshape([D, 1]), &backend).reshape([D]);
        let translated_center = add(&self.translation.tensor, &self.center.tensor, &backend);
        let offset = sub(&translated_center, &rotated_center, &backend).reshape([D, 1]);
        let top = cat(&[&rotation, &offset], 1);
        let mut bottom = vec![0.0; D + 1];
        bottom[D] = 1.0;
        let bottom = Tensor::from_slice_on([1, D + 1], &bottom, &backend);
        cat(&[&top, &bottom], 0)
    }

    /// Materialize the `[D, D]` rotation matrix.
    #[must_use]
    pub fn build_rotation_matrix(&self) -> Tensor<f32, B> {
        self.rotation_matrix_variables().tensor
    }

    /// Apply the transform while retaining the Coeus autograd graph.
    #[must_use]
    pub fn transform_variables(&self, points: &Var<f32, B>) -> Var<f32, B> {
        let shape = points.tensor.shape();
        assert!(
            shape.len() == 2 && shape[1] == D,
            "rigid transform requires points shaped [N, {D}], got {shape:?}"
        );
        let target = vec![shape[0], D];
        let center = broadcast_variable(&reshape(&self.center, [1, D]), target.clone());
        let translation = broadcast_variable(&reshape(&self.translation, [1, D]), target);
        let centered = sub_variables(points, &center);
        let rotated = matmul_variables(&centered, &transpose_2d(&self.rotation_matrix_variables()));
        add_variables(&add_variables(&rotated, &center), &translation)
    }

    fn rotation_matrix_variables(&self) -> Var<f32, B> {
        if D == 3 {
            let x = slice(&self.rotation, &[(0, 1)]);
            let y = slice(&self.rotation, &[(1, 2)]);
            let z = slice(&self.rotation, &[(2, 3)]);
            let (cx, sx) = (cos(&x), sin(&x));
            let (cy, sy) = (cos(&y), sin(&y));
            let (cz, sz) = (cos(&z), sin(&z));

            let r11 = mul_variables(&cz, &cy);
            let r12 = sub_variables(
                &mul_variables(&mul_variables(&cz, &sy), &sx),
                &mul_variables(&sz, &cx),
            );
            let r13 = add_variables(
                &mul_variables(&mul_variables(&cz, &sy), &cx),
                &mul_variables(&sz, &sx),
            );
            let r21 = mul_variables(&sz, &cy);
            let r22 = add_variables(
                &mul_variables(&mul_variables(&sz, &sy), &sx),
                &mul_variables(&cz, &cx),
            );
            let r23 = sub_variables(
                &mul_variables(&mul_variables(&sz, &sy), &cx),
                &mul_variables(&cz, &sx),
            );
            let r31 = neg_variable(&sy);
            let r32 = mul_variables(&cy, &sx);
            let r33 = mul_variables(&cy, &cx);
            matrix_from_rows(&[[&r11, &r12, &r13], [&r21, &r22, &r23], [&r31, &r32, &r33]])
        } else if D == 2 {
            let angle = slice(&self.rotation, &[(0, 1)]);
            let cosine = cos(&angle);
            let sine = sin(&angle);
            let negative_sine = neg_variable(&sine);
            matrix_from_rows(&[[&cosine, &negative_sine], [&sine, &cosine]])
        } else {
            Var::new(Tensor::eye_on(D, &B::default()), false)
        }
    }
}

pub(super) fn matrix_from_rows<T, B, const D: usize>(rows: &[[&Var<T, B>; D]; D]) -> Var<T, B>
where
    T: coeus_core::Scalar,
    B: Backend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let rows = rows
        .iter()
        .map(|row| {
            let cells = row
                .iter()
                .map(|cell| reshape(cell, [1, 1]))
                .collect::<Vec<_>>();
            let references = cells.iter().collect::<Vec<_>>();
            cat_variables(&references, 1)
        })
        .collect::<Vec<_>>();
    let references = rows.iter().collect::<Vec<_>>();
    cat_variables(&references, 0)
}

impl<B: Backend + BackendOps<f32> + Default, const D: usize> Transform<B, D>
    for RigidTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        let shape = points.shape();
        assert!(
            shape.len() == 2 && shape[1] == D,
            "rigid transform requires points shaped [N, {D}], got {shape:?}"
        );
        let backend = B::default();
        let target = [shape[0], D];
        let center = broadcast_to(&self.center.tensor.reshape([1, D]), &target, &backend);
        let translation = broadcast_to(&self.translation.tensor.reshape([1, D]), &target, &backend);
        let centered = sub(&points, &center, &backend);
        let rotated = matmul(&centered, &self.build_rotation_matrix().t(), &backend);
        add(&add(&rotated, &center, &backend), &translation, &backend)
    }
}

impl<B: Backend + BackendOps<f32>, const D: usize> Resampleable<B, D> for RigidTransform<B, D> {
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

impl<B: Backend + BackendOps<f32> + Default, const D: usize> Module<f32, B> for RigidTransform<B, D>
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

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.transform_variables(input)
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        assert_eq!(
            parameters.len(),
            2,
            "invariant: rigid transform owns translation and rotation parameters"
        );
        self.translation = parameters[0].clone();
        self.rotation = parameters[1].clone();
    }
}

#[cfg(test)]
#[path = "tests_rigid.rs"]
mod tests;
