//! Trainable affine transform.

use coeus_autograd::{
    add as add_variables, broadcast_to as broadcast_variable, matmul as matmul_variables, reshape,
    sub as sub_variables, transpose_2d, Var,
};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::{add, broadcast_to, matmul, sub, BackendOps};
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{Resampleable, Transform};
use ritk_image::tensor::Tensor;

/// Trainable affine map `T(x) = A(x - c) + c + t`.
#[derive(Clone)]
pub struct AffineTransform<B: Backend + BackendOps<f32>, const D: usize> {
    matrix: Var<f32, B>,
    translation: Var<f32, B>,
    center: Var<f32, B>,
}

impl<B: Backend + BackendOps<f32>, const D: usize> AffineTransform<B, D> {
    /// Construct a trainable affine map with a fixed center.
    ///
    /// # Panics
    ///
    /// Panics unless the tensors have shapes `[D, D]`, `[D]`, and `[D]`.
    #[must_use]
    pub fn new(
        matrix: Tensor<f32, B>,
        translation: Tensor<f32, B>,
        center: Tensor<f32, B>,
    ) -> Self {
        assert_eq!(
            matrix.shape(),
            [D, D],
            "affine matrix must have shape [{D}, {D}]"
        );
        assert_eq!(
            translation.shape(),
            [D],
            "affine translation must have shape [{D}]"
        );
        assert_eq!(center.shape(), [D], "affine center must have shape [{D}]");
        Self {
            matrix: Var::new(matrix, true),
            translation: Var::new(translation, true),
            center: Var::new(center, false),
        }
    }

    /// Construct the identity map on `backend`.
    #[must_use]
    pub fn identity(center: Option<Tensor<f32, B>>, backend: &B) -> Self {
        let mut values = vec![0.0; D * D];
        for axis in 0..D {
            values[axis * D + axis] = 1.0;
        }
        Self::new(
            Tensor::from_slice_on([D, D], &values, backend),
            Tensor::zeros_on([D], backend),
            center.unwrap_or_else(|| Tensor::zeros_on([D], backend)),
        )
    }

    /// Clone the current linear map.
    #[must_use]
    pub fn matrix(&self) -> Tensor<f32, B> {
        self.matrix.tensor.clone()
    }

    /// Clone the current translation.
    #[must_use]
    pub fn translation(&self) -> Tensor<f32, B> {
        self.translation.tensor.clone()
    }

    /// Clone the fixed center.
    #[must_use]
    pub fn center(&self) -> Tensor<f32, B> {
        self.center.tensor.clone()
    }

    /// Apply the map while retaining the Coeus autograd graph.
    ///
    /// # Panics
    ///
    /// Panics when `points` is not shaped `[N, D]`.
    #[must_use]
    pub fn transform_variables(&self, points: &Var<f32, B>) -> Var<f32, B>
    where
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let shape = points.tensor.shape();
        assert!(
            shape.len() == 2 && shape[1] == D,
            "affine transform requires points shaped [N, {D}], got {shape:?}"
        );
        let target = vec![shape[0], D];
        let center = broadcast_variable(&reshape(&self.center, [1, D]), target.clone());
        let translation = broadcast_variable(&reshape(&self.translation, [1, D]), target);
        let centered = sub_variables(points, &center);
        let linear = matmul_variables(&centered, &transpose_2d(&self.matrix));
        add_variables(&add_variables(&linear, &center), &translation)
    }
}

impl<B: Backend + BackendOps<f32> + Default, const D: usize> Transform<B, D>
    for AffineTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        let shape = points.shape();
        assert!(
            shape.len() == 2 && shape[1] == D,
            "affine transform requires points shaped [N, {D}], got {shape:?}"
        );
        let backend = B::default();
        let target = [shape[0], D];
        let center = broadcast_to(&self.center.tensor.reshape([1, D]), &target, &backend);
        let translation = broadcast_to(&self.translation.tensor.reshape([1, D]), &target, &backend);
        let centered = sub(&points, &center, &backend);
        let linear = matmul(&centered, &self.matrix.tensor.t(), &backend);
        add(&add(&linear, &center, &backend), &translation, &backend)
    }
}

impl<B: Backend + BackendOps<f32>, const D: usize> Resampleable<B, D> for AffineTransform<B, D> {
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

impl<B: Backend + BackendOps<f32> + Default, const D: usize> Module<f32, B>
    for AffineTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        vec![self.matrix.clone(), self.translation.clone()]
    }

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<f32, B>> {
        vec![
            coeus_autograd::Parameter::new(self.matrix.clone(), "matrix"),
            coeus_autograd::Parameter::new(self.translation.clone(), "translation"),
        ]
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.transform_variables(input)
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        assert_eq!(
            parameters.len(),
            2,
            "invariant: affine transform owns matrix and translation parameters"
        );
        self.matrix = parameters[0].clone();
        self.translation = parameters[1].clone();
    }
}

#[cfg(test)]
#[path = "tests_affine.rs"]
mod tests;
