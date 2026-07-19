//! Trainable axis-aligned scale transform.

use coeus_autograd::{
    add as add_variables, broadcast_to as broadcast_variable, mul as mul_variables, reshape,
    sub as sub_variables, Var,
};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::{add, broadcast_to, mul, sub, BackendOps};
use ritk_core::transform::Transform;
use ritk_image::tensor::Tensor;

/// Scale about a fixed center: `T(x) = scale * (x - center) + center`.
#[derive(Clone)]
pub struct ScaleTransform<B: Backend + BackendOps<f32>, const D: usize> {
    scale: Var<f32, B>,
    center: Var<f32, B>,
}

impl<B: Backend + BackendOps<f32>, const D: usize> ScaleTransform<B, D> {
    /// Construct a trainable scale parameter and fixed center.
    ///
    /// # Panics
    ///
    /// Panics unless both tensors have shape `[D]`.
    #[must_use]
    pub fn new(scale: Tensor<f32, B>, center: Tensor<f32, B>) -> Self {
        assert_eq!(scale.shape(), [D], "scale parameter must have shape [{D}]");
        assert_eq!(center.shape(), [D], "scale center must have shape [{D}]");
        Self {
            scale: Var::new(scale, true),
            center: Var::new(center, false),
        }
    }

    /// Construct the identity scale.
    #[must_use]
    pub fn identity(center: Option<Tensor<f32, B>>, backend: &B) -> Self {
        Self::new(
            Tensor::ones_on([D], backend),
            center.unwrap_or_else(|| Tensor::zeros_on([D], backend)),
        )
    }

    /// Clone the current scale factors.
    #[must_use]
    pub fn scale(&self) -> Tensor<f32, B> {
        self.scale.tensor.clone()
    }

    /// Clone the fixed center.
    #[must_use]
    pub fn center(&self) -> Tensor<f32, B> {
        self.center.tensor.clone()
    }

    /// Apply the scale while retaining the Coeus autograd graph.
    #[must_use]
    pub fn transform_variables(&self, points: &Var<f32, B>) -> Var<f32, B>
    where
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let shape = points.tensor.shape();
        assert!(
            shape.len() == 2 && shape[1] == D,
            "scale transform requires points shaped [N, {D}], got {shape:?}"
        );
        let target = vec![shape[0], D];
        let center = broadcast_variable(&reshape(&self.center, [1, D]), target.clone());
        let scale = broadcast_variable(&reshape(&self.scale, [1, D]), target);
        add_variables(
            &mul_variables(&sub_variables(points, &center), &scale),
            &center,
        )
    }
}

impl<B: Backend + BackendOps<f32> + Default, const D: usize> Transform<B, D>
    for ScaleTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        let shape = points.shape();
        assert!(
            shape.len() == 2 && shape[1] == D,
            "scale transform requires points shaped [N, {D}], got {shape:?}"
        );
        let backend = B::default();
        let target = [shape[0], D];
        let center = broadcast_to(&self.center.tensor.reshape([1, D]), &target, &backend);
        let scale = broadcast_to(&self.scale.tensor.reshape([1, D]), &target, &backend);
        add(
            &mul(&sub(&points, &center, &backend), &scale, &backend),
            &center,
            &backend,
        )
    }
}

impl<B: Backend + BackendOps<f32> + Default, const D: usize> Module<f32, B> for ScaleTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        vec![self.scale.clone()]
    }

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<f32, B>> {
        vec![coeus_autograd::Parameter::new(self.scale.clone(), "scale")]
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.transform_variables(input)
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        assert_eq!(
            parameters.len(),
            1,
            "invariant: scale transform owns one parameter"
        );
        self.scale = parameters[0].clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn scales_about_fixed_center() {
        let backend = SequentialBackend;
        let transform = ScaleTransform::<SequentialBackend, 2>::new(
            Tensor::from_slice_on([2], &[2.0, 2.0], &backend),
            Tensor::from_slice_on([2], &[1.0, 1.0], &backend),
        );
        let points = Tensor::from_slice_on([2, 2], &[1.0, 1.0, 2.0, 2.0], &backend);

        let transformed = transform.transform_points(points);

        assert_eq!(transformed.as_slice(), &[1.0, 1.0, 3.0, 3.0]);
    }
}
