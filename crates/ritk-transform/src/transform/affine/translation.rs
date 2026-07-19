//! Trainable translation transform.

use coeus_autograd::{add as add_variables, broadcast_to as broadcast_variable, reshape, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::{add, broadcast_to, BackendOps};
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{Resampleable, Transform};
use ritk_image::tensor::Tensor;

/// Translation by one trainable `[D]` offset vector.
#[derive(Clone)]
pub struct TranslationTransform<B: Backend + BackendOps<f32>, const D: usize> {
    translation: Var<f32, B>,
}

impl<B: Backend + BackendOps<f32>, const D: usize> TranslationTransform<B, D> {
    /// Construct a trainable translation parameter.
    ///
    /// # Panics
    ///
    /// Panics when `translation` is not shaped `[D]`.
    #[must_use]
    pub fn new(translation: Tensor<f32, B>) -> Self {
        assert_eq!(
            translation.shape(),
            [D],
            "translation parameter must have shape [{D}]"
        );
        Self {
            translation: Var::new(translation, true),
        }
    }

    /// Clone the current translation tensor.
    #[must_use]
    pub fn translation(&self) -> Tensor<f32, B> {
        self.translation.tensor.clone()
    }

    /// Apply the translation while retaining the Coeus autograd graph.
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
            "translation transform requires points shaped [N, {D}], got {shape:?}"
        );
        let translation = reshape(&self.translation, [1, D]);
        add_variables(points, &broadcast_variable(&translation, vec![shape[0], D]))
    }
}

impl<B: Backend + BackendOps<f32> + Default, const D: usize> Transform<B, D>
    for TranslationTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        let shape = points.shape();
        assert!(
            shape.len() == 2 && shape[1] == D,
            "translation transform requires points shaped [N, {D}], got {shape:?}"
        );
        let backend = B::default();
        let translation = self.translation.tensor.reshape([1, D]);
        let translated = broadcast_to(&translation, &[shape[0], D], &backend);
        add(&points, &translated, &backend)
    }
}

impl<B: Backend + BackendOps<f32>, const D: usize> Resampleable<B, D>
    for TranslationTransform<B, D>
{
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
    for TranslationTransform<B, D>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        vec![self.translation.clone()]
    }

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<f32, B>> {
        vec![coeus_autograd::Parameter::new(
            self.translation.clone(),
            "translation",
        )]
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.transform_variables(input)
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        assert_eq!(
            parameters.len(),
            1,
            "invariant: translation transform owns one parameter"
        );
        self.translation = parameters[0].clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    #[test]
    fn translates_every_point() {
        let backend = SequentialBackend;
        let translation = Tensor::<f32, _>::from_slice_on([3], &[1.0, 2.0, 3.0], &backend);
        let transform = TranslationTransform::<SequentialBackend, 3>::new(translation);
        let points =
            Tensor::<f32, _>::from_slice_on([2, 3], &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &backend);

        let transformed = transform.transform_points(points);

        assert_eq!(transformed.as_slice(), &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
    }
}
