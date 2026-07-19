//! Ordered composition of two transforms.

use coeus_autograd::{Parameter, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use ritk_core::transform::Transform;
use ritk_image::tensor::Tensor;
use std::marker::PhantomData;

/// Composition `second(first(points))`.
#[derive(Clone)]
pub struct ChainedTransform<B: Backend, T1, T2, const D: usize> {
    /// First transform applied.
    pub first: T1,
    /// Second transform applied.
    pub second: T2,
    backend: PhantomData<fn() -> B>,
}

impl<B: Backend, T1, T2, const D: usize> ChainedTransform<B, T1, T2, D> {
    /// Construct an ordered composition.
    #[must_use]
    pub const fn new(first: T1, second: T2) -> Self {
        Self {
            first,
            second,
            backend: PhantomData,
        }
    }
}

impl<B, T1, T2, const D: usize> Transform<B, D> for ChainedTransform<B, T1, T2, D>
where
    B: Backend,
    T1: Transform<B, D>,
    T2: Transform<B, D>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        self.second
            .transform_points(self.first.transform_points(points))
    }
}

impl<B, T1, T2, const D: usize> Module<f32, B> for ChainedTransform<B, T1, T2, D>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    T1: Module<f32, B>,
    T2: Module<f32, B>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        self.first
            .parameters()
            .into_iter()
            .chain(self.second.parameters())
            .collect()
    }

    fn named_parameters(&self) -> Vec<Parameter<f32, B>> {
        self.first
            .named_parameters()
            .into_iter()
            .map(|parameter| parameter.with_prefix("first"))
            .chain(
                self.second
                    .named_parameters()
                    .into_iter()
                    .map(|parameter| parameter.with_prefix("second")),
            )
            .collect()
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.second.forward(&self.first.forward(input))
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let first_count = self.first.parameters().len();
        assert_eq!(
            parameters.len(),
            first_count + self.second.parameters().len(),
            "invariant: chained parameter inventory matches both child modules"
        );
        let (first, second) = parameters.split_at(first_count);
        self.first.load_parameters(first);
        self.second.load_parameters(second);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::affine::translation::TranslationTransform;
    use coeus_core::SequentialBackend;

    #[test]
    fn chained_translations_sum_additively() {
        let backend = SequentialBackend;
        let first = TranslationTransform::<SequentialBackend, 2>::new(Tensor::from_slice_on(
            [2],
            &[1.0, 0.0],
            &backend,
        ));
        let second = TranslationTransform::<SequentialBackend, 2>::new(Tensor::from_slice_on(
            [2],
            &[0.0, 1.0],
            &backend,
        ));
        let chain = ChainedTransform::new(first, second);
        let points = Tensor::from_slice_on([1, 2], &[0.0, 0.0], &backend);

        let transformed = chain.transform_points(points);

        assert_eq!(transformed.as_slice(), &[1.0, 1.0]);
    }
}
