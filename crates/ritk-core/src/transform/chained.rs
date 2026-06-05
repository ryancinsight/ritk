//! Chained transform implementation.
//!
//! This module provides a mechanism to chain two transforms together.
//! T(x) = T2(T1(x))

use super::trait_::Transform;
use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault};
use burn::record::{PrecisionSettings, Record};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Chained Transform (T2 after T1).
///
/// Applies two transforms in sequence:
/// y = T2(T1(x))
///
/// This allows composing rigid, affine, and b-spline transforms.
#[derive(Clone, Debug)]
pub struct ChainedTransform<B: Backend, T1, T2, const D: usize> {
    pub first: T1,
    pub second: T2,
    pub _phantom: PhantomData<fn() -> B>,
}

/// Record for [`ChainedTransform`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainedTransformRecord<T1R, T2R> {
    first: T1R,
    second: T2R,
}

impl<B: Backend, T1R, T2R> Record<B> for ChainedTransformRecord<T1R, T2R>
where
    T1R: Record<B> + Send,
    T2R: Record<B> + Send,
{
    type Item<S: PrecisionSettings> = ChainedTransformRecord<T1R::Item<S>, T2R::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        ChainedTransformRecord {
            first: self.first.into_item(),
            second: self.second.into_item(),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        ChainedTransformRecord {
            first: Record::<B>::from_item(item.first, device),
            second: Record::<B>::from_item(item.second, device),
        }
    }
}

impl<B: Backend, T1, T2, const D: usize> ChainedTransform<B, T1, T2, D> {
    /// Create a new chained transform.
    ///
    /// # Arguments
    /// * `first` - The first transform to apply
    /// * `second` - The second transform to apply
    pub fn new(first: T1, second: T2) -> Self {
        Self {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, T1: Module<B>, T2: Module<B>, const D: usize> Module<B>
    for ChainedTransform<B, T1, T2, D>
{
    type Record = ChainedTransformRecord<T1::Record, T2::Record>;

    fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.first.visit(visitor);
        self.second.visit(visitor);
    }

    fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        Self {
            first: self.first.map(mapper),
            second: self.second.map(mapper),
            _phantom: PhantomData,
        }
    }

    fn into_record(self) -> Self::Record {
        ChainedTransformRecord {
            first: self.first.into_record(),
            second: self.second.into_record(),
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            first: self.first.load_record(record.first),
            second: self.second.load_record(record.second),
            _phantom: PhantomData,
        }
    }

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        let devices = self.first.collect_devices(devices);
        self.second.collect_devices(devices)
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            first: self.first.to_device(device),
            second: self.second.to_device(device),
            _phantom: PhantomData,
        }
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            first: self.first.fork(device),
            second: self.second.fork(device),
            _phantom: PhantomData,
        }
    }
}

impl<B: AutodiffBackend, T1, T2, const D: usize> AutodiffModule<B>
    for ChainedTransform<B, T1, T2, D>
where
    T1: AutodiffModule<B>,
    T2: AutodiffModule<B>,
{
    type InnerModule = ChainedTransform<B::InnerBackend, T1::InnerModule, T2::InnerModule, D>;

    fn valid(&self) -> Self::InnerModule {
        ChainedTransform {
            first: self.first.valid(),
            second: self.second.valid(),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, T1: ModuleDisplay, T2: ModuleDisplay, const D: usize> ModuleDisplayDefault
    for ChainedTransform<B, T1, T2, D>
{
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add_single(&self.first)
            .add_single(&self.second)
            .optional()
    }
}

impl<B: Backend, T1: ModuleDisplay, T2: ModuleDisplay, const D: usize> ModuleDisplay
    for ChainedTransform<B, T1, T2, D>
{
}

impl<B: Backend, T1, T2, const D: usize> Transform<B, D> for ChainedTransform<B, T1, T2, D>
where
    T1: Transform<B, D> + Module<B>,
    T2: Transform<B, D> + Module<B>,
{
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let intermediate = self.first.transform_points(points);
        self.second.transform_points(intermediate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::translation::TranslationTransform;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_chained_transform_2d() {
        let device = Default::default();

        // T1: Translate by [1, 0]
        let t1_data = TensorData::from([1.0, 0.0]);
        let t1_tensor = Tensor::<TestBackend, 1>::from_data(t1_data, &device);
        let t1 = TranslationTransform::<TestBackend, 2>::new(t1_tensor);

        // T2: Translate by [0, 1]
        let t2_data = TensorData::from([0.0, 1.0]);
        let t2_tensor = Tensor::<TestBackend, 1>::from_data(t2_data, &device);
        let t2 = TranslationTransform::<TestBackend, 2>::new(t2_tensor);

        // Chain: T2(T1(x))
        let chain = ChainedTransform::new(t1, t2);

        // Point: [0, 0]
        // T1 -> [1, 0]
        // T2 -> [1, 1]
        let points = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0]], &device);
        let transformed = chain.transform_points(points);

        let data = transformed.into_data();
        let slice = data.as_slice::<f32>().unwrap();

        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1], 1.0);
    }
}
