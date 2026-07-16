//! Chained transform implementation.
//!
//! This module provides a mechanism to chain two transforms together.
//! T(x) = T2(T1(x))

use ritk_core::transform::Transform;
use coeus_core::Backend;
use coeus_tensor::Tensor;
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

impl<B: Backend, T1, T2, const D: usize> ChainedTransform<B, T1, T2, D> {
    /// Create a new chained transform.
    pub fn new(first: T1, second: T2) -> Self {
        Self {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, T1, T2, const D: usize> Transform<B, D> for ChainedTransform<B, T1, T2, D>
where
    T1: Transform<B, D>,
    T2: Transform<B, D>,
{
    fn transform_points(&self, points: Tensor<f32, B>) -> Tensor<f32, B> {
        let intermediate = self.first.transform_points(points);
        self.second.transform_points(intermediate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::affine::translation::TranslationTransform;
    use coeus_core::SequentialBackend;

    type TestBackend = SequentialBackend;

    #[test]
    fn chained_translations_sum_additively() {
        let device = Default::default();

        let t1_tensor = Tensor::<f32, TestBackend>::from_slice_on([2], &[1.0, 0.0], &device);
        let t1 = TranslationTransform::<TestBackend, 2>::new(t1_tensor);

        let t2_tensor = Tensor::<f32, TestBackend>::from_slice_on([2], &[0.0, 1.0], &device);
        let t2 = TranslationTransform::<TestBackend, 2>::new(t2_tensor);

        let chain = ChainedTransform::new(t1, t2);
        let points = Tensor::<f32, TestBackend>::from_slice_on([1, 2], &[0.0, 0.0], &device);
        let transformed = chain.transform_points(points);
        let data = transformed.as_slice();

        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 1.0);
    }
}
