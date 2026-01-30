//! Chained transform implementation.
//!
//! This module provides a mechanism to chain two transforms together.
//! T(x) = T2(T1(x))

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::module::Module;
use super::trait_::Transform;
use std::marker::PhantomData;

/// Chained Transform (T2 after T1).
///
/// Applies two transforms in sequence:
/// y = T2(T1(x))
///
/// This allows composing rigid, affine, and b-spline transforms.
#[derive(Module, Debug)]
pub struct ChainedTransform<B: Backend, T1, T2, const D: usize> {
    pub first: T1,
    pub second: T2,
    pub _phantom: PhantomData<B>,
}

impl<B: Backend, T1, T2, const D: usize> ChainedTransform<B, T1, T2, D> {
    /// Create a new chained transform.
    ///
    /// # Arguments
    /// * `first` - The first transform to apply
    /// * `second` - The second transform to apply
    pub fn new(first: T1, second: T2) -> Self {
        Self { first, second, _phantom: PhantomData }
    }
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
    use burn_ndarray::NdArray;
    use crate::transform::translation::TranslationTransform;
    use burn::tensor::TensorData;

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
