//! Translation transform implementation.
//!
//! This module provides a simple translation transform.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::module::{Module, Param};
use crate::spatial::{Point, Spacing, Direction};
use super::trait_::{Transform, Resampleable};

/// Simple Translation Transform.
///
/// Translates points by a fixed offset vector.
#[derive(Module, Debug)]
pub struct TranslationTransform<B: Backend, const D: usize> {
    translation: Param<Tensor<B, 1>>,
}

impl<B: Backend, const D: usize> TranslationTransform<B, D> {
    /// Create a new translation transform.
    ///
    /// # Arguments
    /// * `translation` - Tensor of shape `[D]` containing the translation vector
    pub fn new(translation: Tensor<B, 1>) -> Self {
        Self {
            translation: Param::from_tensor(translation),
        }
    }

    /// Get the translation vector.
    pub fn translation(&self) -> Tensor<B, 1> {
        self.translation.val().clone()
    }
}

impl<B: Backend, const D: usize> Transform<B, D> for TranslationTransform<B, D> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        // points: [Batch, D]
        // translation: [D]
        // Broadcast translation to [Batch, D]
        let t = self.translation.val().reshape([1, D]);
        points + t
    }
}

impl<B: Backend, const D: usize> Resampleable<B, D> for TranslationTransform<B, D> {
    fn resample(
        &self,
        _shape: [usize; D],
        _origin: Point<D>,
        _spacing: Spacing<D>,
        _direction: Direction<D>,
    ) -> Self {
        // Translation is independent of grid resolution
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_translation_transform() {
        let device = Default::default();
        let translation = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let transform = TranslationTransform::<TestBackend, 3>::new(translation);

        let points = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            &device,
        );

        let transformed = transform.transform_points(points);
        let data = transformed.to_data();

        assert_eq!(data.as_slice::<f32>().unwrap()[0], 1.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[1], 2.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[2], 3.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[3], 2.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[4], 3.0);
        assert_eq!(data.as_slice::<f32>().unwrap()[5], 4.0);
    }
}
