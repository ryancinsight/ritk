//! Linear interpolation implementation.
//!
//! This module provides linear interpolation for 1D, 2D, 3D, and 4D data.

pub mod dim1;
pub mod dim2;
pub mod dim3;
pub mod dim4;

use super::trait_::Interpolator;
use burn::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::record::{PrecisionSettings, Record};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Linear Interpolator.
///
/// Performs linear interpolation natively restricting bounds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinearInterpolator;

impl<B: Backend> Record<B> for LinearInterpolator {
    type Item<S: PrecisionSettings> = Self;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}

impl<B: Backend> Module<B> for LinearInterpolator {
    type Record = Self;

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {
        // No tensors to visit
    }

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        record
    }

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        devices
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for LinearInterpolator {
    type InnerModule = LinearInterpolator;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl ModuleDisplayDefault for LinearInterpolator {
    fn content(&self, content: Content) -> Option<Content> {
        Some(content.set_top_level_type("LinearInterpolator"))
    }
}

impl ModuleDisplay for LinearInterpolator {}

impl LinearInterpolator {
    /// Create a new linear interpolator.
    pub fn new() -> Self {
        Self
    }
}

impl Default for LinearInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Interpolator<B> for LinearInterpolator {
    fn interpolate<const D: usize>(
        &self,
        data: &Tensor<B, D>,
        indices: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        match D {
            4 => dim4::interpolate_4d(data, indices),
            3 => dim3::interpolate_3d(data, indices),
            2 => dim2::interpolate_2d(data, indices),
            1 => dim1::interpolate_1d(data, indices),
            _ => panic!("LinearInterpolator only supports 1D, 2D, 3D and 4D tensors"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_linear_interpolator_3d_axes() {
        let device = Default::default();
        let data_vec = vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
        let data = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2])),
            &device,
        );

        let interpolator = LinearInterpolator::new();

        let indices = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            &device,
        );
        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[1], 1.0);
        assert_eq!(slice[2], 10.0);
        assert_eq!(slice[3], 100.0);

        let center = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
        let result_center = interpolator.interpolate(&data, center);
        let center_data = result_center.into_data();
        let center_slice = center_data.as_slice::<f32>().unwrap();

        let expected = (0.0 + 1.0 + 10.0 + 11.0 + 100.0 + 101.0 + 110.0 + 111.0) / 8.0;
        assert!(
            (center_slice[0] - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            center_slice[0]
        );
    }

    #[test]
    fn test_linear_interpolator_2d() {
        let device = Default::default();
        let data_vec = vec![0.0, 1.0, 10.0, 11.0];
        let data = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2])),
            &device,
        );

        let interpolator = LinearInterpolator::new();

        let center = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5]], &device);
        let result = interpolator.interpolate(&data, center);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        let expected = (0.0 + 1.0 + 10.0 + 11.0) / 4.0;
        assert!((slice[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_linear_interpolation_at_grid_points() {
        let device = Default::default();
        let data_vec = vec![0.0, 1.0, 2.0, 3.0];
        let data = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2])),
            &device,
        );

        let interpolator = LinearInterpolator::new();

        let indices = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            &device,
        );
        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[1], 1.0);
        assert_eq!(slice[2], 2.0);
        assert_eq!(slice[3], 3.0);
    }

    #[test]
    fn test_linear_interpolator_out_of_bounds() {
        let device = Default::default();
        let data_vec = vec![0.0, 1.0, 2.0, 3.0];
        let data = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2])),
            &device,
        );

        let interpolator = LinearInterpolator::new();

        let indices = Tensor::<TestBackend, 2>::from_floats(
            [[-1.0, -1.0], [5.0, 5.0]],
            &device,
        );
        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[1], 3.0);
    }

    #[test]
    fn test_linear_interpolator_1d() {
        let device = Default::default();
        let data_vec = vec![0.0, 10.0, 20.0, 30.0];
        let data = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([4])),
            &device,
        );

        let interpolator = LinearInterpolator::new();

        let indices = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        assert!((slice[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_interpolator_4d() {
        let device = Default::default();
        let mut data_vec = vec![0.0; 16];
        data_vec[15] = 100.0;

        let data = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2, 2])),
            &device,
        );

        let interpolator = LinearInterpolator::new();

        let indices = Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_data().as_slice::<f32>().unwrap()[0];
        assert_eq!(val, 100.0);

        let center = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5, 0.5]], &device);
        let result_center = interpolator.interpolate(&data, center);
        let val_center = result_center.into_data().as_slice::<f32>().unwrap()[0];

        assert!((val_center - 6.25).abs() < 1e-5);
    }
}
