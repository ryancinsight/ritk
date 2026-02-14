//! Linear interpolation implementation.
//!
//! This module provides linear interpolation for 2D and 3D data.

use burn::tensor::{Tensor, Int};
use burn::tensor::backend::{Backend, AutodiffBackend};
use burn::module::{Module, ModuleVisitor, ModuleMapper, ModuleDisplay, ModuleDisplayDefault, AutodiffModule, Content};
use burn::record::{Record, PrecisionSettings};
use serde::{Serialize, Deserialize};
use super::trait_::Interpolator;

/// Linear Interpolator.
///
/// Performs linear interpolation (bilinear for 2D, trilinear for 3D).
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
    fn interpolate<const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        match D {
            4 => self.interpolate_4d(data, indices),
            3 => self.interpolate_3d(data, indices),
            2 => self.interpolate_2d(data, indices),
            1 => self.interpolate_1d(data, indices),
            _ => panic!("LinearInterpolator only supports 1D, 2D, 3D and 4D tensors"),
        }
    }
}

impl LinearInterpolator {
    fn interpolate_4d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // W (time/4th dim)
        let d1 = shape.dims[1]; // Z
        let d2 = shape.dims[2]; // Y
        let d3 = shape.dims[3]; // X
        let batch_size = indices.dims()[0];
        let device = indices.device();

        // Extract coordinates
        // indices: [Batch, 4] -> (x, y, z, w)
        let x = indices.clone().narrow(1, 0, 1).squeeze::<1>(1);
        let y = indices.clone().narrow(1, 1, 1).squeeze::<1>(1);
        let z = indices.clone().narrow(1, 2, 1).squeeze::<1>(1);
        let w = indices.narrow(1, 3, 1).squeeze::<1>(1);

        // Compute floor coordinates
        let x0 = x.clone().floor();
        let y0 = y.clone().floor();
        let z0 = z.clone().floor();
        let w0 = w.clone().floor();

        // Compute interpolation weights
        let wx = x - x0.clone();
        let wy = y - y0.clone();
        let wz = z - z0.clone();
        let ww = w - w0.clone();

        // Compute x1, y1, z1, w1
        let x1 = x0.clone() + 1.0;
        let y1 = y0.clone() + 1.0;
        let z1 = z0.clone() + 1.0;
        let w1 = w0.clone() + 1.0;

        // Clamp indices to valid range
        let x0_i = x0.clamp(0.0, (d3 - 1) as f64).int();
        let y0_i = y0.clamp(0.0, (d2 - 1) as f64).int();
        let z0_i = z0.clamp(0.0, (d1 - 1) as f64).int();
        let w0_i = w0.clamp(0.0, (d0 - 1) as f64).int();

        let x1_i = x1.clamp(0.0, (d3 - 1) as f64).int();
        let y1_i = y1.clamp(0.0, (d2 - 1) as f64).int();
        let z1_i = z1.clamp(0.0, (d1 - 1) as f64).int();
        let w1_i = w1.clamp(0.0, (d0 - 1) as f64).int();

        // Strides for [W, Z, Y, X] layout (d0, d1, d2, d3)
        let stride_w = (d1 * d2 * d3) as i32;
        let stride_z = (d2 * d3) as i32;
        let stride_y = d3 as i32;

        // Pre-flatten data
        let flat_data = data.clone().reshape([d0 * d1 * d2 * d3]);

        // Gather all 16 values
        let v0000 = Self::gather_4d(&flat_data, &x0_i, &y0_i, &z0_i, &w0_i, stride_y, stride_z, stride_w);
        let v0001 = Self::gather_4d(&flat_data, &x0_i, &y0_i, &z0_i, &w1_i, stride_y, stride_z, stride_w);
        let v0010 = Self::gather_4d(&flat_data, &x0_i, &y0_i, &z1_i, &w0_i, stride_y, stride_z, stride_w);
        let v0011 = Self::gather_4d(&flat_data, &x0_i, &y0_i, &z1_i, &w1_i, stride_y, stride_z, stride_w);
        let v0100 = Self::gather_4d(&flat_data, &x0_i, &y1_i, &z0_i, &w0_i, stride_y, stride_z, stride_w);
        let v0101 = Self::gather_4d(&flat_data, &x0_i, &y1_i, &z0_i, &w1_i, stride_y, stride_z, stride_w);
        let v0110 = Self::gather_4d(&flat_data, &x0_i, &y1_i, &z1_i, &w0_i, stride_y, stride_z, stride_w);
        let v0111 = Self::gather_4d(&flat_data, &x0_i, &y1_i, &z1_i, &w1_i, stride_y, stride_z, stride_w);
        let v1000 = Self::gather_4d(&flat_data, &x1_i, &y0_i, &z0_i, &w0_i, stride_y, stride_z, stride_w);
        let v1001 = Self::gather_4d(&flat_data, &x1_i, &y0_i, &z0_i, &w1_i, stride_y, stride_z, stride_w);
        let v1010 = Self::gather_4d(&flat_data, &x1_i, &y0_i, &z1_i, &w0_i, stride_y, stride_z, stride_w);
        let v1011 = Self::gather_4d(&flat_data, &x1_i, &y0_i, &z1_i, &w1_i, stride_y, stride_z, stride_w);
        let v1100 = Self::gather_4d(&flat_data, &x1_i, &y1_i, &z0_i, &w0_i, stride_y, stride_z, stride_w);
        let v1101 = Self::gather_4d(&flat_data, &x1_i, &y1_i, &z0_i, &w1_i, stride_y, stride_z, stride_w);
        let v1110 = Self::gather_4d(&flat_data, &x1_i, &y1_i, &z1_i, &w0_i, stride_y, stride_z, stride_w);
        let v1111 = Self::gather_4d(&flat_data, &x1_i, &y1_i, &z1_i, &w1_i, stride_y, stride_z, stride_w);

        // Pre-compute (1 - weight)
        let one = Tensor::<B, 1>::ones([batch_size], &device);
        let one_minus_wx = one.clone() - wx.clone();
        let one_minus_wy = one.clone() - wy.clone();
        let one_minus_wz = one.clone() - wz.clone();
        let one_minus_ww = one - ww.clone();

        // Quadrilinear interpolation
        // Interpolate along X
        let c000 = v0000 * one_minus_wx.clone() + v1000 * wx.clone();
        let c001 = v0001 * one_minus_wx.clone() + v1001 * wx.clone();
        let c010 = v0010 * one_minus_wx.clone() + v1010 * wx.clone();
        let c011 = v0011 * one_minus_wx.clone() + v1011 * wx.clone();
        let c100 = v0100 * one_minus_wx.clone() + v1100 * wx.clone();
        let c101 = v0101 * one_minus_wx.clone() + v1101 * wx.clone();
        let c110 = v0110 * one_minus_wx.clone() + v1110 * wx.clone();
        let c111 = v0111 * one_minus_wx.clone() + v1111 * wx.clone();

        // Interpolate along Y
        let c00 = c000 * one_minus_wy.clone() + c100 * wy.clone();
        let c01 = c001 * one_minus_wy.clone() + c101 * wy.clone();
        let c10 = c010 * one_minus_wy.clone() + c110 * wy.clone();
        let c11 = c011 * one_minus_wy.clone() + c111 * wy.clone();

        // Interpolate along Z
        let c0 = c00 * one_minus_wz.clone() + c10 * wz.clone();
        let c1 = c01 * one_minus_wz.clone() + c11 * wz.clone();

        // Interpolate along W
        c0 * one_minus_ww + c1 * ww
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn gather_4d<B: Backend>(
        flat_data: &Tensor<B, 1>,
        xi: &Tensor<B, 1, Int>,
        yi: &Tensor<B, 1, Int>,
        zi: &Tensor<B, 1, Int>,
        wi: &Tensor<B, 1, Int>,
        stride_y: i32,
        stride_z: i32,
        stride_w: i32,
    ) -> Tensor<B, 1> {
        let idx = wi.clone() * stride_w + zi.clone() * stride_z + yi.clone() * stride_y + xi.clone();
        flat_data.clone().gather(0, idx)
    }

    fn interpolate_3d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // Z
        let d1 = shape.dims[1]; // Y
        let d2 = shape.dims[2]; // X
        let batch_size = indices.dims()[0];
        let device = indices.device();

        // Extract coordinates using narrow to avoid unnecessary clones
        // indices: [Batch, 3] -> (x, y, z)
        let x = indices.clone().narrow(1, 0, 1).squeeze::<1>(1);
        let y = indices.clone().narrow(1, 1, 1).squeeze::<1>(1);
        let z = indices.narrow(1, 2, 1).squeeze::<1>(1);

        // Compute floor coordinates
        let x0 = x.clone().floor();
        let y0 = y.clone().floor();
        let z0 = z.clone().floor();

        // Compute interpolation weights
        let wx = x - x0.clone();
        let wy = y - y0.clone();
        let wz = z - z0.clone();

        // Compute x1, y1, z1
        let x1 = x0.clone() + 1.0;
        let y1 = y0.clone() + 1.0;
        let z1 = z0.clone() + 1.0;

        // Clamp indices to valid range
        let x0_i = x0.clamp(0.0, (d2 - 1) as f64).int();
        let y0_i = y0.clamp(0.0, (d1 - 1) as f64).int();
        let z0_i = z0.clamp(0.0, (d0 - 1) as f64).int();

        let x1_i = x1.clamp(0.0, (d2 - 1) as f64).int();
        let y1_i = y1.clamp(0.0, (d1 - 1) as f64).int();
        let z1_i = z1.clamp(0.0, (d0 - 1) as f64).int();

        // Stride for [Z, Y, X] layout (d0, d1, d2)
        let stride_z = (d1 * d2) as i32;
        let stride_y = d2 as i32;

        // Pre-flatten data once to avoid repeated reshaping
        // Pre-flatten data once to avoid repeated reshaping
        let flat_data = data.clone().reshape([d0 * d1 * d2]);

        // Gather all 8 voxel values
        let v000 = Self::gather_3d(&flat_data, &x0_i, &y0_i, &z0_i, stride_y, stride_z);
        let v001 = Self::gather_3d(&flat_data, &x0_i, &y0_i, &z1_i, stride_y, stride_z);
        let v010 = Self::gather_3d(&flat_data, &x0_i, &y1_i, &z0_i, stride_y, stride_z);
        let v011 = Self::gather_3d(&flat_data, &x0_i, &y1_i, &z1_i, stride_y, stride_z);
        let v100 = Self::gather_3d(&flat_data, &x1_i, &y0_i, &z0_i, stride_y, stride_z);
        let v101 = Self::gather_3d(&flat_data, &x1_i, &y0_i, &z1_i, stride_y, stride_z);
        let v110 = Self::gather_3d(&flat_data, &x1_i, &y1_i, &z0_i, stride_y, stride_z);
        let v111 = Self::gather_3d(&flat_data, &x1_i, &y1_i, &z1_i, stride_y, stride_z);

        // Pre-compute (1 - weight) values to reduce operations
        let one = Tensor::<B, 1>::ones([batch_size], &device);
        let one_minus_wx = one.clone() - wx.clone();
        let one_minus_wy = one.clone() - wy.clone();
        let one_minus_wz = one - wz.clone();

        // Trilinear interpolation
        // Interpolate along X
        let c00 = v000 * one_minus_wx.clone() + v100 * wx.clone();
        let c01 = v001 * one_minus_wx.clone() + v101 * wx.clone();
        let c10 = v010 * one_minus_wx.clone() + v110 * wx.clone();
        let c11 = v011 * one_minus_wx + v111 * wx;

        // Interpolate along Y
        let c0 = c00 * one_minus_wy.clone() + c10 * wy.clone();
        let c1 = c01 * one_minus_wy.clone() + c11 * wy.clone();

        // Interpolate along Z
        c0 * one_minus_wz + c1 * wz
    }

    #[inline]
    fn gather_3d<B: Backend>(
        flat_data: &Tensor<B, 1>,
        xi: &Tensor<B, 1, Int>,
        yi: &Tensor<B, 1, Int>,
        zi: &Tensor<B, 1, Int>,
        stride_y: i32,
        stride_z: i32,
    ) -> Tensor<B, 1> {
        let idx = zi.clone() * stride_z + yi.clone() * stride_y + xi.clone();
        flat_data.clone().gather(0, idx)
    }

    fn interpolate_2d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // Y
        let d1 = shape.dims[1]; // X
        let batch_size = indices.dims()[0];
        let device = indices.device();

        // Extract coordinates using narrow
        let x = indices.clone().narrow(1, 0, 1).squeeze::<1>(1);
        let y = indices.narrow(1, 1, 1).squeeze::<1>(1);

        // Compute floor coordinates
        let x0 = x.clone().floor();
        let y0 = y.clone().floor();

        // Compute interpolation weights
        let wx = x - x0.clone();
        let wy = y - y0.clone();

        // Compute x1, y1
        let x1 = x0.clone() + 1.0;
        let y1 = y0.clone() + 1.0;

        // Clamp indices
        let x0_i = x0.clamp(0.0, (d1 - 1) as f64).int();
        let y0_i = y0.clamp(0.0, (d0 - 1) as f64).int();
        let x1_i = x1.clamp(0.0, (d1 - 1) as f64).int();
        let y1_i = y1.clamp(0.0, (d0 - 1) as f64).int();

        // Stride for [Y, X] layout (d0, d1)
        let stride_y = d1 as i32;

        // Pre-flatten data
        let flat_data = data.clone().reshape([d0 * d1]);

        // Gather all 4 voxel values
        let v00 = Self::gather_2d(&flat_data, &x0_i, &y0_i, stride_y);
        let v01 = Self::gather_2d(&flat_data, &x0_i, &y1_i, stride_y);
        let v10 = Self::gather_2d(&flat_data, &x1_i, &y0_i, stride_y);
        let v11 = Self::gather_2d(&flat_data, &x1_i, &y1_i, stride_y);

        // Pre-compute (1 - weight)
        let one = Tensor::<B, 1>::ones([batch_size], &device);
        let one_minus_wx = one.clone() - wx.clone();
        let one_minus_wy = one - wy.clone();

        // Bilinear interpolation
        let c0 = v00 * one_minus_wx.clone() + v10 * wx.clone();
        let c1 = v01 * one_minus_wx + v11 * wx;

        c0 * one_minus_wy + c1 * wy
    }

    #[inline]
    fn gather_2d<B: Backend>(
        flat_data: &Tensor<B, 1>,
        xi: &Tensor<B, 1, Int>,
        yi: &Tensor<B, 1, Int>,
        stride_y: i32,
    ) -> Tensor<B, 1> {
        let idx = yi.clone() * stride_y + xi.clone();
        flat_data.clone().gather(0, idx)
    }

    fn interpolate_1d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // X
        let batch_size = indices.dims()[0];
        let device = indices.device();

        // Extract coordinate
        let x = indices.clone().squeeze::<1>(1);

        // Compute floor coordinate
        let x0 = x.clone().floor();

        // Compute interpolation weight
        let wx = x - x0.clone();

        // Compute x1
        let x1 = x0.clone() + 1.0;

        // Clamp indices
        let x0_i = x0.clamp(0.0, (d0 - 1) as f64).int();
        let x1_i = x1.clamp(0.0, (d0 - 1) as f64).int();

        // Pre-flatten data (identity for 1D but keeps types consistent)
        let flat_data = data.clone().reshape([d0]);

        // Gather 2 values
        let v0 = flat_data.clone().gather(0, x0_i);
        let v1 = flat_data.clone().gather(0, x1_i);

        // Linear interpolation
        let one = Tensor::<B, 1>::ones([batch_size], &device);
        let one_minus_wx = one - wx.clone();

        v0 * one_minus_wx + v1 * wx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn::tensor::TensorData;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_linear_interpolator_3d_axes() {
        let device = Default::default();
        // Shape [Z=2, Y=2, X=2]
        // Flattened:
        // Z=0:
        //   Y=0: X=0(0), X=1(1)
        //   Y=1: X=0(10), X=1(11)
        // Z=1:
        //   Y=0: X=0(100), X=1(101)
        //   Y=1: X=0(110), X=1(111)
        let data_vec = vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
        let data = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2])),
            &device
        );

        let interpolator = LinearInterpolator::new();

        // Test exact grid points
        let indices = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &device
        );
        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        assert_eq!(slice[0], 0.0);  // (0,0,0)
        assert_eq!(slice[1], 1.0);  // (1,0,0)
        assert_eq!(slice[2], 10.0); // (0,1,0)
        assert_eq!(slice[3], 100.0); // (0,0,1)

        // Test interpolation at center (0.5, 0.5, 0.5)
        let center = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
        let result_center = interpolator.interpolate(&data, center);
        let center_data = result_center.into_data();
        let center_slice = center_data.as_slice::<f32>().unwrap();

        // Average of all 8 corners
        let expected = (0.0 + 1.0 + 10.0 + 11.0 + 100.0 + 101.0 + 110.0 + 111.0) / 8.0;
        assert!((center_slice[0] - expected).abs() < 1e-5, "Expected {}, got {}", expected, center_slice[0]);
    }

    #[test]
    fn test_linear_interpolator_2d() {
        let device = Default::default();
        // Shape [Y=2, X=2]
        let data_vec = vec![0.0, 1.0, 10.0, 11.0];
        let data = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2])),
            &device
        );

        let interpolator = LinearInterpolator::new();

        // Test interpolation at center (0.5, 0.5)
        let center = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5]], &device);
        let result = interpolator.interpolate(&data, center);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        // Average of all 4 corners
        let expected = (0.0 + 1.0 + 10.0 + 11.0) / 4.0;
        assert!((slice[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_linear_interpolation_at_grid_points() {
        let device = Default::default();
        let data_vec = vec![0.0, 1.0, 2.0, 3.0];
        let data = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2])),
            &device
        );

        let interpolator = LinearInterpolator::new();

        // Test all 4 grid points
        let indices = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            &device
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
            &device
        );

        let interpolator = LinearInterpolator::new();

        // Test clamping at boundaries
        let indices = Tensor::<TestBackend, 2>::from_floats(
            [[-1.0, -1.0], [5.0, 5.0]], // Outside bounds
            &device
        );
        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        // Should be clamped to valid region
        assert_eq!(slice[0], 0.0); // Clamped to (0,0)
        assert_eq!(slice[1], 3.0); // Clamped to (1,1)
    }

    #[test]
    fn test_linear_interpolator_1d() {
        let device = Default::default();
        // Shape [X=4]
        let data_vec = vec![0.0, 10.0, 20.0, 30.0];
        let data = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([4])),
            &device
        );

        let interpolator = LinearInterpolator::new();

        // Test at x=0.5 -> 5.0
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
        let result = interpolator.interpolate(&data, indices);
        let result_data = result.into_data();
        let slice = result_data.as_slice::<f32>().unwrap();

        assert!((slice[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_interpolator_4d() {
        let device = Default::default();
        // Shape [W=2, Z=2, Y=2, X=2]
        // Flat size 16. Just use indices as values for easy verification?
        // Let's use 0 everywhere except one corner to test indexing.
        let mut data_vec = vec![0.0; 16];
        data_vec[15] = 100.0; // Index (1,1,1,1) is last element.
        // Index is w*8 + z*4 + y*2 + x
        // (1,1,1,1) = 1*8 + 1*4 + 1*2 + 1 = 15. Correct.

        let data = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2, 2])),
            &device
        );

        let interpolator = LinearInterpolator::new();

        // Test at (1,1,1,1)
        let indices = Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_data().as_slice::<f32>().unwrap()[0];
        assert_eq!(val, 100.0);

        // Test at center (0.5, 0.5, 0.5, 0.5)
        // Only v1111 is 100.0, others 0.
        // Weight for v1111 is 0.5*0.5*0.5*0.5 = 1/16.
        // Result should be 100/16 = 6.25
        let center = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5, 0.5]], &device);
        let result_center = interpolator.interpolate(&data, center);
        let val_center = result_center.into_data().as_slice::<f32>().unwrap()[0];

        assert!((val_center - 6.25).abs() < 1e-5);
    }
}
