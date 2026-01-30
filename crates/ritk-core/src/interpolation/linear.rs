//! Linear interpolation implementation.
//!
//! This module provides linear interpolation for 2D and 3D data.

use burn::tensor::{Tensor, Int};
use burn::tensor::backend::Backend;
use super::trait_::Interpolator;

/// Linear Interpolator.
///
/// Performs linear interpolation (bilinear for 2D, trilinear for 3D).
#[derive(Debug, Clone, Copy)]
pub struct LinearInterpolator;

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
        if D == 3 {
            self.interpolate_3d(data, indices)
        } else if D == 2 {
            self.interpolate_2d(data, indices)
        } else {
            panic!("LinearInterpolator only supports 2D and 3D tensors");
        }
    }
}

impl LinearInterpolator {
    fn interpolate_3d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // Z
        let d1 = shape.dims[1]; // Y
        let d2 = shape.dims[2]; // X

        // indices: [Batch, 3] -> (x, y, z)
        let x = indices.clone().slice([0..indices.dims()[0], 0..1]).squeeze::<1>(1);
        let y = indices.clone().slice([0..indices.dims()[0], 1..2]).squeeze::<1>(1);
        let z = indices.clone().slice([0..indices.dims()[0], 2..3]).squeeze::<1>(1);

        let x0 = x.clone().floor();
        let y0 = y.clone().floor();
        let z0 = z.clone().floor();

        let x1 = x0.clone() + 1.0;
        let y1 = y0.clone() + 1.0;
        let z1 = z0.clone() + 1.0;

        let wx = x.clone() - x0.clone();
        let wy = y.clone() - y0.clone();
        let wz = z.clone() - z0.clone();

        // Clamp indices to valid range
        // x corresponds to d2 (X)
        // y corresponds to d1 (Y)
        // z corresponds to d0 (Z)
        
        let x0_i = x0.clamp(0.0, (d2 - 1) as f64).int();
        let y0_i = y0.clamp(0.0, (d1 - 1) as f64).int();
        let z0_i = z0.clamp(0.0, (d0 - 1) as f64).int();

        let x1_i = x1.clamp(0.0, (d2 - 1) as f64).int();
        let y1_i = y1.clamp(0.0, (d1 - 1) as f64).int();
        let z1_i = z1.clamp(0.0, (d0 - 1) as f64).int();

        // Stride for [Z, Y, X] layout (d0, d1, d2)
        // Flat index = z * (d1 * d2) + y * d2 + x
        let stride_z = (d1 * d2) as i32;
        let stride_y = d2 as i32;
        let stride_x = 1;

        let get_val = |xi: Tensor<B, 1, Int>, yi: Tensor<B, 1, Int>, zi: Tensor<B, 1, Int>| -> Tensor<B, 1> {
            // Note: zi is passed as first arg in previous implementation but here we are explicit
            let idx = zi * stride_z + yi * stride_y + xi * stride_x;
            let flat_data = data.clone().reshape([d0 * d1 * d2]);
            flat_data.gather(0, idx)
        };

        // Interpolation weights match the coordinate system
        // Trilinear interpolation:
        // C00 = V000(1-x) + V100x  (along X)
        // C01 = V001(1-x) + V101x
        // ...
        // Here we use x, y, z relative to the cube cell.
        // Let's stick to the previous implementation structure but use correct indices.
        // Vxyz where x,y,z are 0 or 1 offset.
        
        let v000 = get_val(x0_i.clone(), y0_i.clone(), z0_i.clone());
        let v001 = get_val(x0_i.clone(), y0_i.clone(), z1_i.clone());
        let v010 = get_val(x0_i.clone(), y1_i.clone(), z0_i.clone());
        let v011 = get_val(x0_i.clone(), y1_i.clone(), z1_i.clone());
        let v100 = get_val(x1_i.clone(), y0_i.clone(), z0_i.clone());
        let v101 = get_val(x1_i.clone(), y0_i.clone(), z1_i.clone());
        let v110 = get_val(x1_i.clone(), y1_i.clone(), z0_i.clone());
        let v111 = get_val(x1_i.clone(), y1_i.clone(), z1_i.clone());

        // Interpolate along X first
        let c00 = v000 * (wx.clone().neg().add_scalar(1.0)) + v100 * wx.clone(); // (y0, z0)
        let c01 = v001 * (wx.clone().neg().add_scalar(1.0)) + v101 * wx.clone(); // (y0, z1)
        let c10 = v010 * (wx.clone().neg().add_scalar(1.0)) + v110 * wx.clone(); // (y1, z0)
        let c11 = v011 * (wx.clone().neg().add_scalar(1.0)) + v111 * wx.clone(); // (y1, z1)

        // Interpolate along Y
        let c0 = c00 * (wy.clone().neg().add_scalar(1.0)) + c10 * wy.clone(); // (z0)
        let c1 = c01 * (wy.clone().neg().add_scalar(1.0)) + c11 * wy.clone(); // (z1)

        // Interpolate along Z
        c0 * (wz.clone().neg().add_scalar(1.0)) + c1 * wz.clone()
    }

    fn interpolate_2d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // Y
        let d1 = shape.dims[1]; // X

        // indices: [Batch, 2] -> (x, y)
        let x = indices.clone().slice([0..indices.dims()[0], 0..1]).squeeze::<1>(1);
        let y = indices.clone().slice([0..indices.dims()[0], 1..2]).squeeze::<1>(1);

        let x0 = x.clone().floor();
        let y0 = y.clone().floor();

        let x1 = x0.clone() + 1.0;
        let y1 = y0.clone() + 1.0;

        let wx = x.clone() - x0.clone();
        let wy = y.clone() - y0.clone();

        // Clamp indices
        // x corresponds to d1 (X)
        // y corresponds to d0 (Y)
        
        let x0_i = x0.clamp(0.0, (d1 - 1) as f64).int();
        let y0_i = y0.clamp(0.0, (d0 - 1) as f64).int();

        let x1_i = x1.clamp(0.0, (d1 - 1) as f64).int();
        let y1_i = y1.clamp(0.0, (d0 - 1) as f64).int();

        // Stride for [Y, X] layout (d0, d1)
        // Flat index = y * d1 + x
        let stride_y = d1 as i32;
        let stride_x = 1;

        let get_val = |xi: Tensor<B, 1, Int>, yi: Tensor<B, 1, Int>| -> Tensor<B, 1> {
            let idx = yi * stride_y + xi * stride_x;
            let flat_data = data.clone().reshape([d0 * d1]);
            flat_data.gather(0, idx)
        };

        let v00 = get_val(x0_i.clone(), y0_i.clone());
        let v01 = get_val(x0_i.clone(), y1_i.clone());
        let v10 = get_val(x1_i.clone(), y0_i.clone());
        let v11 = get_val(x1_i.clone(), y1_i.clone());

        // Interpolate along X
        let c0 = v00 * (wx.clone().neg().add_scalar(1.0)) + v10 * wx.clone(); // (y0)
        let c1 = v01 * (wx.clone().neg().add_scalar(1.0)) + v11 * wx.clone(); // (y1)

        // Interpolate along Y
        c0 * (wy.clone().neg().add_scalar(1.0)) + c1 * wy.clone()
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
        
        // Test X axis interpolation (0.5, 0, 0) -> Should be avg of 0 and 1 = 0.5
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.0, 0.0]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar() as f32;
        assert!((val - 0.5).abs() < 1e-5, "X axis interpolation failed, got {}", val);
        
        // Test Y axis interpolation (0, 0.5, 0) -> Should be avg of 0 and 10 = 5.0
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.5, 0.0]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar() as f32;
        assert!((val - 5.0).abs() < 1e-5, "Y axis interpolation failed, got {}", val);
        
        // Test Z axis interpolation (0, 0, 0.5) -> Should be avg of 0 and 100 = 50.0
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.5]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar() as f32;
        assert!((val - 50.0).abs() < 1e-5, "Z axis interpolation failed, got {}", val);
        
        // Test trilinear (0.5, 0.5, 0.5)
        // Avg of all corners: (0+1+10+11+100+101+110+111)/8 = 444/8 = 55.5
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar() as f32;
        assert!((val - 55.5).abs() < 1e-5, "Trilinear interpolation failed, got {}", val);
    }
    
    #[test]
    fn test_linear_interpolator_2d_axes() {
        let device = Default::default();
        // Shape [Y=2, X=2]
        // Y=0: X=0(0), X=1(1)
        // Y=1: X=0(10), X=1(11)
        let data_vec = vec![0.0, 1.0, 10.0, 11.0];
        let data = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(data_vec, burn::tensor::Shape::new([2, 2])),
            &device
        );
        
        let interpolator = LinearInterpolator::new();
        
        // Test X axis (0.5, 0) -> 0.5
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.0]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar() as f32;
        assert!((val - 0.5).abs() < 1e-5, "2D X axis interpolation failed, got {}", val);
        
        // Test Y axis (0, 0.5) -> 5.0
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.5]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar() as f32;
        assert!((val - 5.0).abs() < 1e-5, "2D Y axis interpolation failed, got {}", val);
    }
}
