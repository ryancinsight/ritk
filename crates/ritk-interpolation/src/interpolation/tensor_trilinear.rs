use ritk_image::tensor::{backend::Backend, Int, Tensor};

/// Trilinear interpolation for 3D tensors.
///
/// # Arguments
/// * `image` - Input image [B, C, D, H, W]
/// * `grid` - Sampling grid [B, 3, D, H, W] in voxel coordinates (z, y, x)
///
/// # Returns
/// * Interpolated image [B, C, D, H, W]
pub fn trilinear_interpolation<B: Backend>(
    image: Tensor<B, 5>,
    grid: Tensor<B, 5>,
) -> Tensor<B, 5> {
    let [b, c, d, h, w] = image.dims();
    let [_, _, out_d, out_h, out_w] = grid.dims();

    // Split grid into z, y, x
    let z = grid
        .clone()
        .slice([0..b, 0..1, 0..out_d, 0..out_h, 0..out_w]);
    let y = grid
        .clone()
        .slice([0..b, 1..2, 0..out_d, 0..out_h, 0..out_w]);
    let x = grid.slice([0..b, 2..3, 0..out_d, 0..out_h, 0..out_w]);

    // Floor and Ceil
    let z0 = z.clone().floor();
    let z1 = z0.clone().add_scalar(1.0);
    let y0 = y.clone().floor();
    let y1 = y0.clone().add_scalar(1.0);
    let x0 = x.clone().floor();
    let x1 = x0.clone().add_scalar(1.0);

    // Weights
    let wz1 = z.clone().sub(z0.clone());
    let wz0 = wz1.clone().neg().add_scalar(1.0);
    let wy1 = y.clone().sub(y0.clone());
    let wy0 = wy1.clone().neg().add_scalar(1.0);
    let wx1 = x.clone().sub(x0.clone());
    let wx0 = wx1.clone().neg().add_scalar(1.0);

    // Clip coordinates
    let z0_idx = z0.clamp(0.0, (d - 1) as f32).int();
    let z1_idx = z1.clamp(0.0, (d - 1) as f32).int();
    let y0_idx = y0.clamp(0.0, (h - 1) as f32).int();
    let y1_idx = y1.clamp(0.0, (h - 1) as f32).int();
    let x0_idx = x0.clamp(0.0, (w - 1) as f32).int();
    let x1_idx = x1.clamp(0.0, (w - 1) as f32).int();

    // Flatten image once: [B, C, D*H*W]
    let flat_img = image.reshape([b, c, d * h * w]);

    // Pre-calculate strides
    let stride_d = (h * w) as i32;
    let stride_h = w as i32;

    // Pre-calculate offsets
    let z0_off = z0_idx.mul_scalar(stride_d);
    let z1_off = z1_idx.mul_scalar(stride_d);
    let y0_off = y0_idx.mul_scalar(stride_h);
    let y1_off = y1_idx.mul_scalar(stride_h);

    // Calculate combined indices once
    let idx_00 = z0_off.clone() + y0_off.clone();
    let idx_01 = z0_off.clone() + y1_off.clone();
    let idx_10 = z1_off.clone() + y0_off.clone();
    let idx_11 = z1_off.clone() + y1_off.clone();

    let out_elements = out_d * out_h * out_w;
    let idx_000 = (idx_00.clone() + x0_idx.clone()).reshape([b, 1, out_elements]);
    let idx_001 = (idx_00 + x1_idx.clone()).reshape([b, 1, out_elements]);
    let idx_010 = (idx_01.clone() + x0_idx.clone()).reshape([b, 1, out_elements]);
    let idx_011 = (idx_01 + x1_idx.clone()).reshape([b, 1, out_elements]);
    let idx_100 = (idx_10.clone() + x0_idx.clone()).reshape([b, 1, out_elements]);
    let idx_101 = (idx_10 + x1_idx.clone()).reshape([b, 1, out_elements]);
    let idx_110 = (idx_11.clone() + x0_idx.clone()).reshape([b, 1, out_elements]);
    let idx_111 = (idx_11 + x1_idx.clone()).reshape([b, 1, out_elements]);

    let mut out_channels = Vec::with_capacity(c);

    for c_idx in 0..c {
        let channel_img = flat_img
            .clone()
            .slice([0..b, c_idx..c_idx + 1, 0..d * h * w]);

        let gather_val = |idx: &Tensor<B, 3, Int>| -> Tensor<B, 5> {
            channel_img
                .clone()
                .gather(2, idx.clone())
                .reshape([b, 1, out_d, out_h, out_w])
        };

        let v000 = gather_val(&idx_000);
        let v001 = gather_val(&idx_001);
        let v010 = gather_val(&idx_010);
        let v011 = gather_val(&idx_011);
        let v100 = gather_val(&idx_100);
        let v101 = gather_val(&idx_101);
        let v110 = gather_val(&idx_110);
        let v111 = gather_val(&idx_111);

        // Interpolate X first
        let w00 = v000 * wx0.clone() + v001 * wx1.clone();
        let w01 = v010 * wx0.clone() + v011 * wx1.clone();
        let w10 = v100 * wx0.clone() + v101 * wx1.clone();
        let w11 = v110 * wx0.clone() + v111 * wx1.clone();

        // Interpolate Y
        let w0 = w00 * wy0.clone() + w01 * wy1.clone();
        let w1 = w10 * wy0.clone() + w11 * wy1.clone();

        // Interpolate Z
        let c_out = w0 * wz0.clone() + w1 * wz1.clone();
        out_channels.push(c_out);
    }

    Tensor::cat(out_channels, 1)
}

#[cfg(test)]
mod trilinear_tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_image::tensor::{Shape, TensorData};

    type B = NdArray<f32>;

    #[test]
    fn test_trilinear_interpolation_basic() {
        let device = Default::default();

        // Create a 1x1x2x2x2 image (B=1, C=1, D=2, H=2, W=2)
        // Values:
        // z=0: [1.0, 2.0; 3.0, 4.0]
        // z=1: [5.0, 6.0; 7.0, 8.0]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let image =
            Tensor::<B, 5>::from_data(TensorData::new(data, Shape::new([1, 1, 2, 2, 2])), &device);

        // Create a sampling grid [B=1, 3, D=1, H=1, W=1]
        // We want to sample at (z=0.5, y=0.5, x=0.5)
        let grid_data: Vec<f32> = vec![0.5, 0.5, 0.5];
        let grid = Tensor::<B, 5>::from_data(
            TensorData::new(grid_data, Shape::new([1, 3, 1, 1, 1])),
            &device,
        );

        let result = trilinear_interpolation(image, grid);
        let result_val = result.into_data().as_slice::<f32>().unwrap()[0];

        // The expected value is the average of all 8 corners: 4.5
        assert!((result_val - 4.5).abs() < 1e-5);
    }

    #[test]
    fn test_trilinear_interpolation_channels() {
        let device = Default::default();

        // Create a 1x2x2x2x2 image (B=1, C=2, D=2, H=2, W=2)
        // Channel 0: all 10.0
        // Channel 1: all 20.0
        let mut data: Vec<f32> = vec![10.0; 8];
        data.extend(vec![20.0; 8]);

        let image =
            Tensor::<B, 5>::from_data(TensorData::new(data, Shape::new([1, 2, 2, 2, 2])), &device);

        // Grid at (0.0, 0.0, 0.0)
        let grid_data: Vec<f32> = vec![0.0, 0.0, 0.0];
        let grid = Tensor::<B, 5>::from_data(
            TensorData::new(grid_data, Shape::new([1, 3, 1, 1, 1])),
            &device,
        );

        let result = trilinear_interpolation(image, grid);
        let result_slice = result.into_data();
        let result_vals = result_slice.as_slice::<f32>().unwrap();

        assert!((result_vals[0] - 10.0).abs() < 1e-5);
        assert!((result_vals[1] - 20.0).abs() < 1e-5);
    }
}

#[cfg(test)]
#[path = "tests_trilinear.rs"]
mod tests;
