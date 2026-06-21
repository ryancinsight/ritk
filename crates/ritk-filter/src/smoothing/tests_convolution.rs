//! Unit tests for [`SpatialConvolutionFilter`].

use crate::smoothing::SpatialConvolutionFilter;
use burn::tensor::{Shape, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(vals: &[f32], dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(vals.to_vec(), Shape::new(dims));
    let tensor = burn::tensor::Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([1.1, 2.2, 3.3]),
        Spacing::new([0.5, 0.5, 0.5]),
        Direction::identity(),
    )
}

#[test]
fn identity_kernel_is_noop() {
    let vals: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = make_image(&vals, [3, 3, 3]);

    // Identity kernel size 3x3x3: all zeros except centre (1, 1, 1) = 1.0
    let mut kernel = vec![0.0_f32; 27];
    kernel[13] = 1.0;

    let filter = SpatialConvolutionFilter::new(kernel, [3, 3, 3]).unwrap();
    let out = filter.apply(&img).unwrap();

    let (res, _) = ritk_tensor_ops::extract_vec(&out).unwrap();
    assert_eq!(res, vals);
    assert_eq!(out.origin().to_array(), [1.1, 2.2, 3.3]);
}

#[test]
fn box_blur_correct() {
    // 3x3x3 image with a single 1.0 at centre
    let mut vals = vec![0.0_f32; 27];
    vals[13] = 27.0;
    let img = make_image(&vals, [3, 3, 3]);

    // Box blur kernel (all ones)
    let kernel = vec![1.0_f32; 27];

    let filter = SpatialConvolutionFilter::new(kernel, [3, 3, 3]).unwrap();
    let out = filter.apply(&img).unwrap();

    let (res, _) = ritk_tensor_ops::extract_vec(&out).unwrap();
    // Every voxel gets kernel_sum * input_val = 1 * 27 = 27.0 (due to Neumann clamping
    // clamping the single 27.0 at centre for all neighborhood evaluations)
    // Wait, let's verify exact math:
    // For output voxel (1,1,1), center is at (1,1,1). Kernel covers all input voxels exactly once.
    // Sum = 1.0 * 27.0 = 27.0.
    // For corner (0,0,0), kernel covers input voxels clamping to nearest.
    // Clamped indices: for ik in 0..3, index is clamp(0 + ik - 1, 0, 2).
    // The only source index that reaches (1,1,1) is when ik = 2.
    // So sum has one contribution from (1,1,1) (weight 1.0 * 27.0).
    // Thus all output voxels should be 27.0!
    for &val in &res {
        assert!((val - 27.0).abs() < 1e-4);
    }
}
