mod gaussian;
mod salt_pepper;
mod shot;
mod speckle;

use super::*;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::test_support as ts;

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}
