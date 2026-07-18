mod gaussian;
mod salt_pepper;
mod shot;
mod speckle;

use super::*;
use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

fn make_image(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}
