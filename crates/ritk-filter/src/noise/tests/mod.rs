mod gaussian;
mod salt_pepper;
mod shot;
mod speckle;

use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_core::image::Image;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::test_support as ts;

type B = LegacyBurnBackend;

fn make_image(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}
