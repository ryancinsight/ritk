//! Tests for neighborhood_connected
//! Extracted from the main module to keep the 500-line structural limit.
use super::*;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

fn make_image(values: Vec<f32>, shape: [usize; 3]) -> Image<TestBackend, 3> {
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let td = TensorData::new(values, Shape::new(shape));
    let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

fn get_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn count_foreground(image: &Image<TestBackend, 3>) -> usize {
    get_values(image).iter().filter(|&&v| v > 0.5).count()
}

#[path = "adversarial.rs"]
mod adversarial;
#[path = "negative.rs"]
mod negative;
#[path = "positive.rs"]
mod positive;
#[path = "predicate.rs"]
mod predicate;
#[path = "structural.rs"]
mod structural;
