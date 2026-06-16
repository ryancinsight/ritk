//! Tests for neighborhood_connected
//! Extracted from the main module to keep the 500-line structural limit.
use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;
use ritk_core::image::Image;

type TestBackend = NdArray<f32>;


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
