//! Tests for neighborhood_connected
//! Extracted from the main module to keep the 500-line structural limit.
use super::*;
use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::tensor::Tensor;
use ritk_image::test_support::make_image;

type TestBackend = SequentialBackend;

fn get_values(image: &Image<f32, TestBackend, 3>) -> Vec<f32> {
    image.data().to_vec()
}

fn count_foreground(image: &Image<f32, TestBackend, 3>) -> usize {
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
