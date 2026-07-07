use super::*;
use burn_ndarray::NdArray;
use ritk_image::tensor::{Shape, TensorData};
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::TranslationTransform;

type B = NdArray<f32>;

fn create_test_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let tensor = Tensor::from_data(TensorData::new(data, Shape::new(shape)), &device);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let direction = Direction::identity();
    Image::new(tensor, origin, spacing, direction)
}

#[test]
fn test_ncc_identical() {
    let size = 10;
    let count = size * size * size;
    let data: Vec<f32> = (0..count).map(|x| x as f32).collect(); // Gradient
    let image = create_test_image(data.clone(), [size, size, size]);

    let device = Default::default();
    let transform = TranslationTransform::new(Tensor::zeros([3], &device));

    let ncc_metric = NormalizedCrossCorrelation::new();
    let loss = ncc_metric.forward(&image, &image, &transform);

    let loss_val = loss.into_scalar();
    // Identical images should have NCC = 1.0, so Loss = -1.0
    assert!(
        (loss_val + 1.0).abs() < 1e-4,
        "NCC for identical images should be 1.0 (loss -1.0), got {}",
        loss_val
    );
}

#[test]
fn test_ncc_linear_relationship() {
    let size = 10;
    let count = size * size * size;
    let data1: Vec<f32> = (0..count).map(|x| x as f32).collect();
    // data2 = 2 * data1 + 10. Linear relationship should still give NCC = 1.0
    let data2: Vec<f32> = data1.iter().map(|&x| 2.0 * x + 10.0).collect();

    let fixed = create_test_image(data1, [size, size, size]);
    let moving = create_test_image(data2, [size, size, size]);

    let device = Default::default();
    let transform = TranslationTransform::new(Tensor::zeros([3], &device));

    let ncc_metric = NormalizedCrossCorrelation::new();
    let loss = ncc_metric.forward(&fixed, &moving, &transform);

    let loss_val = loss.into_scalar();
    assert!(
        (loss_val + 1.0).abs() < 1e-4,
        "NCC for linear relationship should be 1.0 (loss -1.0), got {}",
        loss_val
    );
}

#[test]
fn test_ncc_inverse_relationship() {
    let size = 10;
    let count = size * size * size;
    let data1: Vec<f32> = (0..count).map(|x| x as f32).collect();
    // data2 = -data1. Inverse relationship should give NCC = -1.0, Loss = 1.0
    let data2: Vec<f32> = data1.iter().map(|&x| -x).collect();

    let fixed = create_test_image(data1, [size, size, size]);
    let moving = create_test_image(data2, [size, size, size]);

    let device = Default::default();
    let transform = TranslationTransform::new(Tensor::zeros([3], &device));

    let ncc_metric = NormalizedCrossCorrelation::new();
    let loss = ncc_metric.forward(&fixed, &moving, &transform);

    let loss_val = loss.into_scalar();
    assert!(
        (loss_val - 1.0).abs() < 1e-4,
        "NCC for inverse relationship should be -1.0 (loss 1.0), got {}",
        loss_val
    );
}

#[test]
fn test_ncc_uncorrelated() {
    let count = 100;
    let data1: Vec<f32> = (0..count).map(|x| x as f32).collect();
    // Alternating pattern should have low correlation with linear ramp
    let data2: Vec<f32> = (0..count)
        .map(|x| if x % 2 == 0 { 10.0 } else { -10.0 })
        .collect();

    let size = 100; // 1D like
    let fixed = create_test_image(data1, [size, 1, 1]);
    let moving = create_test_image(data2, [size, 1, 1]);

    let device = Default::default();
    let transform = TranslationTransform::new(Tensor::zeros([3], &device));

    let ncc_metric = NormalizedCrossCorrelation::new();
    let loss = ncc_metric.forward(&fixed, &moving, &transform);

    let loss_val = loss.into_scalar();
    assert!(
        loss_val.abs() < 0.5,
        "NCC for uncorrelated data should be low (close to 0), got loss {}",
        loss_val
    );
}
