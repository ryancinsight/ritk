use super::*;
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

#[test]
fn test_cr_creation() {
    let metric = CorrelationRatio::<TestBackend>::default_params(&Default::default());
    assert_eq!(metric.histogram_calculator.num_bins, 32);
    assert_eq!(metric.direction, CorrelationDirection::MovingGivenFixed);
}

#[test]
fn test_cr_name() {
    let metric = CorrelationRatio::<TestBackend>::default_params(&Default::default());
    assert_eq!(
        <CorrelationRatio<TestBackend> as Metric<TestBackend, 3>>::name(&metric),
        "Correlation Ratio"
    );
}

#[test]
fn test_cr_gradient_non_zero() {
    use ritk_image::burn::backend::Autodiff;
    use ritk_image::tensor::{Shape, TensorData};
    use ritk_spatial::{Direction, Point, Spacing};
    use ritk_transform::TranslationTransform;

    type B = Autodiff<TestBackend>;

    let device = Default::default();
    let size = 5;
    let count = size * size * size;

    // Fixed: gradient ramp 0-255
    let fixed_data: Vec<f32> = (0..count)
        .map(|x| x as f32 * 255.0 / count as f32)
        .collect();
    // Moving: same ramp shifted by 1 along x
    let moving_data: Vec<f32> = (1..count + 1)
        .map(|x| {
            let val = x as f32 * 255.0 / count as f32;
            val.clamp(0.0, 255.0)
        })
        .collect();

    let fixed_t = Tensor::<B, 3>::from_data(
        TensorData::new(fixed_data, Shape::new([size, size, size])),
        &device,
    );
    let moving_t = Tensor::<B, 3>::from_data(
        TensorData::new(moving_data, Shape::new([size, size, size])),
        &device,
    );

    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let direction = Direction::identity();

    let fixed = Image::new(fixed_t, origin, spacing, direction);
    let moving = Image::new(moving_t, origin, spacing, direction);

    let translation = Tensor::<B, 1>::zeros([3], &device).require_grad();
    let transform = TranslationTransform::<B, 3>::new(translation.clone());

    let metric = CorrelationRatio::<B>::default_params(&device);
    let loss = metric.forward(&fixed, &moving, &transform);
    let grads = loss.backward();

    let translation_grad = translation.grad(&grads).unwrap();
    let grad_data = translation_grad.into_data();
    let grad_vals = grad_data.as_slice::<f32>().unwrap();

    // Gradient should be non-zero (images are misaligned)
    assert!(
        grad_vals.iter().any(|&g| g.abs() > 1e-6),
        "CR gradient should be non-zero for misaligned images, got {:?}",
        grad_vals
    );
}

#[test]
fn test_cr_identical_images() {
    use ritk_image::tensor::{Shape, TensorData};
    use ritk_spatial::{Direction, Point, Spacing};
    use ritk_transform::TranslationTransform;

    let size = 5;
    let count = size * size * size;
    let data: Vec<f32> = (0..count)
        .map(|x| (x as f32 * 255.0) / count as f32)
        .collect(); // Gradient 0-255

    let device = Default::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(data, Shape::new([size, size, size])),
        &device,
    );
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let direction = Direction::identity();

    let fixed = Image::new(tensor.clone(), origin, spacing, direction);
    let moving = Image::new(tensor, origin, spacing, direction);

    let transform =
        TranslationTransform::<TestBackend, 3>::new(Tensor::<TestBackend, 1>::zeros([3], &device));

    let cr_metric = CorrelationRatio::<TestBackend>::default_params(&Default::default());
    let loss = cr_metric.forward(&fixed, &moving, &transform);

    let loss_val = loss.into_scalar();

    // Identical images should have CR near 1.0 (loss near -1.0)
    assert!(
        (loss_val + 1.0).abs() < 0.1,
        "CR for identical images should be approx 1.0 (loss -1.0), got {}",
        loss_val
    );
}
