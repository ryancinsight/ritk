use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_core::transform::TranslationTransform;
use ritk_registration::metric::{CorrelationRatio, CorrelationDirection, Metric};
use burn::tensor::Shape;
use burn::tensor::TensorData;

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
fn test_cr_perfect_match() {
    let size = 10;
    let count = size * size * size;
    // Create a ramp image
    let data: Vec<f32> = (0..count).map(|x| (x as f32) / (count as f32)).collect();
    let image = create_test_image(data.clone(), [size, size, size]);
    
    let device = Default::default();
    let transform = TranslationTransform::new(Tensor::zeros([3], &device));
    
    // Test MovingGivenFixed
    let metric = CorrelationRatio::<B>::new(
        32, 0.0, 1.0, 1.0, CorrelationDirection::MovingGivenFixed
    );
    
    let loss = metric.forward(&image, &image, &transform);
    let loss_val = loss.into_scalar();
    
    println!("Perfect match loss (MovingGivenFixed): {}", loss_val);
    // Correlation Ratio should be close to 1.0 for perfect functional dependence (Y = f(X))
    // Since we return negative CR, it should be close to -1.0
    assert!(loss_val < -0.9, "Loss should be close to -1.0 for perfect match, got {}", loss_val);
}

#[test]
fn test_cr_shift_sensitivity() {
    let size = 10;
    let count = size * size * size;
    let data1: Vec<f32> = (0..count).map(|x| (x as f32) / (count as f32)).collect();
    // Shifted data
    let data2: Vec<f32> = (0..count).map(|x| ((x + 5) % count) as f32 / (count as f32)).collect();
    
    let fixed = create_test_image(data1, [size, size, size]);
    let moving = create_test_image(data2, [size, size, size]);
    
    let device = Default::default();
    let transform = TranslationTransform::new(Tensor::zeros([3], &device));
    
    let metric = CorrelationRatio::<B>::new(
        32, 0.0, 1.0, 1.0, CorrelationDirection::MovingGivenFixed
    );
    
    let loss_shifted = metric.forward(&fixed, &moving, &transform);
    let loss_shifted_val = loss_shifted.into_scalar();
    
    println!("Shifted loss: {}", loss_shifted_val);
    
    // Compare with perfect match
    let loss_perfect = metric.forward(&fixed, &fixed, &transform).into_scalar();
    
    // Perfect match (lower loss/more negative) should be better than shifted
    assert!(loss_perfect < loss_shifted_val, "Perfect match should have lower loss than shifted");
}
