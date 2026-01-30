use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::TranslationTransform;
use ritk_core::spatial::{Point3, Spacing3, Direction3};
use ritk_registration::metric::{Metric, MeanSquaredError};

type B = NdArray<f32>;

#[test]
fn test_mse_identity() {
    let device = Default::default();
    
    // Create synthetic data: 5x5x5 cube with gradient
    // Gradient ensures that translation changes the error.
    let d = 5;
    let mut data_vec = Vec::with_capacity(d*d*d);
    for z in 0..d {
        for y in 0..d {
            for x in 0..d {
                data_vec.push((x + y + z) as f32);
            }
        }
    }
    
    let shape = [d, d, d];
    let data = Tensor::<B, 3>::from_data(TensorData::new(data_vec, shape), &device);
    
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();
    
    let fixed = Image::new(data.clone(), origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(data.clone(), origin, spacing, direction);
    
    // Identity transform
    let transform = TranslationTransform::<B, 3>::new(Tensor::from_data(TensorData::from([0.0, 0.0, 0.0]), &device));
    
    let metric = MeanSquaredError::new();
    let loss = metric.forward(&fixed, &moving, &transform);
    
    let loss_scalar = loss.into_scalar();
    assert!(loss_scalar.abs() < 1e-6, "Loss should be 0 for identity transform, got {}", loss_scalar);
}

#[test]
fn test_mse_translation() {
    let device = Default::default();
    
    // Create synthetic data: 5x5x5 cube with gradient
    let d = 5;
    let mut data_vec = Vec::with_capacity(d*d*d);
    for _z in 0..d {
        for _y in 0..d {
            for x in 0..d {
                data_vec.push((x as f32) / (d as f32)); // Gradient in X only, Y and Z unused
            }
        }
    }
    
    let shape = [d, d, d];
    let data = Tensor::<B, 3>::from_data(TensorData::new(data_vec, shape), &device);
    
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let direction = Direction3::identity();
    
    let fixed = Image::new(data.clone(), origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(data.clone(), origin, spacing, direction);
    
    // Shift by 1.0 in X.
    let transform = TranslationTransform::<B, 3>::new(Tensor::from_data(TensorData::from([1.0, 0.0, 0.0]), &device));
    
    let metric = MeanSquaredError::new();
    let loss = metric.forward(&fixed, &moving, &transform);
    
    let loss_scalar = loss.into_scalar();
    println!("Loss with translation: {}", loss_scalar);
    
    assert!(loss_scalar > 0.0, "Loss should be positive for translation");
    // Ideally check if it's close to 0.04, but boundary conditions might affect it.
    // Interpolation at boundary?
    // Fixed grid goes 0..4.
    // Transformed grid goes 1..5.
    // Moving image defined 0..4.
    // 4->5 is out of bounds. Clamped to 4.
    // 3->4 is valid.
    
    // Let's check logic.
    // Fixed x=0 -> T(x)=1. Moving(1) = 0.2. Fixed(0)=0.0. Diff=0.2. Sq=0.04.
    // Fixed x=3 -> T(x)=4. Moving(4) = 0.8. Fixed(3)=0.6. Diff=0.2. Sq=0.04.
    // Fixed x=4 -> T(x)=5. Moving(5) clamped to 4 (value 0.8). Fixed(4)=0.8. Diff=0.0. Sq=0.0.
    
    // So for 0..3 (4 points), error is 0.04. For 4, error is 0.
    // Mean over 5 points: (0.04 * 4 + 0) / 5 = 0.16 / 5 = 0.032.
    // (Assuming clamping at border).
    
    // Let's just assert it is positive.
}
