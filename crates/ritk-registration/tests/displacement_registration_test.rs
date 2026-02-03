use burn::tensor::{Tensor, TensorData, Shape};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::{DisplacementField, DisplacementFieldTransform};
use ritk_core::interpolation::LinearInterpolator;
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::registration::{Registration, RegistrationConfig};
use ritk_registration::optimizer::AdamOptimizer as Adam;

type B = Autodiff<NdArray<f32>>;
const D: usize = 2;

#[test]
fn test_displacement_registration_2d() {
    let device = Default::default();

    // 1. Create Images (10x10)
    let d = 10;
    let center = 5.0;
    let sigma = 1.5;
    
    let make_blob = |offset_x: f32, offset_y: f32| -> Vec<f32> {
        let mut data = Vec::with_capacity(d*d);
        for y in 0..d {
            for x in 0..d {
                let dx = (x as f32) - (center + offset_x);
                let dy = (y as f32) - (center + offset_y);
                let val = (- (dx*dx + dy*dy) / (2.0 * sigma * sigma)).exp();
                data.push(val);
            }
        }
        data
    };
    
    // Fixed: Center (5,5)
    let fixed_data = make_blob(0.0, 0.0);
    // Moving: Center (7,7) (Shifted by +2,+2)
    // We expect D(x) = +2
    let moving_data = make_blob(2.0, 2.0);
    
    let shape = [d, d];
    let fixed_tensor = Tensor::<B, 2>::from_data(TensorData::new(fixed_data, Shape::new(shape)), &device);
    let moving_tensor = Tensor::<B, 2>::from_data(TensorData::new(moving_data, Shape::new(shape)), &device);
    
    let origin = Point::new([0.0; D]);
    let spacing = Spacing::new([1.0; D]);
    let direction = Direction::identity();
    
    let fixed = Image::new(fixed_tensor.clone(), origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor.clone(), origin.clone(), spacing.clone(), direction.clone());
    
    // 2. Initialize Displacement Field (Zero)
    // Grid matches image size 10x10
    let comp_shape = [d, d];
    
    // Components[0] adds to X, Components[1] adds to Y
    let comp_x = Tensor::<B, 2>::zeros(comp_shape, &device);
    let comp_y = Tensor::<B, 2>::zeros(comp_shape, &device);
    
    let field = DisplacementField::new(
        vec![comp_x, comp_y], 
        origin.clone(),
        spacing.clone(),
        direction.clone(),
    );
    
    let transform = DisplacementFieldTransform::new(field, LinearInterpolator::new());
    
    // 3. Optimizer & Metric
    let optimizer = Adam::new(0.1); // Learning rate
    let metric = MeanSquaredError::new();
    
    let config = RegistrationConfig::new()
        .with_log_interval(10)
        .without_early_stopping(); // Run full iterations
        
    let mut registration = Registration::with_config(optimizer, metric, config);
    
    // 4. Execute
    let iterations = 100;
    let learning_rate = 0.1;
    let result_transform = registration.execute(&fixed, &moving, transform, iterations, learning_rate).unwrap();
    
    // 5. Verify
    // Check center displacement
    // Center index is roughly (5,5)
    let comps = result_transform.field().components();
    // components are [X, Y]
    // data is [Y, X] (10x10)
    // slice uses [dim0_range, dim1_range] -> [y_range, x_range]
    let dx = comps[0].clone().slice([5..6, 5..6]).into_scalar() as f32;
    let dy = comps[1].clone().slice([5..6, 5..6]).into_scalar() as f32;
    
    println!("Recovered Displacement at center: dx={}, dy={}", dx, dy);
    
    // Expect approx 2.0
    // Tolerance slightly loose as it's a small grid and boundaries affect it
    assert!((dx - 2.0).abs() < 0.5, "X displacement should be approx 2.0, got {}", dx);
    assert!((dy - 2.0).abs() < 0.5, "Y displacement should be approx 2.0, got {}", dy);
}
