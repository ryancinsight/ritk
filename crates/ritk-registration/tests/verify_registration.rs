
use burn::tensor::{Tensor, TensorData};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::{TranslationTransform, Transform};
use ritk_core::spatial::{Point2, Spacing2, Direction2};
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::registration::Registration;
use ritk_registration::optimizer::GradientDescent;

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_verify_registration_coordinates() {
    let device = Default::default();

    // Create 2D images: 10x10
    // Gaussian blob
    let d = 10;
    let center = 5.0;
    let sigma = 2.0;
    
    // x is inner loop (Dim 1), y is outer loop (Dim 0)
    let make_blob = |offset_x: f32, offset_y: f32| -> Vec<f32> {
        let mut data = Vec::with_capacity(d*d);
        for y in 0..d {
            for x in 0..d {
                // Physical coordinate matches index because spacing=1, origin=0
                let px = x as f32;
                let py = y as f32;
                
                // Blob center in physical space
                let cx = center + offset_x;
                let cy = center + offset_y;
                
                let dx = px - cx;
                let dy = py - cy;
                let val = (- (dx*dx + dy*dy) / (2.0 * sigma * sigma)).exp();
                data.push(val);
            }
        }
        data
    };
    
    // Fixed image: centered at (5, 5)
    let fixed_data_vec = make_blob(0.0, 0.0);
    
    // Moving image: centered at (5+2, 5+1) = (7, 6)
    // Physical shift: +2 in X, +1 in Y.
    let moving_data_vec = make_blob(2.0, 1.0);
    
    let shape = [d, d];
    let fixed_tensor = Tensor::<B, 2>::from_data(TensorData::new(fixed_data_vec, shape), &device);
    let moving_tensor = Tensor::<B, 2>::from_data(TensorData::new(moving_data_vec, shape), &device);
    
    let origin = Point2::new([0.0, 0.0]);
    let spacing = Spacing2::new([1.0, 1.0]);
    let direction = Direction2::identity();

    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);
    
    // Initial Transform: Identity (0, 0)
    let transform = TranslationTransform::<B, 2>::new(Tensor::from_data(TensorData::from([0.0, 0.0]), &device));
    
    // Optimizer
    let optimizer = GradientDescent::new(1.0);
    
    // Metric
    let metric = MeanSquaredError::new();
    
    // Registration
    let mut registration = Registration::new(optimizer, metric);
    
    // Execute
    // We expect T(x) = x + t.
    // Fixed(x) ~= Moving(x + t).
    // Fixed blob at x=5. Moving blob at x=7.
    // Moving(7) is peak.
    // We want T(5) = 7.
    // 5 + t = 7 => t = 2.
    // Same for y: 5 + t = 6 => t = 1.
    // Expected t = [2.0, 1.0].
    
    let result_transform = registration.execute(&fixed, &moving, transform, 500, 1.0);
    
    let p = Tensor::<B, 2>::zeros([1, 2], &device);
    let t_p = result_transform.transform_points(p);
    
    let t_p_data = t_p.into_data();
    let t_p_slice = t_p_data.as_slice::<f32>().unwrap();
    
    let recovered_x = t_p_slice[0]; // Dim 0 should be X
    let recovered_y = t_p_slice[1]; // Dim 1 should be Y
    
    println!("Recovered Translation: [{}, {}]", recovered_x, recovered_y);
    
    assert!((recovered_x - 2.0).abs() < 0.1, "Expected X=2.0, got {}", recovered_x);
    assert!((recovered_y - 1.0).abs() < 0.1, "Expected Y=1.0, got {}", recovered_y);
}
