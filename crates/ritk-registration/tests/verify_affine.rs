
use burn::tensor::{Tensor, TensorData};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::AffineTransform;
use ritk_core::spatial::{Point2, Spacing2, Direction2};
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::registration::Registration;
use ritk_registration::optimizer::GradientDescent;

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_verify_affine_scale_recovery() {
    let device = Default::default();

    // Create 2D images: 20x20 to have enough space
    // Gaussian blob
    let d = 20;
    let center = 10.0;
    let _sigma = 3.0;
    
    // Helper to make blob
    // If scale_x = 2.0, then feature at x in physical space corresponds to x/2 in index space of "scaled" object?
    // Actually, let's just generate two images:
    // Fixed: Standard blob at center.
    // Moving: Stretched blob.
    // If Moving(x) = Fixed(T(x)) and T(x) = S*x.
    // Then Moving(S*x) matches Fixed(x).
    
    // Let's do it the other way:
    // Fixed image has a blob.
    // Moving image is the SAME blob but "scaled" in physical space? 
    // No, usually Moving image is distorted.
    // We want T such that Moving(T(x)) ~ Fixed(x).
    
    // Let's simulate:
    // Fixed: Blob at (10, 10).
    // Moving: Blob at (5, 5) but with same "shape" in pixel space?
    // No, that's translation.
    
    // Scale:
    // Fixed: Blob width sigma=3.
    // Moving: Blob width sigma=1.5 (narrower).
    // If we map Fixed -> Moving, we need to "shrink" the coordinates?
    // If Moving(x) has sigma=1.5. Fixed(x) has sigma=3.
    // Moving(x/2) would have sigma=3?
    // exp(- (x/2)^2 / (2 * 1.5^2)) = exp(- x^2 / (4 * 2.25)) = exp(-x^2 / 9).
    // Fixed: exp(- x^2 / (2 * 3^2)) = exp(-x^2 / 18).
    // Wait, 4*2.25 = 9. 2*9 = 18.
    // So yes, Moving(x/2) matches Fixed(x).
    // So T(x) = 0.5 * x.
    // Expected Scale = 0.5.
    
    let make_blob = |s: f32| -> Vec<f32> {
        let mut data = Vec::with_capacity(d*d);
        for y in 0..d {
            for x in 0..d {
                let px = x as f32;
                let py = y as f32;
                let dx = px - center;
                let dy = py - center;
                let val = (- (dx*dx + dy*dy) / (2.0 * s * s)).exp();
                data.push(val);
            }
        }
        data
    };
    
    // Fixed: Sigma=3.0
    let fixed_data_vec = make_blob(3.0);
    
    // Moving: Sigma=1.5
    let moving_data_vec = make_blob(1.5);
    
    let shape = [d, d];
    let fixed_tensor = Tensor::<B, 2>::from_data(TensorData::new(fixed_data_vec, shape), &device);
    let moving_tensor = Tensor::<B, 2>::from_data(TensorData::new(moving_data_vec, shape), &device);
    
    let origin = Point2::new([0.0, 0.0]);
    let spacing = Spacing2::new([1.0, 1.0]);
    let direction = Direction2::identity();

    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);
    
    // Initial Transform: Identity
    // Center of rotation should be the image center (10, 10) to avoid translation coupling
    let center_tensor = Tensor::<B, 1>::from_data(TensorData::from([10.0, 10.0]), &device);
    let transform = AffineTransform::<B, 2>::identity(Some(center_tensor), &device);
    
    // Optimizer
    // Use smaller learning rate for matrix parameters
    let optimizer = GradientDescent::new(0.5);
    
    // Metric
    let metric = MeanSquaredError::new();
    
    // Registration
    let mut registration = Registration::new(optimizer, metric);
    
    // Execute
    // We expect T(x) ~ 0.5 * x (around center).
    // Actually T(x) = A(x-c) + c + t.
    // If we only scale, t=0.
    // A should be diag(0.5, 0.5).
    
    let result_transform = registration.execute(&fixed, &moving, transform, 2000, 0.5);
    
    let matrix = result_transform.matrix();
    let data = matrix.into_data();
    let slice = data.as_slice::<f32>().unwrap();
    
    println!("Recovered Matrix: {:?}", slice);
    
    // Check diagonal elements
    let s_x = slice[0];
    let s_y = slice[3]; // 2x2 matrix: 0, 1, 2, 3 -> (0,0), (0,1), (1,0), (1,1)
    
    // We expect 0.5
    assert!((s_x - 0.5).abs() < 0.1, "Expected Scale X=0.5, got {}", s_x);
    assert!((s_y - 0.5).abs() < 0.1, "Expected Scale Y=0.5, got {}", s_y);
}
