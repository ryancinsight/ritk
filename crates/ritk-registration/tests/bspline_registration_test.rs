use burn::tensor::{Tensor, TensorData};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::{BSplineTransform, Transform};
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::registration::Registration;
use ritk_registration::optimizer::AdamOptimizer;

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_registration_bspline_3d() {
    let device = Default::default();

    // 1. Create 3D images (10x10x10)
    let d = 10;
    let shape = [d, d, d];
    
    // Helper to create a sphere
    let make_sphere = |center: [f32; 3], radius: f32| -> Vec<f32> {
        let mut data = Vec::with_capacity(d*d*d);
        for z in 0..d {
            for y in 0..d {
                for x in 0..d {
                    let dz = (z as f32) - center[2];
                    let dy = (y as f32) - center[1];
                    let dx = (x as f32) - center[0];
                    let dist_sq = dx*dx + dy*dy + dz*dz;
                    // Gaussian-like sphere
                    let val = (-dist_sq / (2.0 * radius * radius)).exp();
                    data.push(val);
                }
            }
        }
        data
    };

    // Fixed Image: Sphere at (5, 5, 5)
    let fixed_data = make_sphere([5.0, 5.0, 5.0], 2.0);
    
    // Moving Image: Sphere at (6, 6, 6) (Shifted by +1, +1, +1)
    // We want the transform to map Fixed coordinates to Moving coordinates.
    // T(x) = x + u(x).
    // If Fixed has feature at x=5, and Moving has feature at x=6.
    // We want T(5) = 6.
    // So u(5) should be +1.
    let moving_data = make_sphere([6.0, 6.0, 6.0], 2.0);

    let fixed_tensor = Tensor::<B, 3>::from_data(TensorData::new(fixed_data, shape), &device);
    let moving_tensor = Tensor::<B, 3>::from_data(TensorData::new(moving_data, shape), &device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);

    // 2. Initialize BSpline Transform
    // Grid size 5x5x5 covers the 10x10x10 volume with spacing 2.5
    let grid_size = [5, 5, 5];
    let physical_size = [9.0, 9.0, 9.0]; // 0 to 9
    
    // Initialize coefficients to zero
    let num_control_points = 5 * 5 * 5;
    let coeffs = Tensor::<B, 2>::zeros([num_control_points, 3], &device);
    
    let transform = BSplineTransform::<B, 3>::new(grid_size, physical_size, coeffs);

    // 3. Setup Optimizer and Metric
    let optimizer = AdamOptimizer::new(1e-1);
    let metric = MeanSquaredError::new();
    let mut registration = Registration::new(optimizer, metric);

    // 4. Execute Registration
    // We expect the deformation field to learn a shift of ~ +1.0 in the center region.
    let result_transform = registration.execute(&fixed, &moving, transform, 200, 1e-1);

    // 5. Verify Result
    // Check deformation at the center (5, 5, 5)
    let center_point = Tensor::<B, 2>::from_data(TensorData::from([[5.0, 5.0, 5.0]]), &device);
    let transformed_point = result_transform.transform_points(center_point);
    
    let transformed_data = transformed_point.into_data();
    let transformed_vals = transformed_data.as_slice::<f32>().unwrap();
    
    println!("Transformed Center: {:?}", transformed_vals);
    
    // We expect T(5,5,5) to be close to (6,6,6)
    let err_x = (transformed_vals[0] - 6.0).abs();
    let err_y = (transformed_vals[1] - 6.0).abs();
    let err_z = (transformed_vals[2] - 6.0).abs();
    
    println!("Errors: x={}, y={}, z={}", err_x, err_y, err_z);
    
    assert!(err_x < 0.5, "Error X too high: {}", err_x);
    assert!(err_y < 0.5, "Error Y too high: {}", err_y);
    assert!(err_z < 0.5, "Error Z too high: {}", err_z);
}
