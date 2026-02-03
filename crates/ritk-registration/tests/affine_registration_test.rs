use burn::tensor::{Tensor, TensorData};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::AffineTransform;
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::registration::Registration;
use ritk_registration::optimizer::AdamOptimizer;

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_registration_affine_translation() {
    let device = Default::default();

    // 1. Create 3D images (20x20x20)
    let d = 20;
    let shape = [d, d, d];
    
    // Analytic Ellipsoid function
    let make_ellipsoid = |c: [f32; 3], r: [f32; 3]| -> Vec<f32> {
        let mut data = Vec::with_capacity(d*d*d);
        for z in 0..d {
            for y in 0..d {
                for x in 0..d {
                    let px = x as f32;
                    let py = y as f32;
                    let pz = z as f32;
                    
                    let dx = px - c[0];
                    let dy = py - c[1];
                    let dz = pz - c[2];
                    
                    let val = (-(dx*dx)/(2.0*r[0]*r[0]) - (dy*dy)/(2.0*r[1]*r[1]) - (dz*dz)/(2.0*r[2]*r[2])).exp();
                    data.push(val);
                }
            }
        }
        data
    };

    // Fixed: Center (10, 10, 10), Radii (2, 3, 4)
    let fixed_data = make_ellipsoid([10.0, 10.0, 10.0], [2.0, 3.0, 4.0]);
    
    // Moving: Center (11, 12, 13), Radii (2, 3, 4)
    let moving_data = make_ellipsoid([11.0, 12.0, 13.0], [2.0, 3.0, 4.0]);

    let fixed_tensor = Tensor::<B, 3>::from_data(TensorData::new(fixed_data, shape), &device);
    let moving_tensor = Tensor::<B, 3>::from_data(TensorData::new(moving_data, shape), &device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);

    // 2. Initialize Affine Transform (Identity)
    // Center at (10, 10, 10)
    let center = Tensor::<B, 1>::from_floats([10.0, 10.0, 10.0], &device);
    let transform = AffineTransform::<B, 3>::identity(Some(center), &device);

    // 3. Setup Optimizer and Metric
    let optimizer = AdamOptimizer::new(1e-1);
    let metric = MeanSquaredError::new();
    let mut registration = Registration::new(optimizer, metric);

    // 4. Execute Registration
    let result_transform = registration.execute(&fixed, &moving, transform, 300, 1e-1).unwrap();

    // 5. Verify Result
    let t_est = result_transform.translation().into_data();
    let t_vals = t_est.as_slice::<f32>().unwrap();
    
    println!("Estimated Translation: {:?}", t_vals);
    
    // Expected translation: (1, 2, 3)
    assert!((t_vals[0] - 1.0).abs() < 0.1, "Translation X error: {}", t_vals[0]);
    assert!((t_vals[1] - 2.0).abs() < 0.1, "Translation Y error: {}", t_vals[1]);
    assert!((t_vals[2] - 3.0).abs() < 0.1, "Translation Z error: {}", t_vals[2]);
}

#[test]
fn test_registration_affine_scaling() {
    let device = Default::default();

    // 1. Create 3D images (20x20x20)
    let d = 20;
    let shape = [d, d, d];
    
    let make_ellipsoid = |c: [f32; 3], r: [f32; 3]| -> Vec<f32> {
        let mut data = Vec::with_capacity(d*d*d);
        for z in 0..d {
            for y in 0..d {
                for x in 0..d {
                    let px = x as f32;
                    let py = y as f32;
                    let pz = z as f32;
                    
                    let dx = px - c[0];
                    let dy = py - c[1];
                    let dz = pz - c[2];
                    
                    let val = (-(dx*dx)/(2.0*r[0]*r[0]) - (dy*dy)/(2.0*r[1]*r[1]) - (dz*dz)/(2.0*r[2]*r[2])).exp();
                    data.push(val);
                }
            }
        }
        data
    };

    // Fixed: Radii [4, 6, 8]
    let fixed_data = make_ellipsoid([10.0, 10.0, 10.0], [4.0, 6.0, 8.0]);
    
    // Moving: Radii [2, 3, 4]
    // Moving(x) = Fixed(S * x)
    // If Fixed is r=4, Moving is r=2 => |Sx| < 4 => |x| < 4/S = 2 => S = 2.
    let moving_data = make_ellipsoid([10.0, 10.0, 10.0], [2.0, 3.0, 4.0]);

    let fixed_tensor = Tensor::<B, 3>::from_data(TensorData::new(fixed_data, shape), &device);
    let moving_tensor = Tensor::<B, 3>::from_data(TensorData::new(moving_data, shape), &device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);

    // 2. Initialize Affine Transform (Identity)
    let center = Tensor::<B, 1>::from_floats([10.0, 10.0, 10.0], &device);
    let transform = AffineTransform::<B, 3>::identity(Some(center), &device);

    // 3. Setup Optimizer and Metric
    let optimizer = AdamOptimizer::new(1e-2);
    let metric = MeanSquaredError::new();
    let mut registration = Registration::new(optimizer, metric);

    // 4. Execute Registration
    // Scale optimization might be tricky, use reasonable LR
    let result_transform = registration.execute(&fixed, &moving, transform, 500, 1e-2).unwrap();

    // 5. Verify Result
    let m_est = result_transform.matrix().into_data();
    let m_vals = m_est.as_slice::<f32>().unwrap();
    
    println!("Estimated Matrix: {:?}", m_vals);
    
    // Expected Matrix: Diag(0.5, 0.5, 0.5)
    // Fixed (large, r=4) -> Moving (small, r=2).
    // T(p_fixed) = p_moving.
    // T(4) = 2 => S * 4 = 2 => S = 0.5.
    
    // Indices: 0, 4, 8
    assert!((m_vals[0] - 0.5).abs() < 0.1, "Scale X error: {}", m_vals[0]);
    assert!((m_vals[4] - 0.5).abs() < 0.1, "Scale Y error: {}", m_vals[4]);
    assert!((m_vals[8] - 0.5).abs() < 0.1, "Scale Z error: {}", m_vals[8]);
    
    // Off-diagonals should be close to 0
    assert!(m_vals[1].abs() < 0.2);
    assert!(m_vals[2].abs() < 0.2);
}
