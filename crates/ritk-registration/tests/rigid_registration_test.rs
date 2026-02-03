use burn::tensor::{Tensor, TensorData};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::RigidTransform;
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::registration::Registration;
use ritk_registration::optimizer::AdamOptimizer;

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_registration_rigid_3d() {
    let device = Default::default();

    // 1. Create 3D images (20x20x20)
    let d = 20;
    let shape = [d, d, d];
    
    // Analytic Ellipsoid function
    // Center c, radii r_vec, rotation angle theta (around Z)
    let make_ellipsoid = |c: [f32; 3], r: [f32; 3], theta: f32| -> Vec<f32> {
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        
        let mut data = Vec::with_capacity(d*d*d);
        for z in 0..d {
            for y in 0..d {
                for x in 0..d {
                    // Physical coords (spacing 1, origin 0)
                    let px = x as f32;
                    let py = y as f32;
                    let pz = z as f32;
                    
                    // Translate to local coords relative to center
                    let dx = px - c[0];
                    let dy = py - c[1];
                    let dz = pz - c[2];
                    
                    // Inverse Rotate (to align with axes)
                    // R(theta) * v_local = v_world_diff
                    // v_local = R(-theta) * v_world_diff
                    let lx = dx * cos_t + dy * sin_t;
                    let ly = -dx * sin_t + dy * cos_t;
                    let lz = dz;
                    
                    let val = (-(lx*lx)/(2.0*r[0]*r[0]) - (ly*ly)/(2.0*r[1]*r[1]) - (lz*lz)/(2.0*r[2]*r[2])).exp();
                    data.push(val);
                }
            }
        }
        data
    };

    // Fixed: Center (10, 10, 10), Radii (2, 3, 4), Angle 0
    let fixed_data = make_ellipsoid([10.0, 10.0, 10.0], [2.0, 3.0, 4.0], 0.0);
    
    // Moving: Center (11, 11, 10), Radii (2, 3, 4), Angle 0.0 (Translation Only)
    let moving_data = make_ellipsoid([11.0, 11.0, 10.0], [2.0, 3.0, 4.0], 0.0);

    let fixed_tensor = Tensor::<B, 3>::from_data(TensorData::new(fixed_data, shape), &device);
    let moving_tensor = Tensor::<B, 3>::from_data(TensorData::new(moving_data, shape), &device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);

    // 2. Initialize Rigid Transform
    let translation = Tensor::<B, 1>::zeros([3], &device); 
    let rotation = Tensor::<B, 1>::zeros([3], &device);
    let center = Tensor::<B, 1>::from_floats([10.0, 10.0, 10.0], &device); 
    
    let transform = RigidTransform::<B, 3>::new(translation, rotation, center);

    // 3. Setup Optimizer and Metric
    let optimizer = AdamOptimizer::new(1e-1);
    let metric = MeanSquaredError::new();
    let mut registration = Registration::new(optimizer, metric);

    // 4. Execute Registration
    let result_transform = registration.execute(&fixed, &moving, transform, 300, 1e-1).unwrap();

    // 5. Verify Result
    let t_est = result_transform.translation().into_data();
    let t_vals = t_est.as_slice::<f32>().unwrap();
    
    println!("Estimated Translation (Only): {:?}", t_vals);
    
    assert!((t_vals[0] - 1.0).abs() < 0.1, "Translation X error: {}", t_vals[0]);
    assert!((t_vals[1] - 1.0).abs() < 0.1, "Translation Y error: {}", t_vals[1]);
    assert!((t_vals[2] - 0.0).abs() < 0.1, "Translation Z error: {}", t_vals[2]);
}

#[test]
fn test_registration_rigid_full() {
    let device = Default::default();
    let d = 20;
    let shape = [d, d, d];
    
    let make_ellipsoid = |c: [f32; 3], r: [f32; 3], theta: f32| -> Vec<f32> {
        let cos_t = theta.cos();
        let sin_t = theta.sin();
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
                    let lx = dx * cos_t + dy * sin_t;
                    let ly = -dx * sin_t + dy * cos_t;
                    let lz = dz;
                    let val = (-(lx*lx)/(2.0*r[0]*r[0]) - (ly*ly)/(2.0*r[1]*r[1]) - (lz*lz)/(2.0*r[2]*r[2])).exp();
                    data.push(val);
                }
            }
        }
        data
    };

    // Fixed: Center (10, 10, 10), Radii (2, 3, 4)
    let fixed_data = make_ellipsoid([10.0, 10.0, 10.0], [2.0, 3.0, 4.0], 0.0);
    
    // Moving: Center (10, 10, 10), Rotated 0.2 rad around Z. (No Translation)
    let moving_data = make_ellipsoid([10.0, 10.0, 10.0], [2.0, 3.0, 4.0], 0.2);

    let fixed_tensor = Tensor::<B, 3>::from_data(TensorData::new(fixed_data, shape), &device);
    let moving_tensor = Tensor::<B, 3>::from_data(TensorData::new(moving_data, shape), &device);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();
    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);

    let translation = Tensor::<B, 1>::zeros([3], &device); 
    let rotation = Tensor::<B, 1>::zeros([3], &device);
    let center = Tensor::<B, 1>::from_floats([10.0, 10.0, 10.0], &device); 
    let transform = RigidTransform::<B, 3>::new(translation, rotation, center);

    let optimizer = AdamOptimizer::new(1e-2);
    let metric = MeanSquaredError::new();
    let mut registration = Registration::new(optimizer, metric);

    // Rotation optimization might need lower LR or more steps
    let result_transform = registration.execute(&fixed, &moving, transform, 500, 1e-2).unwrap();

    let r_est = result_transform.rotation().into_data();
    let r_vals = r_est.as_slice::<f32>().unwrap();
    
    println!("Estimated Rotation (Only): {:?}", r_vals);
    
    assert!((r_vals[2] - 0.2).abs() < 0.05, "Rotation Z error: {}", r_vals[2]);
}
