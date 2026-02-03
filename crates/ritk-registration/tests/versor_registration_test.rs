use burn::tensor::{Tensor, TensorData};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::VersorRigid3DTransform;
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::registration::Registration;
use ritk_registration::optimizer::AdamOptimizer;

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_registration_versor_3d_rotation() {
    let device = Default::default();
    let d = 20;
    let shape = [d, d, d];
    
    // Ellipsoid generator
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
                    // Rotate around Z
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
    
    // Moving: Center (10, 10, 10), Rotated 0.2 rad around Z.
    // We want to recover this rotation.
    // If Moving is rotated by +0.2, then T(x) should map Fixed coords to Moving coords.
    // T(x) = R * (x - c) + c
    // If we rotate Fixed by +0.2, it matches Moving.
    let moving_data = make_ellipsoid([10.0, 10.0, 10.0], [2.0, 3.0, 4.0], 0.2);

    let fixed_tensor = Tensor::<B, 3>::from_data(TensorData::new(fixed_data, shape), &device);
    let moving_tensor = Tensor::<B, 3>::from_data(TensorData::new(moving_data, shape), &device);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();
    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);

    let translation = Tensor::<B, 1>::zeros([3], &device); 
    // Initial Rotation: Identity [0, 0, 0, 1] (x, y, z, w)
    let rotation = Tensor::<B, 1>::from_floats([0.0, 0.0, 0.0, 1.0], &device);
    let center = Tensor::<B, 1>::from_floats([10.0, 10.0, 10.0], &device); 
    let transform = VersorRigid3DTransform::<B>::new(translation, rotation, center);

    let optimizer = AdamOptimizer::new(1e-2);
    let metric = MeanSquaredError::new();
    let mut registration = Registration::new(optimizer, metric);

    // Run registration
    let result_transform = registration.execute(&fixed, &moving, transform, 500, 1e-2).unwrap();

    let r_est = result_transform.rotation().into_data();
    let r_vals = r_est.as_slice::<f32>().unwrap();
    
    // Normalize result for comparison
    let norm = (r_vals[0]*r_vals[0] + r_vals[1]*r_vals[1] + r_vals[2]*r_vals[2] + r_vals[3]*r_vals[3]).sqrt();
    let r_norm = [r_vals[0]/norm, r_vals[1]/norm, r_vals[2]/norm, r_vals[3]/norm];
    
    println!("Estimated Rotation (Quaternion): {:?}", r_norm);
    
    // Expected: Rotation around Z by 0.2 rad
    // q = [0, 0, sin(0.1), cos(0.1)] = [0, 0, 0.0998, 0.995]
    // Or negative: [0, 0, -0.0998, -0.995] (equivalent rotation)
    
    let expected_z = (0.2_f32 / 2.0).sin();
    let expected_w = (0.2_f32 / 2.0).cos();
    
    // Check if it matches positive or negative quaternion
    let dist_pos = (r_norm[0]-0.0).powi(2) + (r_norm[1]-0.0).powi(2) + (r_norm[2]-expected_z).powi(2) + (r_norm[3]-expected_w).powi(2);
    let dist_neg = (r_norm[0]+0.0).powi(2) + (r_norm[1]+0.0).powi(2) + (r_norm[2]+expected_z).powi(2) + (r_norm[3]+expected_w).powi(2);
    
    let dist = dist_pos.min(dist_neg);
    
    assert!(dist < 0.01, "Quaternion error too high: dist={}", dist);
}
