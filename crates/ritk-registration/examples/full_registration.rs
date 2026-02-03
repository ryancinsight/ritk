use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_ndarray::{NdArray, NdArrayDevice};
use ritk_core::image::Image;
use ritk_core::spatial::{Point3, Spacing3, Direction3};
use ritk_core::transform::AffineTransform;
use ritk_core::transform::Transform;
use ritk_registration::registration::{Registration, RegistrationConfig};
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::optimizer::AdamOptimizer;
use ritk_core::filter::resample::ResampleImageFilter;
use ritk_core::interpolation::LinearInterpolator;
use std::time::Instant;

type B = NdArray<f32>;

fn create_sphere_image(
    size: [usize; 3],
    origin: Point3,
    spacing: Spacing3,
    center: Point3,
    radius: f64,
) -> Image<B, 3> {
    let device = NdArrayDevice::Cpu;
    let mut data = vec![0.0f32; size[0] * size[1] * size[2]];
    
    for z in 0..size[0] {
        for y in 0..size[1] {
            for x in 0..size[2] {
                let idx = z * size[1] * size[2] + y * size[2] + x;
                
                // Physical coordinate
                let px = origin[0] + (x as f64) * spacing[0];
                let py = origin[1] + (y as f64) * spacing[1];
                let pz = origin[2] + (z as f64) * spacing[2];
                
                let dist_sq = (px - center[0]).powi(2) + (py - center[1]).powi(2) + (pz - center[2]).powi(2);
                
                if dist_sq <= radius.powi(2) {
                    data[idx] = 1.0;
                }
            }
        }
    }
    
    let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
        .reshape(size);
        
    Image::new(tensor, origin, spacing, Direction3::identity())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting full registration example...");
    let device = NdArrayDevice::Cpu;

    // 1. Create Fixed Image (Sphere at center)
    let size = [64, 64, 64];
    let origin = Point3::new([0.0, 0.0, 0.0]);
    let spacing = Spacing3::new([1.0, 1.0, 1.0]);
    let center_fixed = Point3::new([32.0, 32.0, 32.0]);
    let radius = 15.0;
    
    let fixed_image = create_sphere_image(size, origin, spacing, center_fixed, radius);
    println!("Created fixed image with sphere at {:?}", center_fixed);

    // 2. Create Moving Image (Sphere shifted by 5.0 in X)
    // We simulate this by creating a sphere at 32-5 = 27
    let center_moving = Point3::new([27.0, 32.0, 32.0]);
    let moving_image = create_sphere_image(size, origin, spacing, center_moving, radius);
    println!("Created moving image with sphere at {:?}", center_moving);

    // 3. Initialize Transform (Identity)
    // We expect the registration to find translation = [5.0, 0.0, 0.0]
    // Because Fixed(x) ~ Moving(T(x)). 
    // If Moving is at 27, and Fixed is at 32.
    // T(32) should be 27.
    // T(x) = x + t. 32 + t = 27 => t = -5.
    // Wait, let's verify definition.
    // Metric(Fixed, Moving(Transform(x)))
    // Transform maps from Fixed Space -> Moving Space.
    // Point in Fixed is 32. Point in Moving is 27.
    // Transform(32) -> 27.
    // 32 + t = 27 => t = -5.
    // So we expect translation around [-5, 0, 0].

    let initial_transform = AffineTransform::<B, 3>::identity(None, &device);
    
    // 4. Setup Registration Components
    let metric = MeanSquaredError::new();
    let optimizer = AdamOptimizer::new(1e-1); // High LR for fast convergence on simple problem
    
    let config = RegistrationConfig::new()
        .with_gradient_clipping(1.0)
        .with_log_interval(10)
        .with_convergence_detection(ritk_registration::validation::ConvergenceChecker::new(
             20, // window size
             1e-5 // threshold
        ));

    let mut registration = Registration::with_config(optimizer, metric, config);

    // 5. Run Registration
    let start = Instant::now();
    let final_transform = registration.execute(
        &fixed_image, 
        &moving_image, 
        initial_transform, 
        200, // iterations
        0.1  // learning rate
    )?;
    let duration = start.elapsed();
    
    println!("Registration completed in {:.2?}", duration);
    
    // 6. Analyze Results
    let translation = final_transform.translation().into_data();
    let translation_vec: Vec<f32> = translation.to_vec().unwrap();
    println!("Final translation: {:?}", translation_vec);
    
    let expected_translation = [-5.0, 0.0, 0.0];
    println!("Expected translation: {:?}", expected_translation);
    
    let error_x = (translation_vec[0] - expected_translation[0]).abs();
    if error_x < 0.5 {
        println!("SUCCESS: Translation recovered within tolerance!");
    } else {
        println!("FAILURE: Translation error too high.");
    }

    Ok(())
}
