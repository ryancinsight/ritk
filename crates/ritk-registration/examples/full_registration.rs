use burn::backend::Autodiff;
use burn::tensor::Tensor;
use burn_ndarray::{NdArray, NdArrayDevice};
use ritk_core::filter::resample::ResampleImageFilter;
use ritk_core::image::Image;
use ritk_core::interpolation::NearestNeighborInterpolator;
use ritk_core::spatial::{Direction3, Point3, Spacing3};
use ritk_core::transform::translation::TranslationTransform;
use ritk_registration::metric::MeanSquaredError;
use ritk_registration::multires::{MultiResolutionRegistration, RegistrationSchedule};
use ritk_registration::optimizer::AdamOptimizer;
use std::time::Instant;

type B = Autodiff<NdArray<f32>>;

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

                let dist_sq =
                    (px - center[0]).powi(2) + (py - center[1]).powi(2) + (pz - center[2]).powi(2);

                if dist_sq <= radius.powi(2) {
                    data[idx] = 1.0;
                }
            }
        }
    }

    let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape(size);

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

    let initial_transform = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &device));

    // 4. Setup Registration Components
    // For binary shapes like simple spheres, Mean Squared Error is perfectly
    // convex and stable, avoiding the sparse histogram divergence of Mutual Information.
    let metric = MeanSquaredError::new();

    let multires = MultiResolutionRegistration::new(metric);

    // Create a 3-level schedule (shrink by 4 -> 2 -> 1)
    let schedule = RegistrationSchedule::default(3)
        .with_iterations(vec![60, 30, 10])
        .with_learning_rates(vec![1e-1, 5e-2, 1e-2]);

    // 5. Run Multi-Resolution Registration
    let start = Instant::now();
    let final_transform = multires.execute(
        &fixed_image,
        &moving_image,
        initial_transform,
        |lr| AdamOptimizer::new(lr),
        schedule,
    );
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

    // 7. Dump slices for visualization
    println!("Resampling final moving image onto fixed grid...");
    let interpolator = NearestNeighborInterpolator::new();
    let resampler = ResampleImageFilter::new_from_reference(
        &fixed_image,
        final_transform.clone(),
        interpolator,
    );
    let moved_image = resampler.apply(&moving_image);

    println!("Extracting middle slices (z=32)...");
    let fixed_slice = fixed_image
        .data()
        .clone()
        .slice([32..33, 0..64, 0..64])
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let moving_slice = moving_image
        .data()
        .clone()
        .slice([32..33, 0..64, 0..64])
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let moved_slice = moved_image
        .data()
        .clone()
        .slice([32..33, 0..64, 0..64])
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    let write_raw = |name: &str, data: &[f32]| {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        std::fs::write(name, bytes).unwrap();
        println!("Wrote {}", name);
    };

    write_raw("fixed_slice.raw", &fixed_slice);
    write_raw("moving_slice.raw", &moving_slice);
    write_raw("moved_slice.raw", &moved_slice);

    Ok(())
}
