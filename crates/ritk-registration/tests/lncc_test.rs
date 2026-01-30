use burn::tensor::{Tensor, TensorData};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::TranslationTransform;
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_registration::metric::LocalNormalizedCrossCorrelation;
use ritk_registration::registration::Registration;
use ritk_registration::optimizer::AdamOptimizer;

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_lncc_registration_translation() {
    let device = Default::default();

    // 1. Create 3D images (20x20x20)
    let d = 20;
    let shape = [d, d, d];
    
    // Simple Gaussian blob
    let make_blob = |c: [f32; 3], s: f32| -> Vec<f32> {
        let mut data = Vec::with_capacity(d*d*d);
        for z in 0..d {
            for y in 0..d {
                for x in 0..d {
                    let dx = x as f32 - c[0];
                    let dy = y as f32 - c[1];
                    let dz = z as f32 - c[2];
                    let dist2 = dx*dx + dy*dy + dz*dz;
                    data.push((-dist2 / (2.0 * s * s)).exp());
                }
            }
        }
        data
    };

    // Fixed: Center (10, 10, 10)
    let fixed_data = make_blob([10.0, 10.0, 10.0], 3.0);
    // Moving: Center (12, 12, 10) -> Translation (+2, +2, 0)
    let moving_data = make_blob([12.0, 12.0, 10.0], 3.0);

    let fixed_tensor = Tensor::<B, 3>::from_data(TensorData::new(fixed_data, shape), &device);
    let moving_tensor = Tensor::<B, 3>::from_data(TensorData::new(moving_data, shape), &device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);

    // 2. Initialize Transform
    let transform = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &device));

    // 3. Setup LNCC
    // Kernel sigma = 2.0 (window ~ 6-7 pixels)
    let metric = LocalNormalizedCrossCorrelation::new(2.0);
    
    // 4. Setup Optimizer
    let optimizer = AdamOptimizer::new(0.1);
    
    let mut registration = Registration::new(optimizer, metric);

    // 5. Execute
    let result = registration.execute(&fixed, &moving, transform, 100, 0.1);

    // 6. Verify
    let t_est = result.translation().into_data();
    let t_vals = t_est.as_slice::<f32>().unwrap();
    println!("Estimated Translation: {:?}", t_vals);

    assert!((t_vals[0] - 2.0).abs() < 0.2, "X error: {}", t_vals[0]);
    assert!((t_vals[1] - 2.0).abs() < 0.2, "Y error: {}", t_vals[1]);
    assert!((t_vals[2] - 0.0).abs() < 0.2, "Z error: {}", t_vals[2]);
}
