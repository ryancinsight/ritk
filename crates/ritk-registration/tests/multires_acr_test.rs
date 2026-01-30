use burn::tensor::{Tensor, TensorData};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::TranslationTransform;
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_registration::metric::{AdvancedCorrelationRatio, CorrelationDirection};
use ritk_registration::optimizer::AdamOptimizer;
use ritk_registration::multires::{MultiResolutionRegistration, RegistrationSchedule};

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_multires_acr_registration() {
    let device = Default::default();

    // 1. Create larger 3D images (40x40x40)
    let d = 40;
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

    // Fixed: Center (20, 20, 20)
    let fixed_data = make_blob([20.0, 20.0, 20.0], 5.0);
    // Moving: Center (24, 24, 20) -> Translation is (+4, +4, 0)
    // Transform maps Fixed(x) -> Moving(T(x))
    // T(x) = x + t
    // Moving(x+t) ~ Fixed(x)
    // Peak at x=20 matches Moving peak at 24.
    // 20 + t = 24 => t = 4.
    let moving_data = make_blob([24.0, 24.0, 20.0], 5.0);

    let fixed_tensor = Tensor::<B, 3>::from_data(TensorData::new(fixed_data, shape), &device);
    let moving_tensor = Tensor::<B, 3>::from_data(TensorData::new(moving_data, shape), &device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    let fixed = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving = Image::new(moving_tensor, origin, spacing, direction);

    // 2. Initialize Transform
    let transform = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &device));

    // 3. Setup MultiRes with ACR
    // Note: Image values are 0.0 to 1.0
    let metric = AdvancedCorrelationRatio::new(
        32, // bins
        0.0, // min
        1.0, // max
        CorrelationDirection::MovingGivenFixed
    );
    let multires = MultiResolutionRegistration::new(metric);

    // Schedule: 3 levels (Shrink 4, 2, 1)
    let mut schedule = RegistrationSchedule::<3>::default(3);
    
    // Customize iterations and learning rates
    schedule.iterations = vec![100, 100, 100]; 
    schedule.learning_rates = vec![0.5, 0.2, 0.1]; 

    // 4. Execute
    let result = multires.execute(
        &fixed,
        &moving,
        transform,
        |lr| AdamOptimizer::new(lr),
        schedule,
    );

    // 5. Verify
    let t_est = result.translation().into_data();
    let t_vals = t_est.as_slice::<f32>().unwrap();
    println!("Estimated Translation: {:?}", t_vals);

    assert!((t_vals[0] - 4.0).abs() < 0.5, "X error: {}", t_vals[0]);
    assert!((t_vals[1] - 4.0).abs() < 0.5, "Y error: {}", t_vals[1]);
    assert!((t_vals[2] - 0.0).abs() < 0.5, "Z error: {}", t_vals[2]);
}
