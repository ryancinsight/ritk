use ritk_model::ssmmorph::integration::DiffeomorphicSSMMorph;
use ritk_model::ssmmorph::SSMMorphConfig;
use ritk_core::image::Image;
use ritk_core::spatial::{Point, Spacing, Direction};
use burn::tensor::{Tensor, TensorData};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;

type B = Autodiff<NdArray<f32>>;

#[test]
fn test_diffeomorphic_ssmmorph_integration() {
    // 0. Setup Backend
    let device = Default::default();
    
    // 1. Setup Config
    // Use minimal config for speed and memory efficiency during test
    let mut config = SSMMorphConfig::for_3d_registration();
    // Reduce model size
    config.encoder.num_stages = 2; 
    config.encoder.base_channels = 4;
    config.encoder.blocks_per_stage = 1;
    config.encoder.in_channels = 2; // Fixed (1) + Moving (1) = 2
    // Ensure diffeomorphic integration is enabled
    config.diffeomorphic = true;
    config.integration_steps = 5;

    // 2. Initialize Model
    let model = DiffeomorphicSSMMorph::<B>::new(&config, &device);
    
    // 3. Create Dummy Images (16x16x16 to be very fast)
    let d = 16;
    let shape = [d, d, d];
    let data_len = d * d * d;
    
    // Create sphere-like data so it's not just zeros
    let make_sphere = |center: [f32; 3], radius: f32| -> Vec<f32> {
        let mut data = Vec::with_capacity(data_len);
        for z in 0..d {
            for y in 0..d {
                for x in 0..d {
                    let dz = (z as f32) - center[2];
                    let dy = (y as f32) - center[1];
                    let dx = (x as f32) - center[0];
                    let dist_sq = dx*dx + dy*dy + dz*dz;
                    let val = (-dist_sq / (2.0 * radius * radius)).exp();
                    data.push(val);
                }
            }
        }
        data
    };

    let fixed_data = make_sphere([8.0, 8.0, 8.0], 4.0);
    let moving_data = make_sphere([9.0, 9.0, 8.0], 4.0);
    
    // Create Images
    // Shape for Image data is [D, H, W].
    // Note: Image expects Tensor<B, 3> for 3D image.
    let fixed_tensor = Tensor::<B, 3>::from_data(TensorData::new(fixed_data, shape), &device);
        
    let moving_tensor = Tensor::<B, 3>::from_data(TensorData::new(moving_data, shape), &device);
        
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let direction = Direction::identity();
    
    let fixed_image = Image::new(fixed_tensor, origin.clone(), spacing.clone(), direction.clone());
    let moving_image = Image::new(moving_tensor, origin, spacing, direction);
    
    // 4. Run Registration
    println!("Running diffeomorphic registration...");
    let result = model.register_diffeomorphic(&fixed_image, &moving_image);
    
    assert!(result.is_ok(), "Registration failed: {:?}", result.err());
    let transform = result.unwrap();
    
    // 5. Verify Output
    // Transform has displacement field
    let field = transform.field();
    let components = field.components(); // Vec<Tensor<B, 3>>
    assert_eq!(components.len(), 3, "Displacement field should have 3 components");
    
    // Check values are not all zero
    let mut max_disp = 0.0f32;
    for c in components {
        let dims = c.dims();
        assert_eq!(dims, [16, 16, 16], "Component dimensions mismatch");
        
        let data = c.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        // Explicit types to satisfy compiler
        let comp_max = slice.iter().fold(0.0f32, |a: f32, &b: &f32| a.max(b.abs()));
        max_disp = max_disp.max(comp_max);
    }
    
    println!("Max displacement: {}", max_disp);
    assert!(max_disp > 0.0, "Displacement field is all zeros!");
}
