//! Demo Registration Example
//!
//! This example demonstrates rigid registration using the prepared test data.
//! It aligns 'brain_moving.nii.gz' (Subject) to 'brain_fixed.nii.gz' (Template).
//!
//! Usage:
//!   cargo run --example demo_registration

use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, AutoGraphicsApi};
use burn::tensor::Tensor;
use ritk_core::image::Image;
use ritk_core::transform::RigidTransform;
use ritk_core::filter::ResampleImageFilter;
use ritk_core::interpolation::LinearInterpolator;
use ritk_registration::metric::MutualInformation;
use ritk_registration::optimizer::AdamOptimizer;
use ritk_registration::multires::{MultiResolutionRegistration, RegistrationSchedule};
use ritk_io::{read_nifti, write_nifti};
use std::path::Path;

// Use WGPU backend for GPU acceleration
type Backend = Autodiff<Wgpu>;

fn main() -> anyhow::Result<()> {
    println!("RITK Demo Registration (GPU Backend)");
    println!("====================================\n");

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let device = Default::default();

    // 1. Load Images
    let data_dir = Path::new("test_data/registration");
    let fixed_path = data_dir.join("brain_fixed.nii.gz");
    let moving_path = data_dir.join("brain_moving.nii.gz");

    if !fixed_path.exists() || !moving_path.exists() {
        anyhow::bail!(
            "Test data not found in {}. Run: cargo xtask prepare-registration-data",
            data_dir.display()
        );
    }

    println!("Loading fixed image: {}", fixed_path.display());

    // Debug: Check file signature
    let mut file = std::fs::File::open(&fixed_path)?;
    let mut buffer = [0u8; 500];
    use std::io::Read;
    file.read_exact(&mut buffer)?;
    println!("File signature: {:02X?}", &buffer[0..4]);
    println!("File content (string): {}", String::from_utf8_lossy(&buffer));

    let fixed: Image<Backend, 3> = read_nifti(&fixed_path, &device)?;
    println!("  Size: {:?}, Spacing: {:?}", fixed.shape(), fixed.spacing());
    let fixed_data = fixed.data();
    println!("  Fixed Range: [{:.2}, {:.2}]", 
        fixed_data.clone().min().into_scalar(), 
        fixed_data.clone().max().into_scalar()
    );

    println!("Loading moving image: {}", moving_path.display());
    let moving: Image<Backend, 3> = read_nifti(&moving_path, &device)?;
    println!("  Size: {:?}, Spacing: {:?}", moving.shape(), moving.spacing());
    let moving_data = moving.data();
    println!("  Moving Range: [{:.2}, {:.2}]", 
        moving_data.clone().min().into_scalar(), 
        moving_data.clone().max().into_scalar()
    );

    // 2. Initialize Transform
    // Center of rotation at fixed image center
    let center_idx = Tensor::<Backend, 1>::from_floats([
        fixed.shape()[0] as f32 / 2.0,
        fixed.shape()[1] as f32 / 2.0,
        fixed.shape()[2] as f32 / 2.0,
    ], &device);
    let center_phys = fixed.index_to_world_tensor(center_idx.unsqueeze_dim(0)).squeeze(0);
    
    let transform = RigidTransform::<Backend, 3>::identity(Some(center_phys), &device);
    
    // 3. Configure Registration
    // Mutual Information with 32 bins
    // Set max intensity to cover the image range (max is ~91.5)
    let metric = MutualInformation::<Backend>::new(32, 0.0, 100.0, 1.0);

    // Schedule: 3 levels (4x, 2x, 1x)
    let schedule = RegistrationSchedule::<3>::default(3)
        .with_iterations(vec![100, 100, 50])
        .with_learning_rates(vec![0.1, 0.05, 0.01]);

    // 4. Execute
    let registration = MultiResolutionRegistration::new(metric);
    let optimizer_factory = |lr| AdamOptimizer::new(lr);

    println!("\nStarting registration...");
    let final_transform = registration.execute(
        &fixed,
        &moving,
        transform,
        optimizer_factory,
        schedule,
    );

    // 5. Resample and Save
    println!("\nResampling result...");
    let resampler = ResampleImageFilter::new_from_reference(
        &fixed,
        final_transform.clone(),
        LinearInterpolator::new(),
    ).with_default_pixel_value(0.0);

    let registered = resampler.apply(&moving);

    let output_dir = Path::new("demo_output");
    std::fs::create_dir_all(output_dir)?;
    let output_path = output_dir.join("registered.nii.gz");
    
    write_nifti(&output_path, &registered)?;
    println!("Saved registered image to: {}", output_path.display());

    Ok(())
}
