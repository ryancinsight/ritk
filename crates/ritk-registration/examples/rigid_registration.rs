//! Rigid Registration Example
//!
//! This example demonstrates rigid (translation + rotation) registration of two 3D brain MRI images.
//! It uses the ANTs example dataset and shows the complete multi-resolution registration workflow:
//!
//! 1. Load fixed and moving images from NIfTI files
//! 2. Create a rigid transform with initial parameters
//! 3. Set up the Advanced Mutual Information metric
//! 4. Configure the Multi-Resolution Schedule
//! 5. Execute registration
//! 6. Save the registered result
//!
//! Usage:
//!   cargo run --example rigid_registration

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::transform::RigidTransform;
use ritk_core::filter::ResampleImageFilter;
use ritk_core::interpolation::LinearInterpolator;
use ritk_registration::metric::MutualInformation;
use ritk_registration::optimizer::AdamOptimizer;
use ritk_registration::multires::{MultiResolutionRegistration, RegistrationSchedule};
use ritk_io::{read_nifti, write_nifti};
use std::path::Path;

// Use autodiff backend for gradient-based optimization
type Backend = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    println!("RITK Rigid Registration Example");
    println!("================================\n");

    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let device = Default::default();

    // =======================================================================
    // Step 1: Load Images
    // =======================================================================
    println!("Step 1: Loading images...");

    let fixed_path = Path::new("../../test_data/ants_example/S_template.nii.gz");
    let moving_path = Path::new("../../test_data/ants_example/S_templateCerebellum.nii.gz");

    if !fixed_path.exists() {
        anyhow::bail!(
            "Fixed image not found: {}. Run: cargo xtask download-datasets ants",
            fixed_path.display()
        );
    }

    if !moving_path.exists() {
        anyhow::bail!(
            "Moving image not found: {}. Run: cargo xtask download-datasets ants",
            moving_path.display()
        );
    }

    println!("  Loading fixed image: {}", fixed_path.display());
    let fixed: Image<Backend, 3> = read_nifti(fixed_path, &device)?;
    println!("  Fixed image shape: {:?}", fixed.shape());
    println!("  Fixed image spacing: [{:.2}, {:.2}, {:.2}] mm", 
        fixed.spacing()[0], fixed.spacing()[1], fixed.spacing()[2]);

    println!("  Loading moving image: {}", moving_path.display());
    let moving: Image<Backend, 3> = read_nifti(moving_path, &device)?;
    println!("  Moving image shape: {:?}", moving.shape());
    println!("  Moving image spacing: [{:.2}, {:.2}, {:.2}] mm",
        moving.spacing()[0], moving.spacing()[1], moving.spacing()[2]);

    // =======================================================================
    // Step 2: Initialize Rigid Transform
    // =======================================================================
    println!("\nStep 2: Initializing rigid transform...");

    // Calculate image center for rotation center
    let fixed_shape = fixed.shape();
    let center = Tensor::<Backend, 1>::from_data(
        burn::tensor::TensorData::from([
            fixed_shape[0] as f32 / 2.0,
            fixed_shape[1] as f32 / 2.0,
            fixed_shape[2] as f32 / 2.0,
        ]),
        &device,
    );
    // Convert index center to physical point
    let center_point = fixed.index_to_world_tensor(center.unsqueeze_dim(0)).squeeze(0);
    
    println!("  Rotation center (physical): {}", center_point);

    let transform = RigidTransform::<Backend, 3>::identity(Some(center_point), &device);
    
    // Initialize translation (optional, if we want to start closer)
    // transform.set_translation(...);

    // =======================================================================
    // Step 3: Configure Registration
    // =======================================================================
    println!("\nStep 3: Configuring multi-resolution registration...");

    // Metric: Mutual Information
    // 32 bins, 32 samples (not used here as we use full image), sigma=1.0
    // Note: MI uses soft histogramming on full image or sampled points
    let metric = MutualInformation::<Backend>::new(32, 0.0, 255.0, 1.0);
    
    // Schedule: 3 levels (4x, 2x, 1x)
    // We use a gentle schedule for this example
    let levels = 3;
    let schedule = RegistrationSchedule::<3>::default(levels)
        .with_iterations(vec![100, 100, 50])
        .with_learning_rates(vec![1e-1, 5e-2, 1e-2]); // Higher LR for coarse levels

    println!("  Levels: {}", levels);
    println!("  Iterations: {:?}", schedule.iterations);
    println!("  Learning Rates: {:?}", schedule.learning_rates);

    // =======================================================================
    // Step 4: Execute Registration
    // =======================================================================
    println!("\nStep 4: Executing registration...");

    let registration = MultiResolutionRegistration::new(metric);
    
    let optimizer_factory = |lr| AdamOptimizer::new(lr);

    let final_transform = registration.execute(
        &fixed,
        &moving,
        transform,
        optimizer_factory,
        schedule,
    );

    println!("  Registration complete!");
    // println!("  Final parameters: {}", final_transform.params());

    // =======================================================================
    // Step 5: Resample and Export Registered ImageSet
    // =======================================================================
    println!("\nStep 5: Resampling and exporting registered imageset...");

    let resampler = ResampleImageFilter::<Backend, RigidTransform<Backend, 3>, LinearInterpolator, 3>::new_from_reference(
        &fixed,
        final_transform.clone(),
        LinearInterpolator::new(),
    ).with_default_pixel_value(0.0);
    
    // Note: ResampleImageFilter usually applies transform from Output -> Input
    // If final_transform is Fixed -> Moving, this is correct for resampling Moving to Fixed space
    let registered = resampler.apply(&moving);
    
    // Export the complete imageset for comparison
    let output_dir = Path::new("registration_output");
    std::fs::create_dir_all(output_dir)?;
    
    // Save fixed image (reference space)
    let fixed_output = output_dir.join("fixed.nii.gz");
    println!("  Exporting fixed image to: {}", fixed_output.display());
    write_nifti(&fixed_output, &fixed)?;
    
    // Save moving image (original)
    let moving_output = output_dir.join("moving.nii.gz");
    println!("  Exporting moving image to: {}", moving_output.display());
    write_nifti(&moving_output, &moving)?;
    
    // Save registered result (moving aligned to fixed)
    let registered_output = output_dir.join("registered.nii.gz");
    println!("  Exporting registered image to: {}", registered_output.display());
    write_nifti(&registered_output, &registered)?;
    
    // Also save individual result in current directory for convenience
    let result_path = Path::new("registered_rigid.nii.gz");
    println!("  Also saving result to: {}", result_path.display());
    write_nifti(result_path, &registered)?;
    
    println!("\n================================");
    println!("Registration complete!");
    println!("Output files:");
    println!("  - {}", fixed_output.display());
    println!("  - {}", moving_output.display());
    println!("  - {}", registered_output.display());
    println!("  - {}", result_path.display());

    Ok(())
}