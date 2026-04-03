//! Brain CT-MRI Full Registration Example
//!
//! Loads paired CT and T1-MRI PNG slices from the "Paired MRI (T1, T2) and CT Scans Dataset",
//! converts each to a NIfTI volume, and performs rigid + affine registration.
//!
//! Usage:
//!   cargo run --example brain_ct_mri_registration

use burn::backend::Autodiff;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use ritk_core::filter::ResampleImageFilter;
use ritk_core::image::Image;
use ritk_core::interpolation::LinearInterpolator;
use ritk_core::transform::{AffineTransform, RigidTransform};
use ritk_io::{read_png_series, write_nifti};
use ritk_registration::metric::MutualInformation;
use ritk_registration::multires::{MultiResolutionRegistration, RegistrationSchedule};
use ritk_registration::optimizer::AdamOptimizer;

// CPU backend with automatic differentiation
type Backend = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    println!("RITK Brain CT-MRI Full Registration Example");
    println!("==========================================\n");

    tracing_subscriber::fmt().with_env_filter("info").init();

    let device = Default::default();

    let data_root = std::path::Path::new("data/Paired MRI (T1, T2) and CT Scans Dataset");
    let ct_dir = data_root.join("CT/PNG/Patient_01");
    let mri_dir = data_root.join("T1-MRI/PNG/Patient_01");

    if !ct_dir.exists() {
        anyhow::bail!("CT directory not found: {}", ct_dir.display());
    }
    if !mri_dir.exists() {
        anyhow::bail!("MRI directory not found: {}", mri_dir.display());
    }

    // 1. Load PNG series as 3D volumes
    println!("Loading CT PNG series from: {}", ct_dir.display());
    let ct_image: Image<Backend, 3> = read_png_series(&ct_dir, &device)?;
    println!(
        "  CT shape: {:?}, spacing: {:?}",
        ct_image.shape(),
        ct_image.spacing()
    );

    println!("Loading T1-MRI PNG series from: {}", mri_dir.display());
    let mri_image: Image<Backend, 3> = read_png_series(&mri_dir, &device)?;
    println!(
        "  MRI shape: {:?}, spacing: {:?}",
        mri_image.shape(),
        mri_image.spacing()
    );

    // Save intermediate NIfTI files for inspection
    let output_dir = std::path::Path::new("data/output");
    std::fs::create_dir_all(output_dir)?;

    let ct_nifti_path = output_dir.join("patient01_ct.nii.gz");
    let mri_nifti_path = output_dir.join("patient01_mri_t1.nii.gz");

    write_nifti(&ct_nifti_path, &ct_image)?;
    println!("Saved CT as NIfTI: {}", ct_nifti_path.display());

    write_nifti(&mri_nifti_path, &mri_image)?;
    println!("Saved T1-MRI as NIfTI: {}", mri_nifti_path.display());

    // 2. Rigid Registration (CT fixed, MRI moving)
    println!("\n=== Rigid Registration ===");
    let center_idx = Tensor::<Backend, 1>::from_floats(
        [
            ct_image.shape()[0] as f32 / 2.0,
            ct_image.shape()[1] as f32 / 2.0,
            ct_image.shape()[2] as f32 / 2.0,
        ],
        &device,
    );
    let center_phys = ct_image
        .index_to_world_tensor(center_idx.unsqueeze_dim(0))
        .squeeze();

    let rigid_transform = RigidTransform::<Backend, 3>::identity(Some(center_phys), &device);

    let metric = MutualInformation::<Backend>::new(ritk_registration::metric::MutualInformationVariant::Standard, 32, 0.0, 255.0, 1.0);

    let rigid_schedule = RegistrationSchedule::<3>::default(3)
        .with_iterations(vec![50, 50, 25])
        .with_learning_rates(vec![0.1, 0.05, 0.01]);

    let registration = MultiResolutionRegistration::new(metric);
    let optimizer_factory = |lr| AdamOptimizer::new(lr);

    println!("Running rigid registration...");
    let rigid_result = registration.execute(
        &ct_image,
        &mri_image,
        rigid_transform,
        optimizer_factory,
        rigid_schedule,
    );
    println!("Rigid registration complete");

    // 3. Affine Registration
    println!("\n=== Affine Registration ===");
    let affine_transform = AffineTransform::<Backend, 3>::new(
        rigid_result.matrix(),
        rigid_result.translation(),
        rigid_result.center()
    );

    let metric2 = MutualInformation::<Backend>::new(ritk_registration::metric::MutualInformationVariant::Standard, 32, 0.0, 255.0, 1.0);
    let affine_schedule = RegistrationSchedule::<3>::default(3)
        .with_iterations(vec![30, 30, 15])
        .with_learning_rates(vec![0.05, 0.02, 0.005]);

    let registration2 = MultiResolutionRegistration::new(metric2);
    let optimizer_factory2 = |lr| AdamOptimizer::new(lr);

    println!("Running affine registration...");
    let affine_result = registration2.execute(
        &ct_image,
        &mri_image,
        affine_transform,
        optimizer_factory2,
        affine_schedule,
    );
    println!("Affine registration complete");

    // 4. Resample MRI onto CT grid using final transform
    println!("\nResampling registered image...");
    let resampler = ResampleImageFilter::new_from_reference(
        &ct_image,
        affine_result.clone(),
        LinearInterpolator::new(),
    )
    .with_default_pixel_value(0.0);

    let registered_mri = resampler.apply(&mri_image);

    let registered_path = output_dir.join("patient01_mri_registered.nii.gz");
    write_nifti(&registered_path, &registered_mri)?;
    println!("Saved registered MRI: {}", registered_path.display());

    // 5. Print transform parameters
    println!("\n=== Final Transform ===");
    let matrix: Vec<f32> = affine_result.matrix().into_data().to_vec().unwrap();
    println!("Matrix: {:?}", matrix);
    let translation: Vec<f32> = affine_result.translation().into_data().to_vec().unwrap();
    println!("Translation: {:?}", translation);

    println!("\nFiles saved to: {}", output_dir.display());
    println!("  - patient01_ct.nii.gz           (fixed image)");
    println!("  - patient01_mri_t1.nii.gz       (moving image, original)");
    println!("  - patient01_mri_registered.nii.gz (moving image, registered)");

    Ok(())
}
