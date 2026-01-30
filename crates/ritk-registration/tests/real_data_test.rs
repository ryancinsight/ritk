//! Integration tests using real medical imaging data.
//! 
//! These tests are skipped by default. To run them:
//! 1. Download test data: `cargo xtask download-datasets`
//! 2. Run tests: `cargo test --test real_data_test -- --ignored`

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
// use ritk_core::image::Image;
use ritk_core::spatial::Point3; // Used in test
use ritk_core::transform::{Transform, TranslationTransform};
use ritk_io::read_nifti;
use std::path::Path;

type B = NdArray<f32>;

fn get_test_data_dir() -> Option<std::path::PathBuf> {
    // Try to find test_data directory
    let paths = [
        Path::new("test_data"),
        Path::new("../test_data"),
        Path::new("../../test_data"),
    ];
    
    for path in &paths {
        if path.exists() {
            return Some(path.to_path_buf());
        }
    }
    
    None
}

#[test]
#[ignore = "requires test data"]
fn test_nifti_io_real_data() {
    let data_dir = get_test_data_dir()
        .expect("Test data directory not found. Run: cargo xtask download-datasets");
    
    let device = Default::default();
    
    // Find any NIfTI file in test_data
    let nifti_files: Vec<_> = walkdir::WalkDir::new(&data_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            path.extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "nii" || e == "gz")
                .unwrap_or(false)
        })
        .take(1)
        .collect();
    
    if nifti_files.is_empty() {
        panic!("No NIfTI files found in {}", data_dir.display());
    }
    
    let file_path = nifti_files[0].path();
    println!("Testing NIfTI I/O with: {}", file_path.display());
    
    // Test reading
    let image = read_nifti::<B, _>(file_path, &device)
        .expect("Failed to read NIfTI file");
    
    // Verify image properties
    let shape = image.shape();
    println!("Image shape: {:?}", shape);
    assert_eq!(shape.len(), 3, "Expected 3D image");
    
    // Verify spacing is reasonable
    let spacing = image.spacing();
    for i in 0..3 {
        assert!(spacing[i] > 0.0, "Spacing must be positive");
        assert!(spacing[i] < 100.0, "Spacing seems unreasonably large");
    }
    
    // Verify direction is orthogonal
    let direction = image.direction();
    assert!(direction.is_orthogonal(), "Direction matrix should be orthogonal");
    
    println!("NIfTI I/O test passed!");
}

#[test]
#[ignore = "requires test data"]
fn test_image_metadata_consistency() {
    let data_dir = get_test_data_dir()
        .expect("Test data directory not found. Run: cargo xtask download-datasets");
    
    let device = Default::default();
    
    // Find NIfTI files
    let nifti_files: Vec<_> = walkdir::WalkDir::new(&data_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            path.extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "nii" || e == "gz")
                .unwrap_or(false)
        })
        .take(3)
        .collect();
    
    if nifti_files.len() < 2 {
        panic!("Need at least 2 NIfTI files for this test");
    }
    
    for file_entry in &nifti_files {
        let file_path = file_entry.path();
        println!("Testing metadata consistency for: {}", file_path.display());
        
        let image = read_nifti::<B, _>(file_path, &device)
            .expect("Failed to read NIfTI file");
        
        // Test that physical space transformations are consistent
        let shape = image.shape();
        
        // Get a voxel in the middle of the image
        let idx = [
            (shape[0] / 2) as f64,
            (shape[1] / 2) as f64,
            (shape[2] / 2) as f64,
        ];
        let index_point = Point3::new(idx);
        
        // Convert index to physical and back
        let physical = image.transform_continuous_index_to_physical_point(&index_point);
        let back_to_index = image.transform_physical_point_to_continuous_index(&physical);
        
        // Should be approximately equal
        for i in 0..3 {
            let diff = (index_point[i] - back_to_index[i]).abs();
            assert!(
                diff < 1e-6,
                "Round-trip transformation failed: diff={} at dimension {}",
                diff, i
            );
        }
        
        println!("  Round-trip transformation passed");
    }
    
    println!("Metadata consistency test passed!");
}

#[test]
#[ignore = "requires test data"]
fn test_image_resampling() {
    use ritk_core::interpolation::{LinearInterpolator, Interpolator};
    use ritk_core::transform::TranslationTransform;
    
    let data_dir = get_test_data_dir()
        .expect("Test data directory not found. Run: cargo xtask download-datasets");
    
    let device = Default::default();
    
    // Find a NIfTI file
    let nifti_files: Vec<_> = walkdir::WalkDir::new(&data_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            path.extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "nii" || e == "gz")
                .unwrap_or(false)
        })
        .take(1)
        .collect();
    
    if nifti_files.is_empty() {
        panic!("No NIfTI files found");
    }
    
    let file_path = nifti_files[0].path();
    println!("Testing resampling with: {}", file_path.display());
    
    let image = read_nifti::<B, _>(file_path, &device)
        .expect("Failed to read NIfTI file");
    
    // Create a simple translation transform
    let translation = Tensor::<B, 1>::from_floats([0.0, 0.0, 0.0], &device);
    let transform = TranslationTransform::<B, 3>::new(translation);
    
    // Create interpolator
    let interpolator = LinearInterpolator::new();
    
    // Test that we can sample at some points
    let shape = image.shape();
    let device = image.data().device();
    
    // Sample a few points in the center
    let sample_points = Tensor::<B, 2>::from_floats(
        [
            [shape[0] as f32 / 2.0, shape[1] as f32 / 2.0, shape[2] as f32 / 2.0],
            [0.0, 0.0, 0.0],
            [(shape[0] - 1) as f32, (shape[1] - 1) as f32, (shape[2] - 1) as f32],
        ],
        &device,
    );
    
    let values = interpolator.interpolate(image.data(), sample_points);
    let values_data = values.to_data();
    let values_slice = values_data.as_slice::<f32>().unwrap();
    
    println!("Sampled values: {:?}", values_slice);
    
    // Values should be finite
    for (i, &val) in values_slice.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Sampled value at index {} is not finite: {}",
            i, val
        );
    }
    
    println!("Resampling test passed!");
}
