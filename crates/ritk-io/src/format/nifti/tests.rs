use super::*;
use anyhow::Result;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use nalgebra::SMatrix;
use ritk_core::spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

type TestBackend = NdArray<f32>;

#[test]
fn test_read_write_nifti_cycle() -> Result<()> {
    let dir = tempdir()?;
    let file_path = dir.path().join("test_cycle.nii");
    let device = Default::default();

    // Create a synthetic Image
    let shape = Shape::new([5, 4, 3]); // Z, Y, X
    let data = TensorData::new(vec![0.0; 3 * 4 * 5], shape);
    let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);

    let origin = Point::new([10.0, 20.0, 30.0]);
    let spacing = Spacing::new([0.5, 0.5, 2.0]);
    // Simple identity direction
    let direction = Direction(SMatrix::identity());

    let image = Image::new(tensor, origin, spacing, direction);

    // Write
    write_nifti(&file_path, &image)?;

    // Read back
    let loaded = read_nifti::<TestBackend, _>(&file_path, &device)?;

    // Verify Metadata
    let l_origin = loaded.origin();
    let l_spacing = loaded.spacing();

    // Verify sform preservation (which implies spacing/origin/direction are correct)
    assert!((l_origin[0] - 10.0).abs() < 1e-5);
    assert!((l_origin[1] - 20.0).abs() < 1e-5);
    assert!((l_origin[2] - 30.0).abs() < 1e-5);

    // Note: if pixdim is not set, read_nifti might fallback to pixdim if sform_code=0, but we set sform_code=1.
    // So read_nifti should use sform to derive spacing.
    // Columns of sform are [0.5, 0, 0], [0, 0.5, 0], [0, 0, 2.0].
    // Norms are 0.5, 0.5, 2.0. Correct.
    assert!((l_spacing[0] - 0.5).abs() < 1e-5);
    assert!((l_spacing[2] - 2.0).abs() < 1e-5);

    Ok(())
}

#[test]
fn test_read_nifti_error_leak() {
    let path = "/sensitive/path/that/should/not/be/in/error/message.nii";
    let device = Default::default();
    let result = read_nifti::<TestBackend, _>(path, &device);

    match result {
        Ok(_) => panic!("Should fail"),
        Err(e) => {
            let msg = format!("{:?}", e);
            // The nifti crate might include the path in the error message if the file doesn't exist.
            // We want to ensure it DOES NOT leak the path.
            if msg.contains(path) {
                println!(
                    "Vulnerability confirmed: Path leaked in error message: {}",
                    msg
                );
                panic!("Path leaked in error message: {}", msg);
            } else {
                println!("Path NOT leaked in error message: {}", msg);
                assert!(msg.contains("Failed to read NIfTI file"));
                // Check NO underlying cause
                if msg.contains("Caused by") {
                    panic!("Underlying error leaked: {}", msg);
                }
            }
        }
    }
}

#[test]
fn test_read_nifti_invalid_file_error_leak() -> Result<()> {
    use std::io::Write;
    let dir = tempdir()?;
    let file_path = dir.path().join("invalid.nii");
    {
        let mut f = std::fs::File::create(&file_path)?;
        f.write_all(b"NOT A NIFTI FILE")?;
    }

    let path_str = file_path.to_string_lossy().to_string();
    let device = Default::default();
    let result = read_nifti::<TestBackend, _>(&file_path, &device);

    match result {
        Ok(_) => panic!("Should fail"),
        Err(e) => {
            let msg = format!("{:?}", e);
            if msg.contains(&path_str) {
                println!(
                    "Vulnerability confirmed: Path leaked in error message: {}",
                    msg
                );
                panic!("Path leaked in error message: {}", msg);
            } else {
                println!("Path NOT leaked in error message: {}", msg);
                assert!(msg.contains("Failed to read NIfTI file"));
                if msg.contains("Caused by") {
                    panic!("Underlying error leaked: {}", msg);
                }
            }
        }
    }
    Ok(())
}
