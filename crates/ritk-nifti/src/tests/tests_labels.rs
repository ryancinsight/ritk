//! NIfTI label I/O tests.

use super::*;

/// ZYX flat-label round-trip: write then read, verify every voxel is preserved.
///
/// Shape: 2×3×4 (nz=2, ny=3, nx=4) = 24 voxels.
/// Labels are the 1-based flat-ZYX index so that each voxel carries a distinct
/// analytically derivable value: label[z*ny*nx + y*nx + x] = z*ny*nx + y*nx + x + 1.
#[test]
fn write_nifti_labels_round_trip_preserves_all_voxels() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("labels.nii");

    let [nz, ny, nx]: [usize; 3] = [2, 3, 4];
    let n = nz * ny * nx;

    // Analytical labels: 1-based flat ZYX index.
    let labels: Vec<u32> = (1..=n as u32).collect();

    let origin = [10.0_f32, 20.0, 30.0];
    let spacing = [1.5_f32, 1.0, 0.8];

    // Axial LPS direction: columns are depth, row, column.
    let direction: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];

    write_nifti_labels(&path, &labels, [nz, ny, nx], origin, spacing, direction)?;

    let (read_labels, read_shape) = read_nifti_labels(&path)?;

    assert_eq!(read_shape, [nz, ny, nx], "shape must survive round-trip");
    assert_eq!(
        read_labels.len(),
        n,
        "label vector length must equal nz*ny*nx"
    );

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let flat = z * ny * nx + y * nx + x;
                assert_eq!(
                    read_labels[flat], labels[flat],
                    "label at ({z},{y},{x}) must survive round-trip"
                );
            }
        }
    }

    Ok(())
}

/// Background-only label map (all zeros) round-trips correctly.
#[test]
fn write_nifti_labels_all_background_round_trips() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("bg_labels.nii");

    let [nz, ny, nx]: [usize; 3] = [3, 3, 3];
    let labels = vec![0u32; nz * ny * nx];

    let direction: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];

    write_nifti_labels(&path, &labels, [nz, ny, nx], [0.0; 3], [1.0; 3], direction)?;

    let (read_labels, read_shape) = read_nifti_labels(&path)?;

    assert_eq!(read_shape, [nz, ny, nx]);
    assert!(
        read_labels.iter().all(|&v| v == 0),
        "all-zero label map must remain all-zero"
    );

    Ok(())
}

/// Length mismatch between labels slice and shape product returns Err.
#[test]
fn write_nifti_labels_length_mismatch_returns_err() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad.nii");

    let direction: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    // Provide 1 label but declare shape 2×2×2 = 8.
    let result = write_nifti_labels(&path, &[42u32], [2, 2, 2], [0.0; 3], [1.0; 3], direction);

    assert!(result.is_err(), "length mismatch must return Err");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("labels.len()"),
        "error must mention labels.len(): {msg}"
    );
}

/// Single-voxel label survives write-read at the exact analytical value 7.
#[test]
fn write_nifti_labels_single_voxel_label_7_round_trips() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("single.nii");

    let direction: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    write_nifti_labels(&path, &[7u32], [1, 1, 1], [0.0; 3], [1.0; 3], direction)?;

    let (read_labels, read_shape) = read_nifti_labels(&path)?;

    assert_eq!(read_shape, [1, 1, 1]);
    assert_eq!(read_labels, vec![7u32], "label 7 must survive write-read");

    Ok(())
}

/// Sform affine written by write_nifti_labels encodes internal LPS metadata
/// as RAS rows in the NIfTI header. We verify by reading the raw header.
#[test]
fn write_nifti_labels_sform_encodes_origin_and_spacing() -> Result<()> {
    use nifti::{NiftiObject, ReaderOptions};

    let dir = tempdir()?;
    let path = dir.path().join("spatial.nii");

    let direction: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];

    write_nifti_labels(
        &path,
        &[0u32],
        [1, 1, 1],
        [5.0, 6.0, 7.0], // origin [x, y, z] in internal LPS frame
        [2.0, 3.0, 4.0], // spacing [dz, dy, dx]
        direction,
    )?;

    let obj = ReaderOptions::new().read_file(&path)?;
    let header = obj.header();

    // With axial LPS direction and spacing [dz, dy, dx] = [2,3,4],
    // NIfTI file columns [x,y,z] receive internal [col,row,depth].

    assert!(
        (header.srow_x[0] + 4.0).abs() < 1e-5,
        "srow_x[0]={}",
        header.srow_x[0]
    );
    assert!(
        (header.srow_x[3] + 5.0).abs() < 1e-5,
        "srow_x[3]={}",
        header.srow_x[3]
    );
    assert!(
        (header.srow_y[1] + 3.0).abs() < 1e-5,
        "srow_y[1]={}",
        header.srow_y[1]
    );
    assert!(
        (header.srow_y[3] + 6.0).abs() < 1e-5,
        "srow_y[3]={}",
        header.srow_y[3]
    );
    assert!(
        (header.srow_z[2] - 2.0).abs() < 1e-5,
        "srow_z[2]={}",
        header.srow_z[2]
    );
    assert!(
        (header.srow_z[3] - 7.0).abs() < 1e-5,
        "srow_z[3]={}",
        header.srow_z[3]
    );

    Ok(())
}
