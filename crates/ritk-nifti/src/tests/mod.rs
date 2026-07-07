use super::*;
use crate::header::{
    write_single_file_bytes, HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader,
};
use anyhow::Result;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_spatial::{Direction, Point, Spacing};
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
    let direction = Direction::identity();

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
    // Read and write both use sform as the metadata source of truth.
    assert!((l_spacing[0] - 0.5).abs() < 1e-5);
    assert!((l_spacing[2] - 2.0).abs() < 1e-5);

    Ok(())
}

#[test]
fn test_read_nifti_from_bytes_roundtrip() -> Result<()> {
    let dir = tempdir()?;
    let file_path = dir.path().join("test_bytes_roundtrip.nii");
    let device = Default::default();

    let shape = Shape::new([4, 3, 2]); // Z, Y, X
    let data = TensorData::new((0..24).map(|v| v as f32).collect::<Vec<_>>(), shape);
    let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
    let image = Image::new(
        tensor,
        Point::new([4.0, 5.0, 6.0]),
        Spacing::new([1.0, 0.7, 2.3]),
        Direction::identity(),
    );

    write_nifti(&file_path, &image)?;
    let bytes = std::fs::read(&file_path)?;
    let loaded = read_nifti_from_bytes::<TestBackend>(&bytes, &device)?;

    assert_eq!(loaded.shape(), [4, 3, 2]);
    assert!((loaded.origin()[0] - 4.0).abs() < 1e-5);
    assert!((loaded.origin()[1] - 5.0).abs() < 1e-5);
    assert!((loaded.origin()[2] - 6.0).abs() < 1e-5);
    assert!((loaded.spacing()[0] - 1.0).abs() < 1e-5);
    assert!((loaded.spacing()[1] - 0.7).abs() < 1e-5);
    assert!((loaded.spacing()[2] - 2.3).abs() < 1e-5);

    Ok(())
}

#[test]
fn read_nifti_from_bytes_accepts_int16_voxels() -> Result<()> {
    let device = Default::default();
    let header = NiftiHeader::new_3d(
        HeaderDims {
            nx: 3,
            ny: 2,
            nz: 2,
        },
        NiftiDatatype::Int16,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )?;
    let values = [
        -1024_i16, -7, 0, 1, 42, 127, 256, 511, 1024, 2047, 3072, 4095,
    ];
    let mut payload = Vec::with_capacity(values.len() * 2);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }

    let loaded =
        read_nifti_from_bytes::<TestBackend>(&write_single_file_bytes(&header, &payload), &device)?;

    assert_eq!(
        loaded.shape(),
        [2, 2, 3],
        "Int16 NIfTI reader must preserve ZYX shape"
    );
    assert_eq!(
        loaded.try_data_vec()?,
        values.map(f32::from).to_vec(),
        "Int16 NIfTI reader must sign-extend every voxel into the image scalar"
    );

    Ok(())
}

#[test]
fn test_write_nifti2_from_bytes_roundtrip() -> Result<()> {
    let dir = tempdir()?;
    let file_path = dir.path().join("test_nifti2_roundtrip.nii");
    let device = Default::default();

    let shape = Shape::new([3, 2, 4]); // Z, Y, X
    let values = (0..24).map(|v| v as f32 + 0.25).collect::<Vec<_>>();
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(values.clone(), shape), &device);
    let image = Image::new(
        tensor,
        Point::new([8.0, -2.0, 5.0]),
        Spacing::new([1.25, 0.5, 2.0]),
        Direction::identity(),
    );

    write_nifti2(&file_path, &image)?;
    let bytes = std::fs::read(&file_path)?;
    let header = NiftiHeader::parse(&bytes)?;
    assert_eq!(header.version, HeaderVersion::Two);
    assert_eq!(header.dim, [3, 4, 2, 3, 1, 1, 1, 1]);
    assert_eq!(header.vox_offset, 544);

    let loaded = read_nifti_from_bytes::<TestBackend>(&bytes, &device)?;
    assert_eq!(loaded.shape(), [3, 2, 4]);
    assert_eq!(
        loaded.try_data_vec()?,
        values,
        "NIfTI-2 Float32 image round-trip must preserve voxel values"
    );
    assert!((loaded.origin()[0] - 8.0).abs() < 1e-5);
    assert!((loaded.origin()[1] + 2.0).abs() < 1e-5);
    assert!((loaded.origin()[2] - 5.0).abs() < 1e-5);
    assert!((loaded.spacing()[0] - 1.25).abs() < 1e-5);
    assert!((loaded.spacing()[1] - 0.5).abs() < 1e-5);
    assert!((loaded.spacing()[2] - 2.0).abs() < 1e-5);

    Ok(())
}

#[test]
fn test_gzipped_nifti_roundtrip() -> Result<()> {
    let dir = tempdir()?;
    let file_path = dir.path().join("test_gzip_roundtrip.nii.gz");
    let device = Default::default();

    let shape = Shape::new([2, 2, 3]);
    let values = (0..12).map(|v| v as f32).collect::<Vec<_>>();
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(values.clone(), shape), &device);
    let image = Image::new(
        tensor,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.9, 1.1, 1.3]),
        Direction::identity(),
    );

    write_nifti(&file_path, &image)?;
    let bytes = std::fs::read(&file_path)?;
    assert_eq!(
        &bytes[..2],
        &[0x1f, 0x8b],
        "nii.gz output must carry the gzip stream signature"
    );

    let loaded = read_nifti::<TestBackend, _>(&file_path, &device)?;
    assert_eq!(loaded.shape(), [2, 2, 3]);
    let loaded_values = loaded.try_data_vec()?;
    assert_eq!(
        loaded_values, values,
        "gzip round-trip must preserve voxels"
    );

    Ok(())
}

#[test]
fn test_oblique_nifti_round_trip_preserves_affine_and_voxels() -> Result<()> {
    use std::f64::consts::FRAC_PI_6;

    let dir = tempdir()?;
    let file_path = dir.path().join("oblique_roundtrip.nii");
    let device = Default::default();

    let shape = Shape::new([2, 3, 4]); // Z, Y, X
    let values = (0..24).map(|v| v as f32).collect::<Vec<_>>();
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(values.clone(), shape), &device);

    let origin = Point::new([11.0, -7.5, 3.25]);
    let spacing = Spacing::new([2.0, 1.5, 0.75]);
    let cosine = FRAC_PI_6.cos();
    let sine = FRAC_PI_6.sin();
    let direction =
        Direction::from_row_major([cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0]);

    let image = Image::new(tensor, origin, spacing, direction);

    write_nifti(&file_path, &image)?;
    let loaded = read_nifti::<TestBackend, _>(&file_path, &device)?;

    assert_eq!(
        loaded.shape(),
        [2, 3, 4],
        "oblique round-trip must preserve shape"
    );
    for axis in 0..3 {
        assert!(
            (loaded.origin()[axis] - image.origin()[axis]).abs() < 1e-6,
            "oblique round-trip must preserve origin axis {axis}"
        );
        assert!(
            (loaded.spacing()[axis] - image.spacing()[axis]).abs() < 1e-6,
            "oblique round-trip must preserve spacing axis {axis}"
        );
    }
    for row in 0..3 {
        for col in 0..3 {
            assert!(
                (loaded.direction().0[(row, col)] - image.direction().0[(row, col)]).abs() < 1e-6,
                "oblique round-trip must preserve direction entry ({row},{col})"
            );
        }
    }

    let sample = |z: usize, y: usize, x: usize| -> f32 {
        loaded
            .data()
            .clone()
            .slice([z..z + 1, y..y + 1, x..x + 1])
            .into_data()
            .as_slice::<f32>()
            .expect("sampled tensor must be contiguous")[0]
    };
    assert_eq!(sample(0, 0, 0), 0.0, "logical voxel [0,0,0] must survive");
    assert_eq!(sample(0, 1, 2), 6.0, "logical voxel [0,1,2] must survive");
    assert_eq!(sample(1, 2, 3), 23.0, "logical voxel [1,2,3] must survive");

    let index = Point::new([1.0, 2.0, 3.0]);
    let physical = loaded.transform_continuous_index_to_physical_point(&index);
    let expected = image.transform_continuous_index_to_physical_point(&index);
    assert!(
        (physical[0] - expected[0]).abs() < 1e-6,
        "physical x must follow oblique affine"
    );
    assert!(
        (physical[1] - expected[1]).abs() < 1e-6,
        "physical y must follow oblique affine"
    );
    assert!(
        (physical[2] - expected[2]).abs() < 1e-6,
        "physical z must follow oblique affine"
    );

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
            // Public reader errors must not leak local filesystem paths.
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
                assert!(
                    msg.contains("Invalid NIfTI sizeof_hdr"),
                    "decode errors must preserve the violated header invariant: {msg}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn test_write_nifti_sets_sform_header_fields() -> Result<()> {
    let dir = tempdir()?;
    let file_path = dir.path().join("test_sform_header_fields.nii");
    let device = Default::default();

    let shape = Shape::new([2, 3, 4]); // Z, Y, X
    let data = TensorData::new((0..24).map(|v| v as f32).collect::<Vec<_>>(), shape);
    let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);

    let origin = Point::new([11.5, -7.25, 3.0]);
    let spacing = Spacing::new([0.8, 1.2, 2.5]);
    let direction = Direction::from_row_major([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);

    let image = Image::new(tensor, origin, spacing, direction);

    write_nifti(&file_path, &image)?;

    let bytes = std::fs::read(&file_path)?;
    let header = NiftiHeader::parse(&bytes)?;

    assert_eq!(header.sform_code, 1, "writer must set sform_code=1");
    assert_eq!(
        header.qform_code, 0,
        "writer must disable qform when emitting sform SSOT"
    );
    assert!(
        (header.pixdim[1] - 2.5).abs() < 1e-6,
        "pixdim[1] must store NIfTI x-spacing from internal col spacing"
    );
    assert!(
        (header.pixdim[2] - 1.2).abs() < 1e-6,
        "pixdim[2] must store y-spacing"
    );
    assert!(
        (header.pixdim[3] - 0.8).abs() < 1e-6,
        "pixdim[3] must store NIfTI z-spacing from internal depth spacing"
    );
    assert_eq!(
        header.xyzt_units, 2,
        "writer must encode spatial units as millimeters"
    );

    assert!(
        (header.srow_x[0] + 2.5).abs() < 1e-6,
        "srow_x[0] must encode RAS x spacing from internal column axis"
    );
    assert!(
        (header.srow_x[1] - 0.0).abs() < 1e-6,
        "srow_x[1] must remain zero for axial direction"
    );
    assert!(
        (header.srow_x[2] - 0.0).abs() < 1e-6,
        "srow_x[2] must remain zero for axial direction"
    );
    assert!(
        (header.srow_x[3] + 11.5).abs() < 1e-6,
        "srow_x[3] must encode RAS x origin from internal LPS"
    );

    assert!(
        (header.srow_y[0] - 0.0).abs() < 1e-6,
        "srow_y[0] must remain zero for axial direction"
    );
    assert!(
        (header.srow_y[1] + 1.2).abs() < 1e-6,
        "srow_y[1] must encode RAS y spacing from internal LPS"
    );
    assert!(
        (header.srow_y[2] - 0.0).abs() < 1e-6,
        "srow_y[2] must remain zero for axial direction"
    );
    assert!(
        (header.srow_y[3] - 7.25).abs() < 1e-6,
        "srow_y[3] must encode RAS y origin from internal LPS"
    );

    assert!(
        (header.srow_z[0] - 0.0).abs() < 1e-6,
        "srow_z[0] must remain zero for axial direction"
    );
    assert!(
        (header.srow_z[1] - 0.0).abs() < 1e-6,
        "srow_z[1] must remain zero for axial direction"
    );
    assert!(
        (header.srow_z[2] - 0.8).abs() < 1e-6,
        "srow_z[2] must encode RAS z spacing from internal depth axis"
    );
    assert!(
        (header.srow_z[3] - 3.0).abs() < 1e-6,
        "srow_z[3] must encode z origin"
    );

    Ok(())
}

#[test]
fn read_nifti_rejects_zero_sform_column() -> Result<()> {
    let dir = tempdir()?;
    let file_path = dir.path().join("zero_sform_column.nii");
    let device = Default::default();

    let header = NiftiHeader::new_3d(
        HeaderDims {
            nx: 2,
            ny: 2,
            nz: 2,
        },
        NiftiDatatype::Float32,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 0.0, 0.0],
        },
    )?;
    let data = vec![0_u8; 2 * 2 * 2 * 4];
    std::fs::write(&file_path, write_single_file_bytes(&header, &data))?;

    let err = read_nifti::<TestBackend, _>(&file_path, &device)
        .expect_err("zero sform column must be rejected");

    assert!(
        format!("{err:#}").contains("Invalid NIfTI spatial metadata"),
        "error must preserve public reader context: {err:#}"
    );

    Ok(())
}

mod tests_format_sources;
mod tests_labels;
#[path = "tests_native.rs"]
mod tests_native;
