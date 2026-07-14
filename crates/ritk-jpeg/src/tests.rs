//! JPEG tests migrated to the Atlas-native (Coeus) path — ADR 0002.

use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use crate::{read_jpeg, write_jpeg, JpegReader, JpegWriter};

type TestBackend = SequentialBackend;

fn image_from_values(shape: [usize; 3], values: Vec<f32>) -> Image<f32, TestBackend, 3> {
    Image::from_flat_on(
        values,
        shape,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("valid image dimensions")
}

#[test]
fn roundtrip_gradient_32x32() {
    let backend = SequentialBackend;
    let (nz, ny, nx) = (1usize, 32usize, 32usize);
    let total = nz * ny * nx;

    let mut data_vec: Vec<f32> = Vec::with_capacity(total);
    let max_idx = (ny * nx - 1) as f32;
    for y in 0..ny {
        for x in 0..nx {
            let idx = (y * nx + x) as f32;
            let val = if max_idx > 0.0 {
                idx / max_idx * 255.0
            } else {
                0.0
            };
            data_vec.push(val);
        }
    }

    let image = image_from_values([nz, ny, nx], data_vec.clone());
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("gradient.jpg");

    crate::write_jpeg(&path, &image, &backend).expect("write_jpeg failed");
    let loaded = crate::read_jpeg(&path, &backend).expect("read_jpeg failed");

    assert_eq!(loaded.shape(), [nz, ny, nx]);
    let loaded_slice = loaded.data_slice().expect("contiguous host data");
    for i in 0..total {
        let diff = (loaded_slice[i] - data_vec[i]).abs();
        assert!(
            diff <= 5.0,
            "pixel {} differs by {}: original={}, loaded={}",
            i,
            diff,
            data_vec[i],
            loaded_slice[i]
        );
    }
}

#[test]
fn spatial_metadata_defaults() {
    let backend = SequentialBackend;
    let image = image_from_values([1usize, 4, 4], vec![128.0f32; 16]);
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("meta.jpg");

    crate::write_jpeg(&path, &image, &backend).expect("write failed");
    let loaded = crate::read_jpeg(&path, &backend).expect("read failed");

    assert_eq!(
        [loaded.origin()[0], loaded.origin()[1], loaded.origin()[2]],
        [0.0, 0.0, 0.0]
    );
    assert_eq!(
        [
            loaded.spacing()[0],
            loaded.spacing()[1],
            loaded.spacing()[2]
        ],
        [1.0, 1.0, 1.0]
    );
    assert_eq!(loaded.direction(), &Direction::<3>::identity());
}

#[test]
fn roundtrip_non_square_16x48() {
    let backend = SequentialBackend;
    let (nz, ny, nx) = (1usize, 16usize, 48usize);
    let total = nz * ny * nx;

    let mut data_vec: Vec<f32> = Vec::with_capacity(total);
    for _y in 0..ny {
        for x in 0..nx {
            let val = (x as f32) / (nx as f32 - 1.0) * 255.0;
            data_vec.push(val);
        }
    }

    let image = image_from_values([nz, ny, nx], data_vec.clone());
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("rect.jpeg");

    crate::write_jpeg(&path, &image, &backend).expect("write failed");
    let loaded = crate::read_jpeg(&path, &backend).expect("read failed");

    assert_eq!(loaded.shape(), [nz, ny, nx]);
    let loaded_slice = loaded.data_slice().expect("contiguous host data");
    for i in 0..total {
        let diff = (loaded_slice[i] - data_vec[i]).abs();
        assert!(
            diff <= 5.0,
            "pixel {} differs by {}: original={}, loaded={}",
            i,
            diff,
            data_vec[i],
            loaded_slice[i]
        );
    }
}

#[test]
fn write_rejects_nz_not_one() {
    let backend = SequentialBackend;
    let image = image_from_values([2usize, 4, 4], vec![0.0f32; 2 * 4 * 4]);
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("bad.jpg");

    let result = crate::write_jpeg(&path, &image, &backend);
    assert!(result.is_err(), "write_jpeg should reject nz=2");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("nz=2"),
        "error message should contain 'nz=2', got: {}",
        msg
    );
}

#[test]
fn read_nonexistent_file_errors() {
    let backend = SequentialBackend;
    let result = crate::read_jpeg("/nonexistent/path/to/image.jpg", &backend);
    assert!(result.is_err(), "read_jpeg should fail for missing file");
}

#[test]
fn roundtrip_single_pixel() {
    let backend = SequentialBackend;
    let original_val = 137.0f32;
    let image = image_from_values([1usize, 1, 1], vec![original_val]);
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("pixel.jpg");

    crate::write_jpeg(&path, &image, &backend).expect("write failed");
    let loaded = crate::read_jpeg(&path, &backend).expect("read failed");

    assert_eq!(loaded.shape(), [1, 1, 1]);
    let loaded_slice = loaded.data_slice().expect("contiguous host data");
    let diff = (loaded_slice[0] - original_val).abs();
    assert!(
        diff <= 5.0,
        "single pixel differs by {}: original={}, loaded={}",
        diff,
        original_val,
        loaded_slice[0]
    );
}

#[test]
fn reader_and_writer_delegate_to_canonical_operations() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("delegation.jpg");
    let input = image_from_values([1, 1, 3], vec![16.0, 128.0, 240.0]);
    let writer = JpegWriter::new(SequentialBackend);
    writer.write_image(&path, &input)?;
    let reader = JpegReader::new(SequentialBackend);
    let output = reader.read_image(&path)?;
    assert_eq!(output.shape(), [1, 1, 3]);
    assert_eq!(output.data_cow_on(&SequentialBackend).len(), 3);
    Ok(())
}

#[test]
fn writer_rejects_non_planar_and_mismatched_images() -> Result<()> {
    let image = image_from_values([2, 1, 1], vec![0.0, 1.0]);
    let path = tempdir()?.path().join("invalid.jpg");
    let error = write_jpeg(&path, &image, &SequentialBackend).unwrap_err();
    assert!(error.to_string().contains("depth=1"));
    Ok(())
}

#[test]
fn reader_reports_missing_files() {
    let error = read_jpeg("missing/ritk-image.jpg", &SequentialBackend).unwrap_err();
    assert!(error.to_string().contains("failed to open JPEG file"));
}
