use super::{read_jpeg, write_jpeg};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

type TestBackend = NdArray<f32>;

fn image_from_values(
    device: &<TestBackend as ritk_image::tensor::backend::Backend>::Device,
    shape: [usize; 3],
    values: Vec<f32>,
) -> Image<TestBackend, 3> {
    let tensor_data = TensorData::new(values, Shape::new(shape));
    let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

#[test]
fn roundtrip_gradient_32x32() {
    let device = Default::default();
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

    let image = image_from_values(&device, [nz, ny, nx], data_vec.clone());
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("gradient.jpg");

    write_jpeg(&path, &image).expect("write_jpeg failed");
    let loaded = read_jpeg::<TestBackend, _>(&path, &device).expect("read_jpeg failed");

    assert_eq!(loaded.shape(), [nz, ny, nx]);
    let loaded_data = loaded.data().to_data();
    let loaded_slice: &[f32] = loaded_data.as_slice().expect("slice conversion failed");

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
    let device = Default::default();
    let image = image_from_values(&device, [1usize, 4, 4], vec![128.0f32; 16]);

    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("meta.jpg");

    write_jpeg(&path, &image).expect("write failed");
    let loaded = read_jpeg::<TestBackend, _>(&path, &device).expect("read failed");

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
    let device = Default::default();
    let (nz, ny, nx) = (1usize, 16usize, 48usize);
    let total = nz * ny * nx;

    let mut data_vec: Vec<f32> = Vec::with_capacity(total);
    for _y in 0..ny {
        for x in 0..nx {
            let val = (x as f32) / (nx as f32 - 1.0) * 255.0;
            data_vec.push(val);
        }
    }

    let image = image_from_values(&device, [nz, ny, nx], data_vec.clone());
    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("rect.jpeg");

    write_jpeg(&path, &image).expect("write failed");
    let loaded = read_jpeg::<TestBackend, _>(&path, &device).expect("read failed");

    assert_eq!(loaded.shape(), [nz, ny, nx]);
    let loaded_data = loaded.data().to_data();
    let loaded_slice: &[f32] = loaded_data.as_slice().expect("slice conversion failed");

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
    let device = Default::default();
    let image = image_from_values(&device, [2usize, 4, 4], vec![0.0f32; 2 * 4 * 4]);

    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("bad.jpg");

    let result = write_jpeg(&path, &image);
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
    let device = Default::default();
    let result = read_jpeg::<TestBackend, _>("/nonexistent/path/to/image.jpg", &device);
    assert!(result.is_err(), "read_jpeg should fail for missing file");
}

#[test]
fn roundtrip_single_pixel() {
    let device = Default::default();
    let original_val = 137.0f32;
    let image = image_from_values(&device, [1usize, 1, 1], vec![original_val]);

    let dir = tempdir().expect("failed to create tempdir");
    let path = dir.path().join("pixel.jpg");

    write_jpeg(&path, &image).expect("write failed");
    let loaded = read_jpeg::<TestBackend, _>(&path, &device).expect("read failed");

    assert_eq!(loaded.shape(), [1, 1, 1]);
    let loaded_data = loaded.data().to_data();
    let loaded_slice: &[f32] = loaded_data.as_slice().expect("slice conversion failed");
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
fn native_read_jpeg_matches_burn() {
    use coeus_core::SequentialBackend;

    let device = Default::default();
    let (nz, ny, nx) = (1usize, 8usize, 12usize);
    let data_vec: Vec<f32> = (0..(ny * nx))
        .map(|i| (i as f32) / ((ny * nx - 1) as f32) * 255.0)
        .collect();
    let image = image_from_values(&device, [nz, ny, nx], data_vec);
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("coeus.jpg");
    write_jpeg(&path, &image).expect("write");

    let burn = read_jpeg::<TestBackend, _>(&path, &device).expect("burn read");
    let coeus = crate::native::read_jpeg(&path, &SequentialBackend).expect("coeus read");

    assert_eq!(coeus.shape(), [nz, ny, nx]);
    let burn_data = burn.data().to_data();
    let burn_slice: &[f32] = burn_data.as_slice().expect("slice");
    let coeus_slice = coeus.data_slice().expect("contiguous host data");
    assert_eq!(
        coeus_slice, burn_slice,
        "coeus and burn JPEG voxels identical"
    );
}

/// Strongest differential oracle for the (lossy) JPEG writer: since the
/// Atlas-native and Burn writers share the `write_jpeg_flat` encoding core,
/// their output files must be byte-for-byte identical for the same logical
/// image — the same encoder, same quantization, same bytes.
#[test]
fn native_writer_output_is_byte_identical_to_burn_writer() -> anyhow::Result<()> {
    let (nz, ny, nx) = (1usize, 8usize, 12usize);
    let values: Vec<f32> = (0..(nz * ny * nx)).map(|i| (i * 7 % 256) as f32).collect();

    let device: <TestBackend as ritk_image::tensor::backend::Backend>::Device = Default::default();
    let burn_image = image_from_values(&device, [nz, ny, nx], values.clone());

    let backend = coeus_core::SequentialBackend;
    let native_image = ritk_image::native::Image::from_flat_on(
        values,
        [nz, ny, nx],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &backend,
    )?;

    let dir = tempdir()?;
    let burn_path = dir.path().join("burn.jpg");
    let native_path = dir.path().join("native.jpg");
    write_jpeg(&burn_path, &burn_image)?;
    crate::writer::native::write_jpeg(&native_path, &native_image, &backend)?;

    assert_eq!(
        std::fs::read(&burn_path)?,
        std::fs::read(&native_path)?,
        "native and Burn JPEG writers must emit identical bytes"
    );
    Ok(())
}
