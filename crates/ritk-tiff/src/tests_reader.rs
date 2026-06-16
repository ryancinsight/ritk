use super::*;
use crate::write_tiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

type TestBackend = NdArray<f32>;

fn make_image(data: Vec<f32>, nz: usize, ny: usize, nx: usize) -> Image<TestBackend, 3> {
    let device: <TestBackend as Backend>::Device = Default::default();
    let tensor_data = TensorData::new(data, Shape::new([nz, ny, nx]));
    let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

#[test]
fn round_trip_single_slice_preserves_shape_and_values() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("single_slice.tiff");
    let device: <TestBackend as Backend>::Device = Default::default();

    let nz = 1usize;
    let ny = 4usize;
    let nx = 5usize;
    let data_vec: Vec<f32> = (0u32..(nz * ny * nx) as u32)
        .map(|i| i as f32 * 0.5)
        .collect();

    let image = make_image(data_vec.clone(), nz, ny, nx);
    write_tiff(&image, &path)?;

    let loaded = read_tiff::<TestBackend, _>(&path, &device)?;

    assert_eq!(loaded.shape(), [nz, ny, nx], "shape mismatch");
    assert_eq!(loaded.origin(), &Point::new([0.0, 0.0, 0.0]));
    assert_eq!(loaded.spacing(), &Spacing::new([1.0, 1.0, 1.0]));
    assert_eq!(loaded.direction(), &Direction::identity());
    loaded.with_data_slice(|loaded_vals| {
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got,
            );
        }
    });
    Ok(())
}

#[test]
fn round_trip_multi_slice_preserves_shape_and_values() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("multi_slice.tiff");
    let device: <TestBackend as Backend>::Device = Default::default();

    let nz = 3usize;
    let ny = 4usize;
    let nx = 5usize;
    let data_vec: Vec<f32> = (0u32..(nz * ny * nx) as u32).map(|i| i as f32).collect();

    let image = make_image(data_vec.clone(), nz, ny, nx);
    write_tiff(&image, &path)?;

    let loaded = read_tiff::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [nz, ny, nx], "shape mismatch");

    loaded.with_data_slice(|loaded_vals| {
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got,
            );
        }
    });
    Ok(())
}

#[test]
fn slice_ordering_preserved_through_write_read() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("slice_order.tiff");
    let device: <TestBackend as Backend>::Device = Default::default();

    let nz = 4usize;
    let ny = 3usize;
    let nx = 2usize;
    let pixels_per_slice = ny * nx;
    let mut data_vec = Vec::with_capacity(nz * pixels_per_slice);
    for z in 0..nz {
        let fill_value = (z + 1) as f32 * 100.0;
        data_vec.extend(std::iter::repeat_n(fill_value, pixels_per_slice));
    }

    let image = make_image(data_vec.clone(), nz, ny, nx);
    write_tiff(&image, &path)?;

    let loaded = read_tiff::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [nz, ny, nx]);
    loaded.with_data_slice(|loaded_vals| {
        for z in 0..nz {
            let expected = (z + 1) as f32 * 100.0;
            let offset = z * pixels_per_slice;
            for px in 0..pixels_per_slice {
                let got = loaded_vals[offset + px];
                assert!(
                    (got - expected).abs() < 1e-6,
                    "slice {} pixel {}: expected {}, got {}",
                    z,
                    px,
                    expected,
                    got,
                );
            }
        }
    });
    Ok(())
}

#[test]
fn tiff_reader_struct_delegates_to_read_tiff() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("struct_read.tiff");
    let device: <TestBackend as Backend>::Device = Default::default();

    let data_vec: Vec<f32> = vec![42.0; 2 * 3 * 4];
    let image = make_image(data_vec, 2, 3, 4);
    write_tiff(&image, &path)?;

    let reader = TiffReader::<TestBackend>::new(device);
    let loaded = reader.read_image(&path)?;
    assert_eq!(loaded.shape(), [2, 3, 4]);
    loaded.with_data_slice(|loaded_vals| {
        for (i, &v) in loaded_vals.iter().enumerate() {
            assert!(
                (v - 42.0).abs() < 1e-6,
                "voxel[{}]: expected 42.0, got {}",
                i,
                v,
            );
        }
    });
    Ok(())
}

#[test]
fn missing_file_returns_error_with_open_message() {
    let device: <TestBackend as Backend>::Device = Default::default();
    let result = read_tiff::<TestBackend, _>("/nonexistent/path.tiff", &device);
    assert!(result.is_err(), "missing file must produce an error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("Cannot open TIFF file"),
        "error message must mention file open failure, got: {}",
        msg,
    );
}

#[test]
fn invalid_file_returns_error() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("not_a_tiff.tiff");
    std::fs::write(&path, b"this is not a tiff file")?;

    let device: <TestBackend as Backend>::Device = Default::default();
    let result = read_tiff::<TestBackend, _>(&path, &device);
    assert!(result.is_err(), "invalid TIFF must produce an error");

    Ok(())
}

#[test]
fn negative_values_survive_round_trip() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("negative.tiff");
    let device: <TestBackend as Backend>::Device = Default::default();

    let nz = 2usize;
    let ny = 3usize;
    let nx = 4usize;
    let data_vec: Vec<f32> = (0..(nz * ny * nx) as i32)
        .map(|i| (i as f32) - 12.0)
        .collect();

    let image = make_image(data_vec.clone(), nz, ny, nx);
    write_tiff(&image, &path)?;

    let loaded = read_tiff::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [nz, ny, nx]);
    loaded.with_data_slice(|loaded_vals| {
        for (i, (&got, &expected)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "voxel[{}]: expected {}, got {}",
                i,
                expected,
                got,
            );
        }
    });
    Ok(())
}
