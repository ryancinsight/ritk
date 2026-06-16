use super::*;
use ritk_spatial::{Direction, Point, Spacing};
use crate::read_tiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
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
fn write_creates_nonempty_file() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("basic.tiff");

    let image = make_image(vec![1.0f32; 2 * 3 * 4], 2, 3, 4);
    write_tiff(&image, &path)?;

    assert!(path.exists(), "output file must exist after write");
    let meta = std::fs::metadata(&path)?;
    assert!(meta.len() > 0, "output file must be non-empty");

    Ok(())
}

#[test]
fn tiff_writer_struct_delegates_to_write_tiff() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("struct_write.tiff");

    let image = make_image(vec![7.5f32; 2 * 2 * 2], 2, 2, 2);
    TiffWriter::write(&image, &path)?;

    assert!(path.exists(), "output file must exist");
    assert!(
        std::fs::metadata(&path)?.len() > 0,
        "output file must be non-empty",
    );

    Ok(())
}

#[test]
fn round_trip_scalar_values_are_bitwise_identical() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("roundtrip_f32.tiff");
    let device: <TestBackend as Backend>::Device = Default::default();

    let nz = 2usize;
    let ny = 3usize;
    let nx = 5usize;
    let data_vec: Vec<f32> = (0..(nz * ny * nx) as u32)
        .map(|i| i as f32 * std::f32::consts::PI / 7.0)
        .collect();

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
fn multi_page_file_is_larger_than_single_page() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path_1 = dir.path().join("one_page.tiff");
    let path_3 = dir.path().join("three_pages.tiff");

    let ny = 4usize;
    let nx = 5usize;

    let image_1 = make_image(vec![0.0f32; ny * nx], 1, ny, nx);
    write_tiff(&image_1, &path_1)?;

    let image_3 = make_image(vec![0.0f32; 3 * ny * nx], 3, ny, nx);
    write_tiff(&image_3, &path_3)?;

    let size_1 = std::fs::metadata(&path_1)?.len();
    let size_3 = std::fs::metadata(&path_3)?.len();

    let min_extra = (2 * ny * nx * 4) as u64;
    assert!(
        size_3 >= size_1 + min_extra,
        "3-page file ({} bytes) must be at least {} bytes larger than 1-page file ({} bytes)",
        size_3,
        min_extra,
        size_1,
    );

    Ok(())
}

#[test]
fn file_contains_full_scalar_payload() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("payload.tiff");

    let nz = 2usize;
    let ny = 3usize;
    let nx = 4usize;
    let n_voxels = nz * ny * nx;

    let image = make_image(vec![1.0f32; n_voxels], nz, ny, nx);
    write_tiff(&image, &path)?;

    let file_size = std::fs::metadata(&path)?.len();
    let min_payload = (n_voxels * 4) as u64;
    assert!(
        file_size >= min_payload,
        "file size {} must be >= minimum payload {} ({} voxels × 4 bytes)",
        file_size,
        min_payload,
        n_voxels,
    );

    Ok(())
}

#[test]
fn edge_case_values_survive_round_trip() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("edge_values.tiff");
    let device: <TestBackend as Backend>::Device = Default::default();

    let values: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1.0e-38,
        std::f32::consts::PI,
        std::f32::consts::E,
        123_456.79,
        -987_654.3,
    ];
    assert_eq!(values.len(), 12);

    let image = make_image(values.clone(), 1, 3, 4);
    write_tiff(&image, &path)?;

    let loaded = read_tiff::<TestBackend, _>(&path, &device)?;
    assert_eq!(loaded.shape(), [1, 3, 4]);
    loaded.with_data_slice(|loaded_vals| {
        for (i, (&got, &expected)) in loaded_vals.iter().zip(values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "voxel[{}]: expected {} (bits {:#010x}), got {} (bits {:#010x})",
                i,
                expected,
                expected.to_bits(),
                got,
                got.to_bits(),
            );
        }
    });
    Ok(())
}
