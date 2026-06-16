use anyhow::Result;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use crate::{read_analyze, write_analyze};

type TestBackend = NdArray<f32>;

fn image_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .to_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

#[test]
fn analyze_roundtrip_preserves_shape_spacing_origin_and_values() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("volume.hdr");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let values: Vec<f32> = (0..24).map(|v| v as f32 + 0.25).collect();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(values.clone(), Shape::new([2, 3, 4])),
        &device,
    );
    // Core spacing is tensor-axis order [sz, sy, sx]; the file stores file-axis
    // [sx, sy, sz] = [3.75, 2.5, 1.25]. The `originator` field encodes the origin
    // as integer voxel coordinates, so a faithful round-trip requires each
    // world-space [x, y, z] origin component to be an exact integer multiple of
    // its per-axis spacing: ox=2·3.75, oy=2·2.5, oz=3·1.25.
    let image = Image::new(
        tensor,
        Point::new([7.5, 5.0, 3.75]),
        Spacing::new([1.25, 2.5, 3.75]),
        Direction::identity(),
    );

    write_analyze(&path, &image)?;
    let loaded = read_analyze::<TestBackend, _>(&path, &device)?;

    assert_eq!(loaded.shape(), [2, 3, 4]);
    assert_eq!(*loaded.spacing(), Spacing::new([1.25, 2.5, 3.75]));
    assert_eq!(*loaded.origin(), Point::new([7.5, 5.0, 3.75]));
    assert_eq!(*loaded.direction(), Direction::identity());
    assert_eq!(image_values(&loaded), values);

    Ok(())
}

#[test]
fn analyze_writer_emits_pixdim_in_file_axis_order() -> Result<()> {
    // The Analyze header stores spacing in file-axis order pixdim[1..3] = [sx, sy,
    // sz], the reverse of RITK's core tensor-axis spacing [sz, sy, sx]. A symmetric
    // round-trip cannot detect an axis swap, so assert the raw on-disk bytes.
    let dir = tempdir()?;
    let path = dir.path().join("axis.hdr");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0_f32; 24], Shape::new([2, 3, 4])),
        &device,
    );
    // Core spacing [sz, sy, sx] = [1.25, 2.5, 3.75]; expect pixdim = [3.75, 2.5, 1.25].
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.25, 2.5, 3.75]),
        Direction::identity(),
    );
    write_analyze(&path, &image)?;

    let hdr = std::fs::read(&path)?;
    let read_f32 = |off: usize| {
        f32::from_le_bytes([hdr[off], hdr[off + 1], hdr[off + 2], hdr[off + 3]])
    };
    assert_eq!(read_f32(80), 3.75, "pixdim[1] must be sx (file-axis X)");
    assert_eq!(read_f32(84), 2.5, "pixdim[2] must be sy (file-axis Y)");
    assert_eq!(read_f32(88), 1.25, "pixdim[3] must be sz (file-axis Z)");

    Ok(())
}

#[test]
fn analyze_reader_accepts_img_path_and_rejects_invalid_header() -> Result<()> {
    let dir = tempdir()?;
    let hdr_path = dir.path().join("volume.hdr");
    let img_path = dir.path().join("volume.img");
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0, 2.0], Shape::new([1, 1, 2])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    write_analyze(&hdr_path, &image)?;
    let loaded = read_analyze::<TestBackend, _>(&img_path, &device)?;
    assert_eq!(loaded.shape(), [1, 1, 2]);
    assert_eq!(image_values(&loaded), vec![1.0, 2.0]);

    std::fs::write(&hdr_path, [0u8; 348])?;
    let err = read_analyze::<TestBackend, _>(&hdr_path, &device).unwrap_err();
    assert!(
        err.to_string().contains("sizeof_hdr"),
        "error must identify invalid Analyze header, got: {err:#}"
    );

    Ok(())
}
