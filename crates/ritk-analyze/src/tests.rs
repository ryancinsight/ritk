use anyhow::Result;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
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
    let image = Image::new(
        tensor,
        Point::new([2.5, 5.0, 7.5]),
        Spacing::new([1.25, 2.5, 3.75]),
        Direction::identity(),
    );

    write_analyze(&path, &image)?;
    let loaded = read_analyze::<TestBackend, _>(&path, &device)?;

    assert_eq!(loaded.shape(), [2, 3, 4]);
    assert_eq!(*loaded.spacing(), Spacing::new([1.25, 2.5, 3.75]));
    assert_eq!(*loaded.origin(), Point::new([2.5, 5.0, 7.5]));
    assert_eq!(*loaded.direction(), Direction::identity());
    assert_eq!(image_values(&loaded), values);

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
