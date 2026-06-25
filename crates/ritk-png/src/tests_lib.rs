use super::natural_cmp;
use super::*;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use std::cmp::Ordering;
use std::path::Path;
use tempfile::tempdir;

type TestBackend = NdArray<f32>;

fn write_gray_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
    let image = image::GrayImage::from_raw(width, height, pixels.to_vec())
        .expect("test image dimensions must match pixel count");
    image.save(path).expect("test PNG write must succeed");
}

fn tensor_values(image: &ritk_core::image::Image<TestBackend, 3>) -> Vec<f32> {
    image.data_slice().into_owned()
}

#[test]
fn read_png_to_image_preserves_shape_values_and_default_metadata() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("slice.png");
    write_gray_png(&path, 3, 2, &[10, 20, 30, 40, 50, 60]);

    let device: <TestBackend as Backend>::Device = Default::default();
    let image = read_png_to_image::<TestBackend, _>(&path, &device)?;

    assert_eq!(image.shape(), [1, 2, 3]);
    assert_eq!(
        tensor_values(&image),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );
    assert_eq!(
        [image.origin()[0], image.origin()[1], image.origin()[2]],
        [0.0, 0.0, 0.0]
    );
    assert_eq!(
        [image.spacing()[0], image.spacing()[1], image.spacing()[2]],
        [1.0, 1.0, 1.0]
    );
    assert_eq!(
        *image.direction(),
        ritk_spatial::Direction::<3>::identity()
    );

    Ok(())
}

#[test]
fn read_png_series_natural_sorts_and_stacks_slices() -> anyhow::Result<()> {
    let dir = tempdir()?;
    write_gray_png(&dir.path().join("slice10.png"), 2, 1, &[10, 11]);
    write_gray_png(&dir.path().join("slice2.png"), 2, 1, &[2, 3]);
    write_gray_png(&dir.path().join("slice1.png"), 2, 1, &[1, 4]);

    let device: <TestBackend as Backend>::Device = Default::default();
    let image = read_png_series::<TestBackend, _>(dir.path(), &device)?;

    assert_eq!(image.shape(), [3, 1, 2]);
    assert_eq!(tensor_values(&image), vec![1.0, 4.0, 2.0, 3.0, 10.0, 11.0]);

    Ok(())
}

#[test]
fn read_png_series_rejects_dimension_mismatch() -> anyhow::Result<()> {
    let dir = tempdir()?;
    write_gray_png(&dir.path().join("slice1.png"), 2, 1, &[1, 2]);
    write_gray_png(&dir.path().join("slice2.png"), 1, 1, &[3]);

    let device: <TestBackend as Backend>::Device = Default::default();
    let result = read_png_series::<TestBackend, _>(dir.path(), &device);
    let msg = match result {
        Ok(_) => panic!("mismatched PNG dimensions must fail"),
        Err(err) => format!("{err:?}"),
    };

    assert!(
        msg.contains("PNG size mismatch"),
        "error must name the dimension mismatch, got: {msg}"
    );

    Ok(())
}

#[test]
fn natural_cmp_orders_embedded_numbers_by_numeric_value() {
    assert_eq!(natural_cmp("slice2", "slice10"), Ordering::Less);
    assert_eq!(natural_cmp("slice10", "slice2"), Ordering::Greater);
    assert_eq!(natural_cmp("slice2", "slice02"), Ordering::Less);
}
