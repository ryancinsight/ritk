use super::*;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use image::{GrayImage, Luma, Rgb};
use tempfile::tempdir;

type TestBackend = NdArray<f32>;

fn write_rgb_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
    let image = RgbImage::from_raw(width, height, pixels.to_vec())
        .expect("test RGB image dimensions must match pixel count");
    image.save(path).expect("test RGB PNG write must succeed");
}

fn volume_values(volume: &RgbVolume<TestBackend>) -> Vec<f32> {
    volume.with_data_slice(|s| s.to_vec())
}

#[test]
fn read_png_color_to_volume_preserves_interleaved_rgb_samples() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("rgb.png");
    write_rgb_png(&path, 2, 1, &[255, 0, 0, 0, 128, 255]);
    let device = <TestBackend as Backend>::Device::default();

    let volume = read_png_color_to_volume::<TestBackend, _>(&path, &device)?;

    assert_eq!(volume.shape(), [1, 1, 2, 3]);
    assert_eq!(volume.spatial_shape(), [1, 1, 2]);
    assert_eq!(
        volume_values(&volume),
        vec![255.0, 0.0, 0.0, 0.0, 128.0, 255.0]
    );
    assert_eq!(volume.origin().to_array(), [0.0, 0.0, 0.0]);
    assert_eq!(
        [
            volume.spacing()[0],
            volume.spacing()[1],
            volume.spacing()[2]
        ],
        [1.0, 1.0, 1.0]
    );
    Ok(())
}

#[test]
fn read_png_color_series_natural_sorts_and_stacks_rgb_slices() -> Result<()> {
    let dir = tempdir()?;
    write_rgb_png(&dir.path().join("slice10.png"), 1, 1, &[10, 11, 12]);
    write_rgb_png(&dir.path().join("slice2.png"), 1, 1, &[2, 3, 4]);
    write_rgb_png(&dir.path().join("slice1.png"), 1, 1, &[1, 5, 9]);
    let device = <TestBackend as Backend>::Device::default();

    let volume = read_png_color_series::<TestBackend, _>(dir.path(), &device)?;

    assert_eq!(volume.shape(), [3, 1, 1, 3]);
    assert_eq!(
        volume_values(&volume),
        vec![1.0, 5.0, 9.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]
    );
    Ok(())
}

#[test]
fn read_png_color_to_volume_rejects_grayscale_png() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("gray.png");
    let mut image = GrayImage::new(1, 1);
    image.put_pixel(0, 0, Luma([42]));
    image.save(&path)?;
    let device = <TestBackend as Backend>::Device::default();

    let err = read_png_color_to_volume::<TestBackend, _>(&path, &device).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("supports only Rgb8"),
        "expected Rgb8 rejection, got {msg}"
    );
    Ok(())
}

#[test]
fn png_color_reader_delegates_to_rgb_loader() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("reader.png");
    write_rgb_png(&path, 1, 1, &[7, 8, 9]);
    let reader = PngColorReader::<TestBackend>::new(Default::default());

    let volume = reader.read_volume(&path)?;

    assert_eq!(volume.shape(), [1, 1, 1, 3]);
    assert_eq!(volume_values(&volume), vec![7.0, 8.0, 9.0]);
    Ok(())
}

#[test]
fn read_png_color_series_rejects_dimension_mismatch() -> Result<()> {
    let dir = tempdir()?;
    write_rgb_png(&dir.path().join("slice1.png"), 1, 1, &[1, 2, 3]);
    let mut image = RgbImage::new(2, 1);
    image.put_pixel(0, 0, Rgb([4, 5, 6]));
    image.put_pixel(1, 0, Rgb([7, 8, 9]));
    image.save(dir.path().join("slice2.png"))?;
    let device = <TestBackend as Backend>::Device::default();

    let err = read_png_color_series::<TestBackend, _>(dir.path(), &device).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("PNG RGB size mismatch"),
        "expected dimension mismatch, got {msg}"
    );
    Ok(())
}
