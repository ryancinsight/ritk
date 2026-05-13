use std::path::Path;

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use image::{ColorType, RgbImage};
use ritk_core::image::RgbVolume;
use ritk_core::spatial::{Direction, Point, Spacing};

use crate::sorted_png_files;

const RGB_CHANNELS: usize = 3;

/// Read one RGB8 PNG into an `RgbVolume<B>` with shape `[1, height, width, 3]`.
pub fn read_png_color_to_volume<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<RgbVolume<B>> {
    let path = path.as_ref();
    let image = read_rgb8_png(path)?;
    let (width, height) = image.dimensions();
    let pixels = rgb_pixels_to_f32(&image);
    rgb_volume_from_flat_pixels(pixels, 1, height as usize, width as usize, device)
}

/// Read a directory of RGB8 PNG files into an `RgbVolume<B>`.
///
/// PNG files are sorted by natural filename order before stacking.
pub fn read_png_color_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<RgbVolume<B>> {
    let dir = path.as_ref();
    let png_files = sorted_png_files(dir)?;

    let first = read_rgb8_png(&png_files[0])?;
    let (width, height) = first.dimensions();
    let mut pixels =
        Vec::with_capacity(png_files.len() * height as usize * width as usize * RGB_CHANNELS);
    append_rgb_pixels(&mut pixels, &first);

    for file in &png_files[1..] {
        let image = read_rgb8_png(file)?;
        let (w, h) = image.dimensions();
        if w != width || h != height {
            bail!(
                "PNG RGB size mismatch: {} is {}x{} but expected {}x{}",
                file.display(),
                w,
                h,
                width,
                height
            );
        }
        append_rgb_pixels(&mut pixels, &image);
    }

    rgb_volume_from_flat_pixels(
        pixels,
        png_files.len(),
        height as usize,
        width as usize,
        device,
    )
}

/// DIP boundary for standard RGB PNG single slices.
pub struct PngColorReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> PngColorReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read_volume<P: AsRef<Path>>(&self, path: P) -> Result<RgbVolume<B>> {
        read_png_color_to_volume(path, &self.device)
    }
}

/// DIP boundary for RGB PNG sequential volumes.
pub struct PngColorSeriesReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> PngColorSeriesReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read_volume<P: AsRef<Path>>(&self, path: P) -> Result<RgbVolume<B>> {
        read_png_color_series(path, &self.device)
    }
}

fn read_rgb8_png(path: &Path) -> Result<RgbImage> {
    let image =
        image::open(path).with_context(|| format!("Failed to open PNG: {}", path.display()))?;
    let color = image.color();
    if color != ColorType::Rgb8 {
        bail!(
            "PNG RGB color loader supports only Rgb8; {} decoded as {:?}",
            path.display(),
            color
        );
    }
    Ok(image.to_rgb8())
}

fn append_rgb_pixels(out: &mut Vec<f32>, image: &RgbImage) {
    out.extend(image.as_raw().iter().map(|&v| v as f32));
}

fn rgb_pixels_to_f32(image: &RgbImage) -> Vec<f32> {
    image.as_raw().iter().map(|&v| v as f32).collect()
}

fn rgb_volume_from_flat_pixels<B: Backend>(
    pixels: Vec<f32>,
    depth: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Result<RgbVolume<B>> {
    let expected = depth
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| n.checked_mul(RGB_CHANNELS))
        .context("PNG RGB volume shape overflow")?;
    if pixels.len() != expected {
        bail!(
            "PNG RGB pixel count {} does not match shape [{}, {}, {}, 3]",
            pixels.len(),
            depth,
            height,
            width
        );
    }

    let tensor = Tensor::<B, 4>::from_data(
        TensorData::new(pixels, Shape::new([depth, height, width, RGB_CHANNELS])),
        device,
    );

    RgbVolume::try_new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

#[cfg(test)]
mod tests {
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
        let data = volume.data().clone().to_data();
        data.as_slice::<f32>().unwrap().to_vec()
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
        assert_eq!(volume.origin().to_vec(), vec![0.0, 0.0, 0.0]);
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
}
