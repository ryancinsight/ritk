use std::path::Path;

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use image::{ColorType, RgbImage};
use ritk_core::image::RgbVolume;
use ritk_core::spatial::{Direction, Point, Spacing};

const RGB_CHANNELS: usize = 3;

/// Read one decoded RGB8 JPEG into an `RgbVolume<B>` with shape `[1, height, width, 3]`.
pub fn read_jpeg_color_to_volume<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<RgbVolume<B>> {
    let path = path.as_ref();
    let image = read_rgb8_jpeg(path)?;
    let (width, height) = image.dimensions();
    let pixels = rgb_pixels_to_f32(&image);
    rgb_volume_from_flat_pixels(pixels, height as usize, width as usize, device)
}

/// Device-bound RGB JPEG reader.
pub struct JpegColorReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> JpegColorReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read_volume<P: AsRef<Path>>(&self, path: P) -> Result<RgbVolume<B>> {
        read_jpeg_color_to_volume(path, &self.device)
    }
}

fn read_rgb8_jpeg(path: &Path) -> Result<RgbImage> {
    let image = image::open(path)
        .with_context(|| format!("Failed to open JPEG file: {}", path.display()))?;
    let color = image.color();
    if color != ColorType::Rgb8 {
        bail!(
            "JPEG RGB color loader supports only Rgb8; {} decoded as {:?}",
            path.display(),
            color
        );
    }
    Ok(image.to_rgb8())
}

fn rgb_pixels_to_f32(image: &RgbImage) -> Vec<f32> {
    image.as_raw().iter().map(|&v| v as f32).collect()
}

fn rgb_volume_from_flat_pixels<B: Backend>(
    pixels: Vec<f32>,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Result<RgbVolume<B>> {
    let expected = height
        .checked_mul(width)
        .and_then(|n| n.checked_mul(RGB_CHANNELS))
        .context("JPEG RGB volume shape overflow")?;
    if pixels.len() != expected {
        bail!(
            "JPEG RGB pixel count {} does not match shape [1, {}, {}, 3]",
            pixels.len(),
            height,
            width
        );
    }

    let tensor = Tensor::<B, 4>::from_data(
        TensorData::new(pixels, Shape::new([1, height, width, RGB_CHANNELS])),
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
    use image::codecs::jpeg::JpegEncoder;
    use image::{GrayImage, Luma, RgbImage};
    use std::fs::File;
    use std::io::BufWriter;
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    fn write_rgb_jpeg(path: &Path, width: u32, height: u32, pixels: &[u8]) -> Result<()> {
        let image = RgbImage::from_raw(width, height, pixels.to_vec())
            .expect("test RGB image dimensions must match pixel count");
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut encoder = JpegEncoder::new_with_quality(writer, 100);
        encoder.encode_image(&image)?;
        Ok(())
    }

    fn write_gray_jpeg(path: &Path) -> Result<()> {
        let mut image = GrayImage::new(2, 1);
        image.put_pixel(0, 0, Luma([24]));
        image.put_pixel(1, 0, Luma([192]));
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut encoder = JpegEncoder::new_with_quality(writer, 100);
        encoder.encode_image(&image)?;
        Ok(())
    }

    fn decoded_rgb_values(path: &Path) -> Result<Vec<f32>> {
        Ok(image::open(path)?
            .to_rgb8()
            .as_raw()
            .iter()
            .map(|&v| v as f32)
            .collect())
    }

    fn volume_values(volume: &RgbVolume<TestBackend>) -> Vec<f32> {
        volume.data_vec()
    }

    #[test]
    fn read_jpeg_color_to_volume_preserves_decoded_interleaved_rgb_samples() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("rgb.jpg");
        write_rgb_jpeg(&path, 2, 2, &[255, 0, 0, 0, 255, 0, 0, 0, 255, 240, 240, 0])?;
        let device = <TestBackend as Backend>::Device::default();

        let volume = read_jpeg_color_to_volume::<TestBackend, _>(&path, &device)?;
        let expected = decoded_rgb_values(&path)?;

        assert_eq!(volume.shape(), [1, 2, 2, 3]);
        assert_eq!(volume.spatial_shape(), [1, 2, 2]);
        assert_eq!(volume_values(&volume), expected);
        assert_eq!(
            [volume.origin()[0], volume.origin()[1], volume.origin()[2]],
            [0.0, 0.0, 0.0]
        );
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
    fn read_jpeg_color_to_volume_rejects_grayscale_jpeg() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("gray.jpg");
        write_gray_jpeg(&path)?;
        let device = <TestBackend as Backend>::Device::default();

        let err = read_jpeg_color_to_volume::<TestBackend, _>(&path, &device).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("supports only Rgb8"),
            "expected Rgb8 rejection, got {msg}"
        );
        Ok(())
    }

    #[test]
    fn jpeg_color_reader_delegates_to_rgb_loader() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("reader.jpg");
        write_rgb_jpeg(&path, 1, 1, &[32, 128, 224])?;
        let reader = JpegColorReader::<TestBackend>::new(Default::default());

        let volume = reader.read_volume(&path)?;

        assert_eq!(volume.shape(), [1, 1, 1, 3]);
        assert_eq!(volume_values(&volume), decoded_rgb_values(&path)?);
        Ok(())
    }
}
