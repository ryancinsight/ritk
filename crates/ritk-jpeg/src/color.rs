use std::path::Path;

use anyhow::{bail, Context, Result};
use coeus_core::ComputeBackend;
use image::{ColorType, RgbImage};
use ritk_image::native::RgbVolume;
use ritk_spatial::{Direction, Point, Spacing};

const RGB_CHANNELS: usize = 3;

/// Reads an RGB8 JPEG into a native image with shape `[1, height, width, 3]`.
pub fn read_jpeg_color_to_volume<B, P>(path: P, backend: &B) -> Result<RgbVolume<f32, B>>
where
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let image = read_rgb8_jpeg(path)?;
    let (width, height) = image.dimensions();
    let pixels = image.as_raw().iter().copied().map(f32::from).collect();
    rgb_volume_from_flat_pixels(pixels, height as usize, width as usize, backend)
}

/// Backend-bound RGB JPEG reader.
pub struct JpegColorReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> JpegColorReader<B> {
    /// Creates a reader that constructs RGB volumes on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Reads an RGB8 JPEG on the configured backend.
    pub fn read_volume<P: AsRef<Path>>(&self, path: P) -> Result<RgbVolume<f32, B>> {
        read_jpeg_color_to_volume(path, &self.backend)
    }
}

fn read_rgb8_jpeg(path: &Path) -> Result<RgbImage> {
    let image = image::open(path)
        .with_context(|| format!("failed to open JPEG file: {}", path.display()))?;
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

fn rgb_volume_from_flat_pixels<B: ComputeBackend>(
    pixels: Vec<f32>,
    height: usize,
    width: usize,
    backend: &B,
) -> Result<RgbVolume<f32, B>> {
    let expected = height
        .checked_mul(width)
        .and_then(|count| count.checked_mul(RGB_CHANNELS))
        .context("JPEG RGB volume shape overflow")?;
    if pixels.len() != expected {
        bail!(
            "JPEG RGB pixel count {} does not match shape [1, {height}, {width}, 3]",
            pixels.len()
        );
    }
    RgbVolume::from_flat_on(
        pixels,
        [1, height, width],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        backend,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use image::codecs::jpeg::JpegEncoder;
    use image::{GrayImage, Luma};
    use std::fs::File;
    use std::io::BufWriter;
    use tempfile::tempdir;

    fn write_rgb_jpeg(path: &Path, width: u32, height: u32, pixels: &[u8]) -> Result<()> {
        let image = RgbImage::from_raw(width, height, pixels.to_vec())
            .expect("invariant: test RGB dimensions match the pixel count");
        let writer = BufWriter::new(File::create(path)?);
        JpegEncoder::new_with_quality(writer, 100).encode_image(&image)?;
        Ok(())
    }

    #[test]
    fn color_reader_preserves_decoded_interleaved_samples() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("rgb.jpg");
        write_rgb_jpeg(&path, 2, 1, &[255, 0, 0, 0, 255, 0])?;
        let backend = SequentialBackend;
        let volume = read_jpeg_color_to_volume(&path, &backend)?;
        let expected: Vec<f32> = image::open(&path)?
            .to_rgb8()
            .into_raw()
            .into_iter()
            .map(f32::from)
            .collect();
        assert_eq!(volume.shape(), [1, 1, 2, 3]);
        assert_eq!(volume.data_cow_on(&backend).as_ref(), expected.as_slice());
        assert_eq!(volume.spatial_shape(), [1, 1, 2]);
        assert_eq!(volume.channels(), 3);
        assert_eq!(volume.origin().to_array(), [0.0; 3]);
        assert_eq!(volume.spacing().to_array(), [1.0; 3]);
        Ok(())
    }

    #[test]
    fn color_reader_rejects_grayscale_jpeg() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("gray.jpg");
        let mut image = GrayImage::new(1, 1);
        image.put_pixel(0, 0, Luma([24]));
        JpegEncoder::new_with_quality(BufWriter::new(File::create(&path)?), 100)
            .encode_image(&image)?;
        let error = read_jpeg_color_to_volume(&path, &SequentialBackend).unwrap_err();
        assert!(error.to_string().contains("supports only Rgb8"));
        Ok(())
    }
}
