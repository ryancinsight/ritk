use std::path::Path;

use anyhow::{bail, Context, Result};
use coeus_core::ComputeBackend;
use consus_raster::PixelFormat;
use ritk_image::RgbVolume;
use ritk_spatial::{Direction, Point, Spacing};

use crate::decode::decode_file;

const RGB_CHANNELS: usize = 3;

/// Reads an RGB8 JPEG into a native image with shape `[1, height, width, 3]`.
///
/// Encoded raster orientation is preserved; EXIF display orientation is not
/// applied.
///
/// # Errors
///
/// Returns an error if the file is not an RGB JPEG, cannot be read, or contains
/// malformed, unsupported, truncated, or over-limit data.
pub fn read_jpeg_color_to_volume<B, P>(path: P, backend: &B) -> Result<RgbVolume<f32, B>>
where
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let image = decode_file(path)?;
    if image.format() != PixelFormat::Rgb {
        bail!(
            "JPEG RGB color loader supports only RGB; {} decoded as {:?}",
            path.display(),
            image.format()
        );
    }
    let width = usize::try_from(image.width()).context("JPEG width exceeds usize")?;
    let height = usize::try_from(image.height()).context("JPEG height exceeds usize")?;
    let pixels = image.into_pixels().into_iter().map(f32::from).collect();
    rgb_volume_from_flat_pixels(pixels, height, width, backend)
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
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as
    /// [`read_jpeg_color_to_volume`].
    pub fn read_volume<P: AsRef<Path>>(&self, path: P) -> Result<RgbVolume<f32, B>> {
        read_jpeg_color_to_volume(path, &self.backend)
    }
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
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;
    use coeus_core::SequentialBackend;
    use image::codecs::jpeg::JpegEncoder;
    use image::{GrayImage, Luma, RgbImage};
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

    fn mark_components_as_direct_rgb(path: &Path) -> Result<()> {
        let mut jpeg = std::fs::read(path)?;
        let frame = jpeg
            .windows(2)
            .position(|bytes| bytes == [0xFF, 0xC0])
            .expect("invariant: test encoder emits a baseline frame");
        let scan = jpeg
            .windows(2)
            .position(|bytes| bytes == [0xFF, 0xDA])
            .expect("invariant: test encoder emits a scan header");
        for (index, identifier) in (*b"RGB").into_iter().enumerate() {
            jpeg[frame + 10 + 3 * index] = identifier;
            jpeg[scan + 5 + 2 * index] = identifier;
        }
        std::fs::write(path, jpeg)?;
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
    fn color_reader_preserves_direct_rgb_components() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("direct-rgb.jpg");
        let pixels: Vec<u8> = [120, 64, 32].into_iter().cycle().take(8 * 8 * 3).collect();
        write_rgb_jpeg(&path, 8, 8, &pixels)?;
        mark_components_as_direct_rgb(&path)?;
        let expected: Vec<f32> = image::open(&path)?
            .to_rgb8()
            .into_raw()
            .into_iter()
            .map(f32::from)
            .collect();

        let volume = read_jpeg_color_to_volume(&path, &SequentialBackend)?;

        assert_eq!(volume.shape(), [1, 8, 8, 3]);
        assert_eq!(
            volume.data_cow_on(&SequentialBackend).as_ref(),
            expected.as_slice()
        );
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
        assert!(error.to_string().contains("supports only RGB"));
        Ok(())
    }
}
