use std::path::Path;

use anyhow::{bail, Context, Result};
use coeus_core::ComputeBackend;
use image::{ColorType, RgbImage};
use ritk_image::RgbVolume;
use ritk_spatial::{Direction, Point, Spacing};

use crate::sorted_png_files;

const RGB_CHANNELS: usize = 3;

/// Reads one RGB8 PNG into a native RGB volume shaped `[1, height, width, 3]`.
pub fn read_png_color_to_volume<B, P>(path: P, backend: &B) -> Result<RgbVolume<f32, B>>
where
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let image = read_rgb8_png(path.as_ref())?;
    let (width, height) = image.dimensions();
    rgb_volume_from_flat_pixels(
        image.into_raw().into_iter().map(f32::from).collect(),
        [1, height as usize, width as usize],
        backend,
    )
}

/// Reads a naturally sorted directory of RGB8 PNGs into a native RGB volume.
pub fn read_png_color_series<B, P>(path: P, backend: &B) -> Result<RgbVolume<f32, B>>
where
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let files = sorted_png_files(path.as_ref())?;
    let first = read_rgb8_png(&files[0])?;
    let (width, height) = first.dimensions();
    let mut pixels = Vec::new();
    append_rgb_pixels(&mut pixels, &first)?;
    for file in &files[1..] {
        let image = read_rgb8_png(file)?;
        let (actual_width, actual_height) = image.dimensions();
        if (actual_width, actual_height) != (width, height) {
            bail!(
                "PNG RGB size mismatch: {} is {actual_width}x{actual_height} but expected {width}x{height}",
                file.display()
            );
        }
        append_rgb_pixels(&mut pixels, &image)?;
    }
    rgb_volume_from_flat_pixels(
        pixels,
        [files.len(), height as usize, width as usize],
        backend,
    )
}

/// Backend-bound RGB PNG reader.
pub struct PngColorReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> PngColorReader<B> {
    /// Creates a single-slice color reader on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Reads one RGB8 PNG.
    pub fn read_volume<P: AsRef<Path>>(&self, path: P) -> Result<RgbVolume<f32, B>> {
        read_png_color_to_volume(path, &self.backend)
    }
}

/// Backend-bound RGB PNG series reader.
pub struct PngColorSeriesReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> PngColorSeriesReader<B> {
    /// Creates a color-series reader on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Reads a naturally sorted RGB8 PNG series.
    pub fn read_volume<P: AsRef<Path>>(&self, path: P) -> Result<RgbVolume<f32, B>> {
        read_png_color_series(path, &self.backend)
    }
}

fn read_rgb8_png(path: &Path) -> Result<RgbImage> {
    let image =
        image::open(path).with_context(|| format!("failed to open PNG: {}", path.display()))?;
    let color = image.color();
    if color != ColorType::Rgb8 {
        bail!(
            "PNG RGB color loader supports only Rgb8; {} decoded as {color:?}",
            path.display()
        );
    }
    Ok(image.to_rgb8())
}

fn append_rgb_pixels(output: &mut Vec<f32>, image: &RgbImage) -> Result<()> {
    output
        .try_reserve(image.as_raw().len())
        .context("PNG RGB series pixel allocation failed")?;
    output.extend(image.as_raw().iter().copied().map(f32::from));
    Ok(())
}

fn rgb_volume_from_flat_pixels<B: ComputeBackend>(
    pixels: Vec<f32>,
    spatial_shape: [usize; 3],
    backend: &B,
) -> Result<RgbVolume<f32, B>> {
    let expected = spatial_shape
        .into_iter()
        .chain([RGB_CHANNELS])
        .try_fold(1usize, |product, dimension| product.checked_mul(dimension))
        .context("PNG RGB volume shape overflow")?;
    if pixels.len() != expected {
        bail!(
            "PNG RGB pixel count {} does not match spatial shape {spatial_shape:?} with three channels",
            pixels.len()
        );
    }
    RgbVolume::from_flat_on(
        pixels,
        spatial_shape,
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
    use image::{GrayImage, Luma};
    use tempfile::tempdir;

    fn write_rgb_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
        RgbImage::from_raw(width, height, pixels.to_vec())
            .expect("invariant: test dimensions match RGB pixel count")
            .save(path)
            .expect("test RGB PNG write must succeed");
    }

    #[test]
    fn color_slice_preserves_interleaved_samples() -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("rgb.png");
        write_rgb_png(&path, 2, 1, &[255, 0, 0, 0, 128, 255]);
        let volume = read_png_color_to_volume(&path, &SequentialBackend)?;
        assert_eq!(volume.shape(), [1, 1, 2, 3]);
        assert_eq!(
            volume.data_cow_on(&SequentialBackend).as_ref(),
            &[255.0, 0.0, 0.0, 0.0, 128.0, 255.0]
        );
        Ok(())
    }

    #[test]
    fn color_series_natural_sorts_slices() -> Result<()> {
        let directory = tempdir()?;
        write_rgb_png(&directory.path().join("slice10.png"), 1, 1, &[10, 11, 12]);
        write_rgb_png(&directory.path().join("slice2.png"), 1, 1, &[2, 3, 4]);
        write_rgb_png(&directory.path().join("slice1.png"), 1, 1, &[1, 5, 9]);
        let volume = read_png_color_series(directory.path(), &SequentialBackend)?;
        assert_eq!(volume.shape(), [3, 1, 1, 3]);
        assert_eq!(
            volume.data_cow_on(&SequentialBackend).as_ref(),
            &[1.0, 5.0, 9.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]
        );
        Ok(())
    }

    #[test]
    fn color_reader_rejects_grayscale_png() -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("gray.png");
        let mut image = GrayImage::new(1, 1);
        image.put_pixel(0, 0, Luma([42]));
        image.save(&path)?;
        let error = read_png_color_to_volume(&path, &SequentialBackend).unwrap_err();
        assert!(error.to_string().contains("supports only Rgb8"));
        Ok(())
    }

    #[test]
    fn color_series_rejects_dimension_mismatch() -> Result<()> {
        let directory = tempdir()?;
        write_rgb_png(&directory.path().join("slice1.png"), 1, 1, &[1, 2, 3]);
        write_rgb_png(
            &directory.path().join("slice2.png"),
            2,
            1,
            &[4, 5, 6, 7, 8, 9],
        );
        let error = read_png_color_series(directory.path(), &SequentialBackend).unwrap_err();
        assert!(error.to_string().contains("PNG RGB size mismatch"));
        Ok(())
    }
}
