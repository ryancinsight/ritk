//! Native PNG single-slice, sequential-volume, and RGB image I/O.

use anyhow::{Context, Result};
use coeus_core::ComputeBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::{Path, PathBuf};

mod color;

pub use color::{
    read_png_color_series, read_png_color_to_volume, PngColorReader, PngColorSeriesReader,
};

/// Reads a grayscale PNG into a native image shaped `[1, height, width]`.
pub fn read_png_to_image<B, P>(path: P, backend: &B) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let (pixels, dimensions) = decode_png_single(path.as_ref())?;
    image_from_flat_pixels(pixels, dimensions, backend)
}

/// Reads a naturally sorted directory of grayscale PNGs into `[depth, height, width]`.
pub fn read_png_series<B, P>(path: P, backend: &B) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let (pixels, dimensions) = decode_png_series(path.as_ref())?;
    image_from_flat_pixels(pixels, dimensions, backend)
}

fn decode_png_single(path: &Path) -> Result<(Vec<f32>, [usize; 3])> {
    let image = image::open(path)
        .with_context(|| format!("failed to open PNG: {}", path.display()))?
        .to_luma8();
    let (width, height) = image.dimensions();
    Ok((
        image.into_raw().into_iter().map(f32::from).collect(),
        [1, height as usize, width as usize],
    ))
}

fn decode_png_series(directory: &Path) -> Result<(Vec<f32>, [usize; 3])> {
    let files = sorted_png_files(directory)?;
    let first = image::open(&files[0])
        .with_context(|| format!("failed to open PNG: {}", files[0].display()))?
        .to_luma8();
    let (width, height) = first.dimensions();
    let mut pixels = Vec::new();
    append_gray_pixels(&mut pixels, &first)?;
    for file in &files[1..] {
        let image = image::open(file)
            .with_context(|| format!("failed to open PNG: {}", file.display()))?
            .to_luma8();
        let (actual_width, actual_height) = image.dimensions();
        if (actual_width, actual_height) != (width, height) {
            anyhow::bail!(
                "PNG size mismatch: {} is {actual_width}x{actual_height} but expected {width}x{height}",
                file.display()
            );
        }
        append_gray_pixels(&mut pixels, &image)?;
    }
    Ok((pixels, [files.len(), height as usize, width as usize]))
}

fn append_gray_pixels(output: &mut Vec<f32>, image: &image::GrayImage) -> Result<()> {
    output
        .try_reserve(image.as_raw().len())
        .context("PNG series pixel allocation failed")?;
    output.extend(image.as_raw().iter().copied().map(f32::from));
    Ok(())
}

fn image_from_flat_pixels<B: ComputeBackend>(
    pixels: Vec<f32>,
    dimensions: [usize; 3],
    backend: &B,
) -> Result<Image<f32, B, 3>> {
    Image::from_flat_on(
        pixels,
        dimensions,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        backend,
    )
}

pub(crate) fn sorted_png_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(directory)
        .with_context(|| format!("failed to read directory: {}", directory.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        })
        .collect();
    if files.is_empty() {
        anyhow::bail!("no PNG files found in {}", directory.display());
    }
    files.sort_by(|left, right| {
        natural_cmp(
            left.file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or(""),
            right
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or(""),
        )
    });
    Ok(files)
}

/// Backend-bound grayscale PNG reader.
pub struct PngReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> PngReader<B> {
    /// Creates a single-slice reader on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Reads one grayscale PNG.
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<f32, B, 3>> {
        read_png_to_image(path, &self.backend)
    }
}

/// Backend-bound grayscale PNG series reader.
pub struct PngSeriesReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> PngSeriesReader<B> {
    /// Creates a series reader on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Reads a naturally sorted grayscale PNG series.
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<f32, B, 3>> {
        read_png_series(path, &self.backend)
    }
}

fn natural_cmp(left: &str, right: &str) -> std::cmp::Ordering {
    let mut left_chars = left.chars().peekable();
    let mut right_chars = right.chars().peekable();
    loop {
        match (left_chars.peek(), right_chars.peek()) {
            (Some(&left_char), Some(&right_char))
                if left_char.is_ascii_digit() && right_char.is_ascii_digit() =>
            {
                let left_digits: String = left_chars
                    .clone()
                    .take_while(char::is_ascii_digit)
                    .collect();
                let right_digits: String = right_chars
                    .clone()
                    .take_while(char::is_ascii_digit)
                    .collect();
                let left_significant = left_digits.trim_start_matches('0');
                let right_significant = right_digits.trim_start_matches('0');
                let left_significant = if left_significant.is_empty() {
                    "0"
                } else {
                    left_significant
                };
                let right_significant = if right_significant.is_empty() {
                    "0"
                } else {
                    right_significant
                };
                let numeric_order = left_significant
                    .len()
                    .cmp(&right_significant.len())
                    .then_with(|| left_significant.cmp(right_significant));
                if numeric_order != std::cmp::Ordering::Equal {
                    return numeric_order;
                }
                let length_order = left_digits.len().cmp(&right_digits.len());
                left_chars.nth(left_digits.len() - 1);
                right_chars.nth(right_digits.len() - 1);
                if length_order != std::cmp::Ordering::Equal {
                    return length_order;
                }
            }
            (Some(left_char), Some(right_char)) => {
                let order = left_char.cmp(right_char);
                if order != std::cmp::Ordering::Equal {
                    return order;
                }
                left_chars.next();
                right_chars.next();
            }
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (None, None) => return std::cmp::Ordering::Equal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use std::cmp::Ordering;
    use tempfile::tempdir;

    fn write_gray_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
        image::GrayImage::from_raw(width, height, pixels.to_vec())
            .expect("invariant: test dimensions match pixel count")
            .save(path)
            .expect("test PNG write must succeed");
    }

    #[test]
    fn single_slice_preserves_values_and_metadata() -> Result<()> {
        let directory = tempdir()?;
        let path = directory.path().join("slice.png");
        write_gray_png(&path, 3, 2, &[10, 20, 30, 40, 50, 60]);
        let image = read_png_to_image(&path, &SequentialBackend)?;
        assert_eq!(image.shape(), [1, 2, 3]);
        assert_eq!(image.data_slice()?, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        assert_eq!(image.origin().to_array(), [0.0; 3]);
        assert_eq!(image.spacing().to_array(), [1.0; 3]);
        Ok(())
    }

    #[test]
    fn series_natural_sorts_and_stacks_slices() -> Result<()> {
        let directory = tempdir()?;
        write_gray_png(&directory.path().join("slice10.png"), 2, 1, &[10, 11]);
        write_gray_png(&directory.path().join("slice2.png"), 2, 1, &[2, 3]);
        write_gray_png(&directory.path().join("slice1.png"), 2, 1, &[1, 4]);
        let image = read_png_series(directory.path(), &SequentialBackend)?;
        assert_eq!(image.shape(), [3, 1, 2]);
        assert_eq!(image.data_slice()?, &[1.0, 4.0, 2.0, 3.0, 10.0, 11.0]);
        Ok(())
    }

    #[test]
    fn series_rejects_dimension_mismatch() -> Result<()> {
        let directory = tempdir()?;
        write_gray_png(&directory.path().join("slice1.png"), 2, 1, &[1, 2]);
        write_gray_png(&directory.path().join("slice2.png"), 1, 1, &[3]);
        let error = read_png_series(directory.path(), &SequentialBackend).unwrap_err();
        assert!(error.to_string().contains("PNG size mismatch"));
        Ok(())
    }

    #[test]
    fn natural_order_compares_embedded_decimal_runs() {
        assert_eq!(natural_cmp("slice2", "slice10"), Ordering::Less);
        assert_eq!(natural_cmp("slice10", "slice2"), Ordering::Greater);
        assert_eq!(natural_cmp("slice2", "slice02"), Ordering::Less);
        assert_eq!(
            natural_cmp("slice99999999999999999999", "slice100000000000000000000"),
            Ordering::Less
        );
    }
}
