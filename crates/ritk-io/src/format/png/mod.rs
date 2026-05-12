use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::path::Path;

/// Read a single grayscale PNG into an Image<B, 3> with shape [1, height, width].
pub fn read_png_to_image<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let path = path.as_ref();
    let img = image::open(path)
        .with_context(|| format!("Failed to open PNG: {}", path.display()))?
        .to_luma8();

    let (width, height) = img.dimensions();
    let pixels: Vec<f32> = img.iter().map(|&v| v as f32).collect();

    let shape = Shape::new([1, height as usize, width as usize]);
    let data = TensorData::new(pixels, shape);
    let tensor = Tensor::<B, 3>::from_data(data, device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Read a series of PNG files from a directory into a 3D Image [depth, height, width].
///
/// PNGs are sorted by filename (natural sort for numbered files).
pub fn read_png_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let dir = path.as_ref();

    // Collect all PNG files
    let mut png_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();

    if png_files.is_empty() {
        anyhow::bail!("No PNG files found in {}", dir.display());
    }

    // Natural sort by filename numbers
    png_files.sort_by(|a, b| {
        let a_name = a.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let b_name = b.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        natural_cmp(a_name, b_name)
    });

    // Load first image to get dimensions
    let first_img = image::open(&png_files[0])
        .with_context(|| format!("Failed to open PNG: {}", png_files[0].display()))?
        .to_luma8();
    let (width, height) = first_img.dimensions();

    // Load all slices
    let mut all_pixels: Vec<f32> =
        Vec::with_capacity(png_files.len() * height as usize * width as usize);

    for file in &png_files {
        let img = image::open(file)
            .with_context(|| format!("Failed to open PNG: {}", file.display()))?
            .to_luma8();

        let (w, h) = img.dimensions();
        if w != width || h != height {
            anyhow::bail!(
                "PNG size mismatch: {} is {}x{} but expected {}x{}",
                file.display(),
                w,
                h,
                width,
                height
            );
        }

        all_pixels.extend(img.iter().map(|&v| v as f32));
    }

    let depth = png_files.len();
    let shape = Shape::new([depth, height as usize, width as usize]);
    let data = TensorData::new(all_pixels, shape);
    let tensor = Tensor::<B, 3>::from_data(data, device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Natural string comparison that handles embedded numbers.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let mut a_chars = a.chars().peekable();
    let mut b_chars = b.chars().peekable();

    loop {
        match (a_chars.peek(), b_chars.peek()) {
            (Some(&ac), Some(&bc)) if ac.is_ascii_digit() && bc.is_ascii_digit() => {
                // Parse full number from both
                let a_num: String = a_chars.clone().take_while(|c| c.is_ascii_digit()).collect();
                let b_num: String = b_chars.clone().take_while(|c| c.is_ascii_digit()).collect();
                let a_val: u64 = a_num.parse().unwrap_or(0);
                let b_val: u64 = b_num.parse().unwrap_or(0);
                match a_val.cmp(&b_val) {
                    std::cmp::Ordering::Equal => {
                        let ord = a_num.len().cmp(&b_num.len());
                        for _ in 0..a_num.len() {
                            a_chars.next();
                        }
                        for _ in 0..b_num.len() {
                            b_chars.next();
                        }
                        if ord != std::cmp::Ordering::Equal {
                            return ord;
                        }
                    }
                    ord => return ord,
                }
            }
            (Some(&ac), Some(&bc)) => match ac.cmp(&bc) {
                std::cmp::Ordering::Equal => {
                    a_chars.next();
                    b_chars.next();
                }
                ord => return ord,
            },
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (None, None) => return std::cmp::Ordering::Equal,
        }
    }
}

use crate::domain::ImageReader;

/// DIP boundary executing strict `ImageReader` invariants over standard PNG single slices.
pub struct PngReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> PngReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> ImageReader<B, 3> for PngReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_png_to_image(path, &self.device).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// DIP boundary executing strict `ImageReader` invariants over PNG sequential volumes.
pub struct PngSeriesReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> PngSeriesReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> ImageReader<B, 3> for PngSeriesReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_png_series(path, &self.device).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{natural_cmp, read_png_series, read_png_to_image};
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
        let data = image.data().clone().to_data();
        data.as_slice::<f32>().unwrap().to_vec()
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
            image.direction().0,
            nalgebra::SMatrix::<f64, 3, 3>::identity()
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
}
