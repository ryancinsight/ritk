pub mod reader;
pub mod writer;

pub use reader::read_jpeg;
pub use writer::write_jpeg;

use crate::domain::{ImageReader, ImageWriter};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

/// DIP boundary executing strict `ImageReader` invariants over grayscale JPEG images.
///
/// JPEG is 2D-only. The reader produces an `Image<B, 3>` with tensor shape
/// `[1, height, width]` (i.e. `[nz=1, ny, nx]`), origin `(0,0,0)`, unit
/// spacing, and identity direction.
pub struct JpegReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> JpegReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> ImageReader<B, 3> for JpegReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_jpeg(path, &self.device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

/// DIP boundary executing strict `ImageWriter` invariants over grayscale JPEG images.
///
/// Requires `nz == 1`. Pixel values are clamped to `[0, 255]`, rounded, and
/// written as 8-bit grayscale. The `image` crate infers JPEG format from the
/// `.jpg` or `.jpeg` file extension.
pub struct JpegWriter<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for JpegWriter<B> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> ImageWriter<B, 3> for JpegWriter<B> {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_jpeg(path, image)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    /// Round-trip a 1×32×32 gradient image through JPEG write/read.
    ///
    /// JPEG is lossy; pixel values are asserted within ±5 of the original.
    #[test]
    fn roundtrip_gradient_32x32() {
        let device = Default::default();
        let (nz, ny, nx) = (1usize, 32usize, 32usize);
        let total = nz * ny * nx;

        // Gradient: value = (y * nx + x) scaled into [0, 255].
        let mut data_vec: Vec<f32> = Vec::with_capacity(total);
        let max_idx = (ny * nx - 1) as f32;
        for y in 0..ny {
            for x in 0..nx {
                let idx = (y * nx + x) as f32;
                let val = if max_idx > 0.0 {
                    idx / max_idx * 255.0
                } else {
                    0.0
                };
                data_vec.push(val);
            }
        }

        let shape = Shape::new([nz, ny, nx]);
        let tensor_data = TensorData::new(data_vec.clone(), shape);
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
        let origin = Point::new([0.0, 0.0, 0.0]);
        let spacing = Spacing::new([1.0, 1.0, 1.0]);
        let direction = Direction::identity();
        let image = Image::new(tensor, origin, spacing, direction);

        let dir = tempdir().expect("failed to create tempdir");
        let path = dir.path().join("gradient.jpg");

        write_jpeg(&path, &image).expect("write_jpeg failed");
        let loaded = read_jpeg::<TestBackend, _>(&path, &device).expect("read_jpeg failed");

        let loaded_shape = loaded.shape();
        assert_eq!(loaded_shape[0], nz, "nz mismatch");
        assert_eq!(loaded_shape[1], ny, "ny mismatch");
        assert_eq!(loaded_shape[2], nx, "nx mismatch");

        let loaded_data = loaded.data().to_data();
        let loaded_slice: &[f32] = loaded_data.as_slice().expect("slice conversion failed");

        for i in 0..total {
            let diff = (loaded_slice[i] - data_vec[i]).abs();
            assert!(
                diff <= 5.0,
                "pixel {} differs by {}: original={}, loaded={}",
                i,
                diff,
                data_vec[i],
                loaded_slice[i]
            );
        }
    }

    /// Verify that spatial metadata returned by `read_jpeg` matches the 2D
    /// JPEG default: origin (0,0,0), spacing (1,1,1), identity direction.
    #[test]
    fn spatial_metadata_defaults() {
        let device = Default::default();
        let shape = Shape::new([1usize, 4, 4]);
        let data = TensorData::new(vec![128.0f32; 16], shape);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        let dir = tempdir().expect("failed to create tempdir");
        let path = dir.path().join("meta.jpg");

        write_jpeg(&path, &image).expect("write failed");
        let loaded = read_jpeg::<TestBackend, _>(&path, &device).expect("read failed");

        let o = loaded.origin();
        assert!((o[0]).abs() < 1e-9, "origin[0]={}", o[0]);
        assert!((o[1]).abs() < 1e-9, "origin[1]={}", o[1]);
        assert!((o[2]).abs() < 1e-9, "origin[2]={}", o[2]);

        let s = loaded.spacing();
        assert!((s[0] - 1.0).abs() < 1e-9, "spacing[0]={}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-9, "spacing[1]={}", s[1]);
        assert!((s[2] - 1.0).abs() < 1e-9, "spacing[2]={}", s[2]);

        let d = loaded.direction();
        let id = Direction::<3>::identity();
        assert_eq!(d, &id, "direction is not identity");
    }

    /// Round-trip a non-square 1×16×48 image.
    #[test]
    fn roundtrip_non_square_16x48() {
        let device = Default::default();
        let (nz, ny, nx) = (1usize, 16usize, 48usize);
        let total = nz * ny * nx;

        let mut data_vec: Vec<f32> = Vec::with_capacity(total);
        for _y in 0..ny {
            for x in 0..nx {
                // Horizontal gradient per row, scaled to [0, 255].
                let val = (x as f32) / (nx as f32 - 1.0) * 255.0;
                data_vec.push(val);
            }
        }

        let shape = Shape::new([nz, ny, nx]);
        let tensor_data = TensorData::new(data_vec.clone(), shape);
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        let dir = tempdir().expect("failed to create tempdir");
        let path = dir.path().join("rect.jpeg");

        write_jpeg(&path, &image).expect("write failed");
        let loaded = read_jpeg::<TestBackend, _>(&path, &device).expect("read failed");

        let ls = loaded.shape();
        assert_eq!(ls[0], nz);
        assert_eq!(ls[1], ny);
        assert_eq!(ls[2], nx);

        let loaded_data = loaded.data().to_data();
        let loaded_slice: &[f32] = loaded_data.as_slice().expect("slice conversion failed");

        for i in 0..total {
            let diff = (loaded_slice[i] - data_vec[i]).abs();
            assert!(
                diff <= 5.0,
                "pixel {} differs by {}: original={}, loaded={}",
                i,
                diff,
                data_vec[i],
                loaded_slice[i]
            );
        }
    }

    /// Writing an image with nz != 1 must return an error containing "nz=".
    #[test]
    fn write_rejects_nz_not_one() {
        let device = Default::default();
        let shape = Shape::new([2usize, 4, 4]);
        let data = TensorData::new(vec![0.0f32; 2 * 4 * 4], shape);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        let dir = tempdir().expect("failed to create tempdir");
        let path = dir.path().join("bad.jpg");

        let result = write_jpeg(&path, &image);
        assert!(result.is_err(), "write_jpeg should reject nz=2");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("nz=2"),
            "error message should contain 'nz=2', got: {}",
            msg
        );
    }

    /// Reading a nonexistent file must return an error.
    #[test]
    fn read_nonexistent_file_errors() {
        let device = Default::default();
        let result = read_jpeg::<TestBackend, _>("/nonexistent/path/to/image.jpg", &device);
        assert!(result.is_err(), "read_jpeg should fail for missing file");
    }

    /// Round-trip a single-pixel 1×1×1 image (boundary case).
    #[test]
    fn roundtrip_single_pixel() {
        let device = Default::default();
        let original_val = 137.0f32;

        let shape = Shape::new([1usize, 1, 1]);
        let tensor_data = TensorData::new(vec![original_val], shape);
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        let dir = tempdir().expect("failed to create tempdir");
        let path = dir.path().join("pixel.jpg");

        write_jpeg(&path, &image).expect("write failed");
        let loaded = read_jpeg::<TestBackend, _>(&path, &device).expect("read failed");

        let ls = loaded.shape();
        assert_eq!(ls, [1, 1, 1]);

        let loaded_data = loaded.data().to_data();
        let loaded_slice: &[f32] = loaded_data.as_slice().expect("slice conversion failed");
        let diff = (loaded_slice[0] - original_val).abs();
        assert!(
            diff <= 5.0,
            "single pixel differs by {}: original={}, loaded={}",
            diff,
            original_val,
            loaded_slice[0]
        );
    }
}
