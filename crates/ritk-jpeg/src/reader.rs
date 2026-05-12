use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::path::Path;

/// Read a JPEG file into a 3-D grayscale `Image` with shape `[1, height, width]`.
///
/// The image is converted to Luma8. Pixel intensities are stored as `f32` values
/// in `[0.0, 255.0]` with no normalization to `[0, 1]`.
pub fn read_jpeg<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let path = path.as_ref();

    let img = image::open(path)
        .with_context(|| format!("Failed to open JPEG file: {}", path.display()))?
        .to_luma8();

    let (width, height) = img.dimensions();

    tracing::debug!(
        path = %path.display(),
        width = width,
        height = height,
        dtype = "Luma8",
        "Read JPEG image"
    );

    let num_pixels = height as usize * width as usize;
    let mut data: Vec<f32> = Vec::with_capacity(num_pixels);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y).0[0];
            data.push(pixel as f32);
        }
    }

    let shape = Shape::new([1_usize, height as usize, width as usize]);
    let tensor_data = TensorData::new(data, shape);
    let tensor = Tensor::<B, 3>::from_data(tensor_data, device);

    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();

    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Device-bound JPEG reader.
pub struct JpegReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> JpegReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<B, 3>> {
        read_jpeg(path, &self.device)
    }
}
