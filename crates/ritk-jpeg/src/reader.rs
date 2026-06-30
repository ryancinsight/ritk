use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

/// Read a JPEG file into a 3-D grayscale `Image` with shape `[1, height, width]`.
///
/// The image is converted to Luma8. Pixel intensities are stored as `f32` values
/// in `[0.0, 255.0]` with no normalization to `[0, 1]`.
pub fn read_jpeg<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let DecodedJpeg { data, dims } = decode_jpeg(path)?;
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), device);
    Ok(Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    ))
}

/// Read a JPEG file into a Coeus-backed 3-D grayscale image on `backend`.
///
/// The Atlas-tensor counterpart to [`read_jpeg`]: shares Luma8 decoding via
/// `decode_jpeg`, differing only in the final image construction.
#[cfg(feature = "coeus")]
pub fn read_jpeg_coeus<B, P>(path: P, backend: &B) -> Result<ritk_image::coeus::Image<f32, B, 3>>
where
    B: coeus_core::ComputeBackend,
    P: AsRef<Path>,
{
    let DecodedJpeg { data, dims } = decode_jpeg(path)?;
    ritk_image::coeus::Image::from_flat_on(
        data,
        dims,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        backend,
    )
}

/// Backend-agnostic decoded grayscale JPEG: Luma8 voxels as `f32` in
/// `[0.0, 255.0]` plus the `[1, height, width]` shape. Shared by the Burn and
/// Coeus reader paths.
struct DecodedJpeg {
    data: Vec<f32>,
    dims: [usize; 3],
}

fn decode_jpeg<P: AsRef<Path>>(path: P) -> Result<DecodedJpeg> {
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

    // Luma8 storage is already row-major `[height][width]`, matching the
    // `[1, height, width]` tensor layout, so the raw buffer converts directly
    // without per-pixel bounds-checked indexing.
    let data: Vec<f32> = img.into_raw().into_iter().map(f32::from).collect();

    Ok(DecodedJpeg {
        data,
        dims: [1, height as usize, width as usize],
    })
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
