use anyhow::{Context, Result};
use coeus_core::ComputeBackend;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

/// Read a JPEG file into a native 3-D grayscale image with shape `[1, height, width]`.
///
/// The image is converted to Luma8. Pixel intensities are stored as `f32`
/// values in `[0.0, 255.0]` without normalization.
pub fn read_jpeg<B, P>(path: P, backend: &B) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    P: AsRef<Path>,
{
    let DecodedJpeg { data, dims } = decode_jpeg(path)?;
    Image::from_flat_on(
        data,
        dims,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        backend,
    )
}

struct DecodedJpeg {
    data: Vec<f32>,
    dims: [usize; 3],
}

fn decode_jpeg<P: AsRef<Path>>(path: P) -> Result<DecodedJpeg> {
    let path = path.as_ref();
    let image = image::open(path)
        .with_context(|| format!("failed to open JPEG file: {}", path.display()))?
        .to_luma8();
    let (width, height) = image.dimensions();
    tracing::debug!(path = %path.display(), width, height, dtype = "Luma8", "read JPEG image");
    Ok(DecodedJpeg {
        data: image.into_raw().into_iter().map(f32::from).collect(),
        dims: [1, height as usize, width as usize],
    })
}

/// Backend-bound native JPEG reader.
pub struct JpegReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> JpegReader<B> {
    /// Creates a reader that constructs images on `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Reads a grayscale JPEG on the configured backend.
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<f32, B, 3>> {
        read_jpeg(path, &self.backend)
    }
}
