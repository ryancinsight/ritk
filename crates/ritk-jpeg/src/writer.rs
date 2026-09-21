use anyhow::{Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use std::fs::File;
use std::io::Write;
use std::path::Path;

// Preserve the quality used by image 0.24's JPEG file writer, which previously
// owned this public operation.
const JPEG_QUALITY: u8 = 75;

/// Writes a native grayscale image with shape `[1, height, width]` as JPEG.
///
/// Values are rounded, clamped to `[0, 255]`, and encoded as Luma8.
///
/// # Errors
///
/// Returns an error for a non-planar image, inconsistent dimensions, an image
/// too large for JPEG dimension fields, encoding failure, or file I/O failure.
pub fn write_jpeg<B, P>(path: P, image: &Image<f32, B, 3>, backend: &B) -> Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let voxels = image.data_cow_on(backend);
    write_jpeg_flat(path.as_ref(), image.shape(), &voxels)
}

fn write_jpeg_flat(path: &Path, shape: [usize; 3], values: &[f32]) -> Result<()> {
    let [depth, height, width] = shape;
    if depth != 1 {
        anyhow::bail!("JPEG only supports 2-D images (depth=1), got depth={depth}");
    }
    let expected = height
        .checked_mul(width)
        .context("JPEG dimensions overflow the host address space")?;
    if values.len() != expected {
        anyhow::bail!(
            "JPEG voxel count {} does not match shape {shape:?}",
            values.len()
        );
    }
    let width_u32 = u32::try_from(width).context("JPEG width exceeds u32")?;
    let height_u32 = u32::try_from(height).context("JPEG height exceeds u32")?;
    let pixels: Vec<u8> = values
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect();
    let encoded = consus_raster::jpeg::encode_gray(&pixels, width_u32, height_u32, JPEG_QUALITY)
        .context("failed to encode JPEG grayscale image")?;
    tracing::debug!(width, height, path = %path.display(), "write JPEG grayscale image");
    let mut file =
        File::create(path).with_context(|| format!("failed to create JPEG: {}", path.display()))?;
    file.write_all(&encoded)
        .with_context(|| format!("failed to write JPEG: {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush JPEG: {}", path.display()))
}

/// Backend-bound native JPEG writer.
pub struct JpegWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> JpegWriter<B> {
    /// Creates a writer that extracts image data through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B> JpegWriter<B>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    /// Writes a grayscale JPEG through the configured backend.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`write_jpeg`].
    pub fn write_image<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> Result<()> {
        write_jpeg(path, image, &self.backend)
    }
}
