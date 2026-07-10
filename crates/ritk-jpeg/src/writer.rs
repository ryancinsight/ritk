use anyhow::{Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use image::GrayImage;
use ritk_image::native::Image;
use std::path::Path;

/// Writes a native grayscale image with shape `[1, height, width]` as JPEG.
///
/// Values are rounded, clamped to `[0, 255]`, and encoded as Luma8.
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
    let image = GrayImage::from_raw(width_u32, height_u32, pixels)
        .context("validated JPEG dimensions did not match the pixel buffer")?;
    tracing::debug!(width, height, path = %path.display(), "write JPEG grayscale image");
    image
        .save(path)
        .with_context(|| format!("failed to save JPEG: {}", path.display()))
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
    pub fn write_image<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> Result<()> {
        write_jpeg(path, image, &self.backend)
    }
}
