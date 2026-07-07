use anyhow::{Context, Result};
use image::{GrayImage, Luma};
use ritk_core::image::Image;
use ritk_image::tensor::backend::Backend;
use std::marker::PhantomData;
use std::path::Path;

/// Write a grayscale `Image<B, 3>` with shape `[1, height, width]` to a JPEG file.
///
/// Tensor values are rounded, clamped to `[0, 255]`, and encoded as Luma8.
pub fn write_jpeg<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let f32_vec = image.try_data_vec()?;
    write_jpeg_flat(path.as_ref(), image.shape(), &f32_vec)
}

/// Substrate-agnostic JPEG serialization core: the shared SSOT the Burn and
/// Atlas-native writers both wrap. Takes flat `[1, height, width]` voxels;
/// values are rounded, clamped to `[0, 255]`, and encoded as Luma8. JPEG
/// carries no physical-space metadata.
fn write_jpeg_flat(path: &Path, shape: [usize; 3], f32_slice: &[f32]) -> Result<()> {
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];

    if nz != 1 {
        anyhow::bail!("JPEG only supports 2D images (nz=1), got nz={}", nz);
    }

    tracing::debug!(
        nx = nx,
        ny = ny,
        path = %path.display(),
        "writing JPEG grayscale image"
    );

    let mut gray_img = GrayImage::new(nx as u32, ny as u32);

    for y in 0..ny {
        for x in 0..nx {
            let idx = y * nx + x;
            let val = f32_slice[idx].round().clamp(0.0, 255.0) as u8;
            gray_img.put_pixel(x as u32, y as u32, Luma([val]));
        }
    }

    gray_img
        .save(path)
        .with_context(|| format!("failed to save JPEG: {}", path.display()))?;

    Ok(())
}

/// Stateless JPEG writer.
pub struct JpegWriter<B: Backend> {
    _marker: PhantomData<fn() -> B>,
}

impl<B: Backend> Default for JpegWriter<B> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<B: Backend> JpegWriter<B> {
    pub fn write_image<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> Result<()> {
        write_jpeg(path, image)
    }
}

/// Atlas-native-substrate JPEG writers (plain end-state names, disambiguated
/// from the Burn functions by module path only; folds away when the Burn path
/// is deleted — ADR 0002 A1).
pub mod native {
    use super::write_jpeg_flat;
    use anyhow::Result;
    use std::path::Path;

    /// Write an Atlas-native grayscale `[1, height, width]` image to a JPEG file.
    ///
    /// Host data is extracted layout-independently via `data_cow_on`, then
    /// serialized through the same [`write_jpeg_flat`](super::write_jpeg_flat)
    /// core as the Burn [`write_jpeg`](super::write_jpeg) — byte-identical
    /// output for the same logical image.
    pub fn write_jpeg<B, P>(
        path: P,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> Result<()>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
        P: AsRef<Path>,
    {
        let voxels = image.data_cow_on(backend);
        write_jpeg_flat(path.as_ref(), image.shape(), &voxels)
    }
}
