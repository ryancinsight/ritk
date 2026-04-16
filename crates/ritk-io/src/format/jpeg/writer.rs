//! JPEG grayscale image writer.
//!
//! Encodes a 3-D `Image<B, 3>` with shape `[nz=1, ny, nx]` as an 8-bit
//! grayscale JPEG file.  The f32 tensor values are clamped to [0, 255],
//! rounded, and cast to `u8`.
//!
//! # Invariants
//! - `nz` must equal 1; JPEG is a 2-D format.
//! - Output format is inferred from the file extension (`.jpg` or `.jpeg`).

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use image::{GrayImage, Luma};
use ritk_core::image::Image;
use std::path::Path;

/// Write a grayscale `Image<B, 3>` with shape `[1, height, width]` to a JPEG file.
///
/// # Errors
/// - Returns an error if `nz != 1`.
/// - Returns an error if tensor data cannot be read as `f32`.
/// - Returns an error if the file cannot be written.
pub fn write_jpeg<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();
    let shape = image.shape(); // [nz, ny, nx]
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

    let data = image.data().clone().to_data();
    let slice = data
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to read tensor data as f32: {:?}", e))?;

    let mut gray_img = GrayImage::new(nx as u32, ny as u32);

    for y in 0..ny {
        for x in 0..nx {
            let idx = y * nx + x; // nz=1, so z-stride term is zero
            let val = slice[idx].round().clamp(0.0, 255.0) as u8;
            gray_img.put_pixel(x as u32, y as u32, Luma([val]));
        }
    }

    gray_img
        .save(path)
        .with_context(|| format!("failed to save JPEG: {}", path.display()))?;

    Ok(())
}
