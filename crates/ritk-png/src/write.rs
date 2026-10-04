//! PNG grayscale writing.
//!
//! # Contract
//! Every entry point here round-trips through `ritk-png`'s own reader: a volume
//! written by [`write_png_volume`] reads back through [`read_png_series`] with
//! the same shape and the same sample values, because PNG stores 8-bit
//! grayscale and the reader scales by exactly that.
//!
//! # Why values are normalised, not truncated
//!
//! PNG's grayscale format is 8-bit unsigned. An `Image<f32>` can hold any
//! modality value, including negatives, so writing one requires choosing a
//! stored range. This module uses the same min/max rule as the DICOM writers --
//! map `[min, max]` onto `[0, 255]` -- and records nothing about it, because PNG
//! has no rescale tags to record it in. That is lossy in a way the DICOM path is
//! not, and it is stated here rather than hidden: a reader of a written PNG gets
//! back the *ranks* and the shape, not the original scale. Callers who need the
//! original values in the file's own units must pre-window the image.

use anyhow::{bail, Context, Result};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use image::{GrayImage, Luma};
use ritk_image::Image;
use std::path::Path;

/// Writes `[1, rows, cols]` to `path` as a single 8-bit grayscale PNG.
///
/// # Errors
///
/// Returns an error when the image is not `[1, rows, cols]`, when any dimension
/// is zero, or when the file cannot be written.
pub fn write_png<B, P>(image: &Image<f32, B, 3>, path: P) -> Result<()>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let [depth, rows, cols] = image.shape();
    if depth != 1 {
        bail!(
            "write_png expects a single slice shaped [1, rows, cols]; got depth {depth}. \
             Use write_png_volume for a sequential volume."
        );
    }
    if rows == 0 || cols == 0 {
        bail!("write_png: shape [1, {rows}, {cols}] must have rows and cols > 0");
    }
    // Writes the file directly rather than delegating to the volume writer:
    // that one calls `create_dir_all`, which on the single-slice path would
    // create a *directory* at `path` and leave the reader unable to open it.
    let path = path.as_ref();
    encode_png_slice(image, 0)?
        .save(path)
        .with_context(|| format!("write PNG slice to {path:?}"))
}

/// Writes `[depth, rows, cols]` to `directory` as `slice-000.png`-style files,
/// naturally sorted so that `read_png_series` reads them back in order.
///
/// # Errors
///
/// Returns an error when any dimension is zero, when `directory` cannot be
/// created, or when a file cannot be written.
pub fn write_png_volume<B, P>(image: &Image<f32, B, 3>, directory: P) -> Result<()>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let [depth, rows, cols] = image.shape();
    if depth == 0 || rows == 0 || cols == 0 {
        bail!("write_png_volume: shape [{depth}, {rows}, {cols}] must have every dimension > 0");
    }
    let directory = directory.as_ref();
    std::fs::create_dir_all(directory)
        .with_context(|| format!("create PNG series directory {directory:?}"))?;

    let pixels = image
        .data_slice()
        .context("PNG writing requires contiguous f32 image data")?;
    let (minimum, maximum) = window(pixels);
    let per_slice = rows * cols;
    for (index, slice) in pixels.chunks_exact(per_slice).enumerate() {
        let gray = GrayImage::from_fn(cols as u32, rows as u32, |x, y| {
            let offset = y as usize * cols + x as usize;
            Luma([scale(slice[offset], minimum, maximum)])
        });
        let path = directory.join(slice_filename(index));
        gray.save(&path)
            .with_context(|| format!("write PNG slice {index} to {path:?}"))?;
    }
    Ok(())
}

/// Builds the in-memory grayscale buffer for one slice, for callers that encode
/// themselves (a montage, a tiled atlas, an animation).
pub fn encode_png_slice<B>(image: &Image<f32, B, 3>, index: usize) -> Result<GrayImage>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let [depth, rows, cols] = image.shape();
    let per_slice = rows * cols;
    if index >= depth {
        bail!("PNG slice {index} is out of range for depth {depth}");
    }
    let pixels = image
        .data_slice()
        .context("PNG writing requires contiguous f32 image data")?;
    let (minimum, maximum) = window(pixels);
    let slice = &pixels[index * per_slice..(index + 1) * per_slice];
    Ok(GrayImage::from_fn(cols as u32, rows as u32, |x, y| {
        let offset = y as usize * cols + x as usize;
        Luma([scale(slice[offset], minimum, maximum)])
    }))
}

/// The value window the writer normalises by, exposed so a caller that writes a
/// single slice differently can match the series exactly.
pub fn window_bounds<B: ComputeBackend>(image: &Image<f32, B, 3>) -> Result<(f32, f32)>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    Ok(window(image.data_slice().context(
        "PNG writing requires contiguous f32 image data",
    )?))
}

/// Zero-width and constant images both need a window that is not degenerate:
/// dividing by zero produces NaN, and every NaN clamps to zero, so a constant
/// image would come back as an all-black slice.
fn window(pixels: &[f32]) -> (f32, f32) {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for &value in pixels {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        return (0.0, 255.0);
    }
    (minimum, maximum.max(minimum + f32::EPSILON))
}

fn scale(value: f32, minimum: f32, maximum: f32) -> u8 {
    let normalized = (value - minimum) / (maximum - minimum);
    // Round half away from zero so a symmetric distribution does not bias every
    // tie toward zero, then clamp: a non-finite sample must not become a NaN
    // that `as u8` would turn into zero without saying so.
    if !normalized.is_finite() {
        return if normalized.is_sign_negative() {
            0
        } else {
            u8::MAX
        };
    }
    ((normalized * f32::from(u8::MAX)).round()).clamp(0.0, f32::from(u8::MAX)) as u8
}

/// Zero-padded so a natural sort returns slice order past nine slices.
fn slice_filename(index: usize) -> String {
    format!("slice-{index:04}.png")
}

#[cfg(test)]
mod tests;
