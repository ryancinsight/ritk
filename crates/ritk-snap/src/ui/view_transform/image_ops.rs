//! Applying a [`ViewTransform`] to RGBA storage.

use super::{RotationSteps, ViewTransform};
use anyhow::{anyhow, bail, Result};

/// Apply a view transform into caller-owned RGBA storage.
///
/// The input and output buffers are separate so a presentation frame can swap
/// them after the transform. The output capacity is retained across calls;
/// identity leaves it untouched and returns the source dimensions.
///
/// # Errors
/// Returns an error when `rgba` does not hold four bytes per pixel of `size`
/// or the output byte count overflows `usize`.
pub fn apply_to_rgba_into(
    size: [usize; 2],
    rgba: &[u8],
    transform: ViewTransform,
    output: &mut Vec<u8>,
) -> Result<[usize; 2]> {
    let [width, height] = size;
    validate_rgba_storage(size, rgba.len())?;
    if transform.is_identity() {
        return Ok(size);
    }

    let output_size = transform.output_size(size);
    let output_pixel_count = output_size[0]
        .checked_mul(output_size[1])
        .ok_or_else(|| anyhow!("RGBA transform output pixel count overflows usize"))?;
    let output_byte_count = output_pixel_count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("RGBA transform output byte count overflows usize"))?;
    output.resize(output_byte_count, 0);

    for (index, pixel) in rgba.chunks_exact(4).enumerate() {
        let row = index / width;
        let column = index % width;
        let row = if transform.flip_v {
            height - 1 - row
        } else {
            row
        };
        let column = if transform.flip_h {
            width - 1 - column
        } else {
            column
        };
        let (output_row, output_column) = match transform.rotation {
            RotationSteps::Zero => (row, column),
            RotationSteps::Ninety => (column, height - 1 - row),
            RotationSteps::OneEighty => (height - 1 - row, width - 1 - column),
            RotationSteps::TwoSeventy => (width - 1 - column, row),
        };
        let output_index = output_row
            .checked_mul(output_size[0])
            .and_then(|offset| offset.checked_add(output_column))
            .and_then(|pixel| pixel.checked_mul(4))
            .ok_or_else(|| anyhow!("RGBA transform output offset overflows usize"))?;
        let output_end = output_index
            .checked_add(4)
            .ok_or_else(|| anyhow!("RGBA transform output range overflows usize"))?;
        let destination = output
            .get_mut(output_index..output_end)
            .ok_or_else(|| anyhow!("RGBA transform output offset is outside storage"))?;
        destination.copy_from_slice(pixel);
    }

    Ok(output_size)
}

fn validate_rgba_storage(size: [usize; 2], length: usize) -> Result<()> {
    let [width, height] = size;
    if width == 0 || height == 0 {
        bail!("RGBA transform dimensions must be nonzero");
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| anyhow!("RGBA transform pixel count overflows usize"))?;
    let byte_count = pixel_count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("RGBA transform byte count overflows usize"))?;
    if length != byte_count {
        bail!(
            "RGBA transform byte count {} does not match {}x{} storage",
            length,
            width,
            height
        );
    }
    Ok(())
}
