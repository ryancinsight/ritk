//! JPEG frame decoding for encapsulated DICOM fragments.
//!
//! # Contract
//! The JPEG decoder produces integer sample values in image raster order. RITK
//! validates that the decoded raster shape and sample representation match the
//! DICOM metadata, then applies the same linear modality LUT used by native
//! uncompressed pixel data: `output = sample * slope + intercept`. RGB24 output
//! is preserved as interleaved samples in raster order.
//!
use anyhow::{bail, Context, Result};
use consus_raster::{jpeg, DecodeLimits, PixelFormat};

use crate::{decode_native_pixel_bytes_checked, PixelLayout};

/// Decode one JPEG fragment using the supplied DICOM pixel layout.
///
/// The encoded raster is decoded without applying display-orientation
/// metadata. Its dimensions, component count, and sample width must match the
/// DICOM layout. Signed interpretation and the modality rescale are then
/// applied from that layout. A single DICOM even-length zero pad after the
/// terminal JPEG EOI marker is accepted; other trailing data is rejected.
///
/// # Errors
///
/// Returns an error for malformed, unsupported, truncated, or over-limit JPEG
/// data, for a mismatch with `layout`, or for invalid rescale parameters.
pub fn decode_jpeg_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    let limits = decode_limits(fragment, layout)?;
    let codestream = strip_dicom_padding(fragment);
    let decoded =
        jpeg::decode(codestream, limits).context("failed to decode DICOM JPEG fragment")?;
    validate_jpeg_layout(
        decoded.width(),
        decoded.height(),
        decoded.format(),
        decoded.pixels().len(),
        layout,
    )?;

    match decoded.format() {
        PixelFormat::Gray | PixelFormat::Rgb => {
            decode_native_pixel_bytes_checked(decoded.pixels(), layout)
        }
        PixelFormat::GrayWide => decode_gray_wide(decoded.pixels(), layout),
        _ => bail!("JPEG decoder returned an unsupported pixel format"),
    }
}

fn strip_dicom_padding(fragment: &[u8]) -> &[u8] {
    if fragment.len().is_multiple_of(2) && fragment.ends_with(&[0xFF, 0xD9, 0x00]) {
        &fragment[..fragment.len() - 1]
    } else {
        fragment
    }
}

fn decode_limits(fragment: &[u8], layout: PixelLayout) -> Result<DecodeLimits> {
    let pixels = layout.pixels_per_frame()?;
    let width = u32::try_from(layout.cols).context("DICOM JPEG columns exceed u32")?;
    let height = u32::try_from(layout.rows).context("DICOM JPEG rows exceed u32")?;
    let max_working_bytes = jpeg::working_storage_bound(width, height)
        .context("failed to derive DICOM JPEG working storage bound")?;

    Ok(DecodeLimits {
        max_encoded_bytes: fragment.len().max(1),
        max_dimension: width.max(height).max(1),
        max_pixels: pixels,
        max_working_bytes,
    })
}

fn validate_jpeg_layout(
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    decoded_len: usize,
    layout: PixelLayout,
) -> Result<()> {
    let width = usize::try_from(width).context("JPEG width exceeds the host address space")?;
    let height = usize::try_from(height).context("JPEG height exceeds the host address space")?;
    if width != layout.cols || height != layout.rows {
        bail!(
            "JPEG dimensions {}x{} do not match DICOM layout {}x{}",
            width,
            height,
            layout.cols,
            layout.rows
        );
    }
    let (expected_samples_per_pixel, pixel_bytes, bits_allocated) = match pixel_format {
        PixelFormat::Gray => (1, 1, 8),
        PixelFormat::GrayWide => (1, 2, 16),
        PixelFormat::Rgb => (3, 3, 8),
        _ => bail!("JPEG decoder returned an unsupported pixel format"),
    };
    if layout.samples_per_pixel != expected_samples_per_pixel {
        bail!(
            "JPEG decoded format {:?} requires samples_per_pixel={}; layout declares {}",
            pixel_format,
            expected_samples_per_pixel,
            layout.samples_per_pixel
        );
    }

    let expected_bytes = layout
        .pixels_per_frame()?
        .checked_mul(pixel_bytes)
        .context("JPEG decoded byte length overflow")?;
    if layout.bits_allocated != bits_allocated {
        bail!(
            "JPEG decoded format {:?} is incompatible with DICOM BitsAllocated={}",
            pixel_format,
            layout.bits_allocated
        );
    }

    let expected_layout_bytes = layout.bytes_per_frame()?;
    if expected_bytes != expected_layout_bytes {
        bail!(
            "JPEG decoded byte length {} does not match DICOM layout byte length {}",
            expected_bytes,
            expected_layout_bytes
        );
    }
    if decoded_len != expected_bytes {
        bail!(
            "JPEG decoder returned {} bytes; expected {} bytes for decoded format {:?}",
            decoded_len,
            expected_bytes,
            pixel_format
        );
    }
    Ok(())
}

fn decode_gray_wide(bytes: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    layout.validate_rescale_parameters()?;
    if !bytes.len().is_multiple_of(2) {
        bail!(
            "wide grayscale JPEG decoder returned odd byte length {}",
            bytes.len()
        );
    }
    let pixels = bytes
        .chunks_exact(2)
        .map(|sample| match layout.pixel_representation {
            crate::PixelSignedness::Signed => f32::from(i16::from_ne_bytes([sample[0], sample[1]])),
            crate::PixelSignedness::Unsigned => {
                f32::from(u16::from_ne_bytes([sample[0], sample[1]]))
            }
        })
        .map(|sample| sample * layout.rescale_slope + layout.rescale_intercept)
        .collect();
    Ok(pixels)
}

#[cfg(test)]
#[path = "tests_jpeg_decode.rs"]
mod tests;
