//! JPEG frame decoding for encapsulated DICOM fragments.
//!
//! # Contract
//! The JPEG decoder produces integer sample values in image raster order. RITK
//! validates that the decoded raster shape and sample representation match the
//! DICOM metadata, interprets signed samples using the precision declared by
//! the JPEG frame, then applies the same linear modality LUT used by native
//! uncompressed pixel data: `output = sample * slope + intercept`. RGB output
//! is preserved as interleaved full-precision samples in raster order.
//!
use anyhow::{bail, Context, Result};
use consus_raster::{jpeg, Compression, DecodeLimits, PixelFormat};

use crate::pixel_layout::{decode_compressed_samples, encode_gray_u8_samples, PixelSignedness};
use crate::PixelLayout;

/// Decode one JPEG fragment using the supplied DICOM pixel layout.
///
/// The encoded raster is decoded without applying display-orientation
/// metadata. Its dimensions, component count, and sample width must match the
/// DICOM layout. Signed interpretation uses the JPEG frame precision rather
/// than the storage container width, and the modality rescale is then applied
/// from `layout`. A single DICOM even-length zero pad after the terminal JPEG
/// EOI marker is accepted; other trailing data is rejected.
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
    if !matches!(decoded.compression(), Compression::Lossless)
        && layout.pixel_representation.is_signed()
    {
        bail!("lossy JPEG DCT does not support signed DICOM pixel representation");
    }
    validate_jpeg_layout(
        decoded.width(),
        decoded.height(),
        decoded.format(),
        decoded.sample_precision(),
        decoded.pixels().len(),
        layout,
    )?;

    match decoded.format() {
        PixelFormat::Gray | PixelFormat::Rgb => {
            decode_jpeg_samples(decoded.pixels(), decoded.sample_precision(), 1, layout)
        }
        PixelFormat::GrayWide | PixelFormat::RgbWide => {
            decode_jpeg_samples(decoded.pixels(), decoded.sample_precision(), 2, layout)
        }
        _ => bail!("JPEG decoder returned an unsupported pixel format"),
    }
}

/// Encode one frame as a DICOM encapsulated baseline JPEG fragment.
///
/// This is the write half of [`decode_jpeg_fragment`]: the same `layout` that
/// describes how to read a frame back describes how to write it. Samples arrive
/// in the modality domain and are mapped back through the inverse rescale, so a
/// frame written and re-read returns the values it started with up to the lossy
/// DCT step.
///
/// Only eight-bit unsigned grayscale is encodable, because that is what the
/// pure-Rust provider encodes; every other DICOM JPEG flavour the reader accepts
/// (RGB, 12/16-bit, signed) has no encoder here and is rejected rather than
/// silently narrowed.
///
/// A DICOM encapsulated fragment must have even length, so a single pad byte is
/// appended when the codestream is odd. The reader removes that padding byte
/// before decoding.
///
/// # Errors
///
/// Returns an error when `layout` is not eight-bit unsigned grayscale, when the
/// sample count does not match `layout`, when the rescale is not invertible, or
/// when the provider rejects the request.
pub fn encode_jpeg_fragment(samples: &[f32], layout: PixelLayout, quality: u8) -> Result<Vec<u8>> {
    validate_encodable_layout(layout)?;
    let pixels = encode_gray_u8_samples(samples.iter().copied(), layout)?;
    let width = u32::try_from(layout.cols).context("DICOM JPEG columns exceed u32")?;
    let height = u32::try_from(layout.rows).context("DICOM JPEG rows exceed u32")?;
    let mut fragment = jpeg::encode_gray(&pixels, width, height, quality)
        .context("failed to encode DICOM JPEG fragment")?;
    if !fragment.len().is_multiple_of(2) {
        fragment.push(0);
    }
    Ok(fragment)
}

fn validate_encodable_layout(layout: PixelLayout) -> Result<()> {
    layout.pixels_per_frame()?;
    layout.validate_rescale_parameters()?;
    if layout.samples_per_pixel != 1 {
        bail!(
            "JPEG baseline encoding is grayscale only; layout declares {} samples per pixel",
            layout.samples_per_pixel
        );
    }
    if layout.bits_allocated != 8 {
        bail!(
            "JPEG baseline encoding is eight-bit only; layout allocates {} bits",
            layout.bits_allocated
        );
    }
    if layout.bits_stored != 8 {
        bail!(
            "JPEG baseline encoding is eight-bit only; layout stores {} bits",
            layout.bits_stored
        );
    }
    if layout.pixel_representation != PixelSignedness::Unsigned {
        bail!("JPEG baseline encoding cannot represent signed DICOM samples");
    }
    Ok(())
}

fn strip_dicom_padding(fragment: &[u8]) -> &[u8] {
    if fragment.len().is_multiple_of(2) && fragment.ends_with(&[0xFF, 0xD9, 0x00]) {
        &fragment[..fragment.len() - 1]
    } else {
        fragment
    }
}

fn decode_limits(fragment: &[u8], layout: PixelLayout) -> Result<DecodeLimits> {
    layout.bytes_per_frame()?;
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
    sample_precision: u8,
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
    let (expected_samples_per_pixel, bytes_per_sample, precision_range) = match pixel_format {
        PixelFormat::Gray => (1, 1, 2..=8),
        PixelFormat::GrayWide => (1, 2, 9..=16),
        PixelFormat::Rgb => (3, 1, 8..=8),
        PixelFormat::RgbWide => (3, 2, 12..=12),
        _ => bail!("JPEG decoder returned an unsupported pixel format"),
    };
    if !precision_range.contains(&sample_precision) {
        bail!(
            "JPEG decoded format {:?} is incompatible with sample precision {}",
            pixel_format,
            sample_precision
        );
    }
    if layout.bits_stored != u16::from(sample_precision) {
        bail!(
            "JPEG sample precision {} does not match DICOM BitsStored={}",
            sample_precision,
            layout.bits_stored
        );
    }
    if layout.samples_per_pixel != expected_samples_per_pixel {
        bail!(
            "JPEG decoded format {:?} requires samples_per_pixel={}; layout declares {}",
            pixel_format,
            expected_samples_per_pixel,
            layout.samples_per_pixel
        );
    }

    let expected_bytes = layout
        .samples_per_frame()?
        .checked_mul(bytes_per_sample)
        .context("JPEG decoded byte length overflow")?;
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

fn decode_jpeg_samples(
    bytes: &[u8],
    sample_precision: u8,
    bytes_per_sample: usize,
    layout: PixelLayout,
) -> Result<Vec<f32>> {
    if !matches!(bytes_per_sample, 1 | 2) {
        bail!(
            "JPEG sample representation precision={} bytes_per_sample={} is unsupported",
            sample_precision,
            bytes_per_sample
        );
    }
    if !bytes.len().is_multiple_of(bytes_per_sample) {
        bail!(
            "JPEG decoder returned {} bytes, not divisible by bytes_per_sample={}",
            bytes.len(),
            bytes_per_sample
        );
    }
    let samples = bytes.chunks_exact(bytes_per_sample).map(|sample| {
        if bytes_per_sample == 1 {
            u16::from(sample[0])
        } else {
            u16::from_ne_bytes([sample[0], sample[1]])
        }
    });
    decode_compressed_samples(samples, sample_precision, layout)
}

#[cfg(test)]
#[path = "tests_jpeg_decode.rs"]
mod tests_decode;

#[cfg(test)]
#[path = "tests_jpeg_encode.rs"]
mod tests_encode;
