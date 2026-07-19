//! Native JPEG frame decoding for encapsulated DICOM fragments.
//!
//! # Contract
//! The JPEG decoder produces integer sample values in image raster order. RITK
//! validates that the decoded raster shape and sample representation match the
//! DICOM metadata, then applies the same linear modality LUT used by native
//! uncompressed pixel data: `output = sample * slope + intercept`. RGB24 output
//! is preserved as interleaved samples in raster order.
//!
//! # Backend boundary
//! `backend::JpegDecodeBackend` is a sealed, static-dispatch boundary.
//! `RitkJpegDecoder` is the authoritative implementation; all external
//! `jpeg-decoder` crate dependency has been removed.

mod backend;
pub(crate) mod color;
pub(crate) mod constants;
pub(crate) mod huffman;
pub(crate) mod idct;
pub(crate) mod marker;
pub(crate) mod ritk_decoder;
pub(crate) mod scan_dct;
pub(crate) mod scan_lossless;

use anyhow::{bail, Context, Result};

use self::backend::{JpegDecodeBackend, JpegPixelFormat};
use self::ritk_decoder::RitkJpegDecoder;
use crate::{decode_native_pixel_bytes_checked, PixelLayout};

pub fn decode_jpeg_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    decode_jpeg_fragment_with::<RitkJpegDecoder>(fragment, layout)
}

#[inline]
fn decode_jpeg_fragment_with<B: JpegDecodeBackend>(
    fragment: &[u8],
    layout: PixelLayout,
) -> Result<Vec<f32>> {
    let decoded = B::decode(fragment)?;
    validate_jpeg_layout(
        decoded.width,
        decoded.height,
        decoded.pixel_format,
        decoded.pixels.len(),
        layout,
    )?;

    match decoded.pixel_format {
        JpegPixelFormat::L8 | JpegPixelFormat::Rgb24 => {
            decode_native_pixel_bytes_checked(&decoded.pixels, layout)
        }
        JpegPixelFormat::L16 => decode_l16_native_endian(&decoded.pixels, layout),
    }
}

fn validate_jpeg_layout(
    width: usize,
    height: usize,
    pixel_format: JpegPixelFormat,
    decoded_len: usize,
    layout: PixelLayout,
) -> Result<()> {
    if width != layout.cols || height != layout.rows {
        bail!(
            "JPEG dimensions {}x{} do not match DICOM layout {}x{}",
            width,
            height,
            layout.cols,
            layout.rows
        );
    }
    let expected_samples_per_pixel = pixel_format.samples_per_pixel();
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
        .checked_mul(pixel_format.pixel_bytes())
        .context("JPEG decoded byte length overflow")?;
    if layout.bits_allocated != pixel_format.bits_allocated() {
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

fn decode_l16_native_endian(bytes: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    layout.validate_rescale_parameters()?;
    if !bytes.len().is_multiple_of(2) {
        bail!("L16 JPEG decoder returned odd byte length {}", bytes.len());
    }
    let pixels = bytes
        .chunks_exact(2)
        .map(|sample| match layout.pixel_representation {
            crate::PixelSignedness::Signed => i16::from_ne_bytes([sample[0], sample[1]]) as f32,
            crate::PixelSignedness::Unsigned => u16::from_ne_bytes([sample[0], sample[1]]) as f32,
        })
        .map(|sample| sample * layout.rescale_slope + layout.rescale_intercept)
        .collect();
    Ok(pixels)
}

#[cfg(test)]
#[path = "tests_jpeg_decode.rs"]
mod tests;
