//! Native JPEG-LS (ISO 14495-1) codec for DICOM encapsulated frames (lossless and near-lossless).
//!
//! # Architecture
//! - `bitstream`: bit-level reader and writer with JPEG-LS 0xFF stuffing.
//! - `context`: ISO 14495-1 context model and threshold computation.
//! - `scan`: ISO 14495-1 regular-mode and run-mode scan decoder.
//! - [`encoder`]: ISO 14495-1 encoder, lossless and near-lossless.
//! - `decoder`: header-derived decoder state and scan-to-byte conversion.
//! - `parser`: marker parsing for SOI, SOF55, SOS, LSE, DRI, DNL, and EOI.

mod bitstream;
mod context;
mod decoder;
pub mod encoder;
mod marker;
mod parser;
mod scan;

pub(crate) use decoder::{ComponentInfo, JpegLsDecoder};
pub(crate) use marker::{DNL, DRI, EOI, LSE, SOF55, SOI, SOS};
use parser::{find_scan_data, parse_jpeg_ls_headers};

use anyhow::{bail, Context, Result};

use crate::{decode_native_pixel_bytes_checked, PixelLayout};

#[cfg(test)]
mod tests;

/// Decode a JPEG-LS encapsulated DICOM frame.
///
/// `fragment` is the complete JPEG-LS frame byte stream from SOI through EOI.
/// `layout` is the DICOM pixel layout contract used for final native-byte
/// conversion and modality LUT application.
pub fn decode_jpeg_ls_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    let mut decoder = JpegLsDecoder::new();
    parse_jpeg_ls_headers(&mut decoder, fragment).context("Failed to parse JPEG-LS headers")?;

    if decoder.width != layout.cols || decoder.height != layout.rows {
        bail!(
            "JPEG-LS dimensions {}x{} do not match DICOM layout {}x{}",
            decoder.width,
            decoder.height,
            layout.cols,
            layout.rows
        );
    }

    let scan_data = find_scan_data(fragment)
        .context("JPEG-LS scan data not found (SOS marker missing or truncated)")?;
    let decoded_bytes = decoder
        .decode_fragment(scan_data)
        .context("JPEG-LS decode failed")?;

    decode_native_pixel_bytes_checked(&decoded_bytes, layout)
}
