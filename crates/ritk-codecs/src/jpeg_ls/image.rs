//! DICOM image contract for decoded JPEG-LS samples.

use anyhow::{bail, Context, Result};

use super::decoder::JpegLsDecoder;
use super::parser::parse_jpeg_ls_headers;
use crate::pixel_layout::decode_compressed_samples;
use crate::PixelLayout;

/// Decode a JPEG-LS encapsulated DICOM frame.
///
/// `fragment` is the complete JPEG-LS frame byte stream from SOI through EOI.
/// `layout` is the DICOM pixel layout contract used for final native-byte
/// conversion and modality LUT application.
pub fn decode_jpeg_ls_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    let mut decoder = JpegLsDecoder::new();
    let scan_data =
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
    if decoder.bits_per_sample != u32::from(layout.bits_stored) {
        bail!(
            "JPEG-LS precision {} does not match DICOM BitsStored={}",
            decoder.bits_per_sample,
            layout.bits_stored
        );
    }

    let decoded_bytes = decoder
        .decode_fragment(scan_data)
        .context("JPEG-LS decode failed")?;

    let sample_precision = u8::try_from(decoder.bits_per_sample)
        .context("JPEG-LS precision does not fit the supported sample representation")?;
    let bytes_per_sample = usize::from(sample_precision).div_ceil(8);
    let samples = decoded_bytes.chunks_exact(bytes_per_sample).map(|sample| {
        if bytes_per_sample == 1 {
            u16::from(sample[0])
        } else {
            u16::from_le_bytes([sample[0], sample[1]])
        }
    });
    decode_compressed_samples(samples, sample_precision, layout)
}
