//! RITK-owned JPEG decoder implementing `JpegDecodeBackend`.
//!
//! Routes based on the SOF marker encountered in the bitstream:
//! - SOF0/SOF1 â†’ Baseline/Extended sequential DCT decode (`scan_dct`)
//! - SOF3      â†’ Lossless Huffman prediction decode (`scan_lossless`)

use anyhow::{bail, Context, Result};

use super::backend::{JpegDecodeBackend, JpegDecoded};
use super::marker::{parse_jpeg, SOF0, SOF1, SOF3};
use super::{scan_dct, scan_lossless};
use crate::dimensions::checked_pixel_count;

/// RITK-owned JPEG decoder. Implements the sealed `JpegDecodeBackend` trait.
/// Replaces the external `jpeg-decoder` crate dependency.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RitkJpegDecoder;

impl super::backend::private::Sealed for RitkJpegDecoder {}

impl JpegDecodeBackend for RitkJpegDecoder {
    fn decode(fragment: &[u8]) -> Result<JpegDecoded> {
        let frame = parse_jpeg(fragment)?;
        // Bound the decode against a hostile/corrupt SOF before the scan
        // allocates per-pixel buffers from `width Ã— height` (each a u16 field).
        checked_pixel_count(frame.sof.width as usize, frame.sof.height as usize)
            .context("JPEG frame dimensions")?;
        let entropy = &fragment[frame.scan_data_start..];
        match frame.sof.sof_marker {
            SOF0 | SOF1 => scan_dct::decode_baseline_scan(&frame, entropy),
            SOF3 => scan_lossless::decode_lossless_scan(&frame, entropy),
            other => bail!("unsupported JPEG SOF marker: 0x{other:04X}") }
    }
}
