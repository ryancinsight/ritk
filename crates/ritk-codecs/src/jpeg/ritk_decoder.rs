//! RITK-owned JPEG decoder implementing `JpegDecodeBackend`.
//!
//! Routes based on the SOF marker encountered in the bitstream:
//! - SOF0/SOF1 → Baseline/Extended sequential DCT decode (`scan_dct`)
//! - SOF3      → Lossless Huffman prediction decode (`scan_lossless`)

use anyhow::{bail, Result};

use super::backend::{JpegDecodeBackend, JpegDecoded};
use super::marker::{parse_jpeg, SOF0, SOF1, SOF3};
use super::{scan_dct, scan_lossless};

/// RITK-owned JPEG decoder. Implements the sealed `JpegDecodeBackend` trait.
/// Replaces the external `jpeg-decoder` crate dependency.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RitkJpegDecoder;

impl super::backend::private::Sealed for RitkJpegDecoder {}

impl JpegDecodeBackend for RitkJpegDecoder {
    fn decode(fragment: &[u8]) -> Result<JpegDecoded> {
        let frame = parse_jpeg(fragment)?;
        let entropy = &fragment[frame.scan_data_start..];
        match frame.sof.sof_marker {
            SOF0 | SOF1 => scan_dct::decode_baseline_scan(&frame, entropy),
            SOF3 => scan_lossless::decode_lossless_scan(&frame, entropy),
            other => bail!("unsupported JPEG SOF marker: 0x{other:04X}"),
        }
    }
}
