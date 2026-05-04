//! Native JPEG-LS (ISO 14495-1) lossless decoder for DICOM encapsulated frames.
//!
//! # Architecture
//! The decoder is split into four focused sub-modules:
//! - [`bitstream`]: Bit-level reader with JPEG-LS 0xFF/0x00 stuffing-byte handling.
//! - [`context`]:   ISO 14495-1 §A context model, gradient quantization, threshold computation.
//! - [`scan`]:      ISO 14495-1 §A.3/§A.6 regular-mode and run-mode scan decoder.
//! - This module:   Header parsing (markers SOI, SOF55, SOS, LSE, DRI, DNL), public API.
//!
//! # Specification
//! JPEG-LS lossless (NEAR=0) decodes each sample by:
//! 1. Computing local gradients D1=d−b, D2=b−c, D3=c−a from the 4-neighbor causal context.
//! 2. Quantizing gradients to Q1, Q2, Q3 ∈ [−4,4] via thresholds T1, T2, T3.
//! 3. Entering run mode when Q1=Q2=Q3=0 (flat region).
//! 4. Sign-normalizing the context triplet and mapping to index q ∈ [0, 365).
//! 5. Computing predictor Px (edge-detecting) with bias correction C[q].
//! 6. Computing Golomb-Rice parameter k from A[q] and N[q].
//! 7. Decoding MErrval from the bitstream; inverse-mapping to errval.
//! 8. Reconstructing Rx = clamp(Px + sign·errval, 0, MAXVAL).
//! 9. Updating context A, B, C, N.

mod bitstream;
mod context;
mod scan;

pub(crate) use context::ContextState;

use anyhow::{bail, Context, Result};

use crate::{PixelLayout, decode_native_pixel_bytes_checked};
use bitstream::BitReader;
use scan::{Predictor, ScanParams, decode_scan};

// ─── JPEG-LS Markers ──────────────────────────────────────────────────────────

/// JPEG Start of Image marker (also used in JPEG-LS).
pub(crate) const SOI: u16 = 0xFFD8;
/// JPEG-LS Start of Frame marker (ISO 14495-1 §C.1.2: SOF55).
pub(crate) const SOF55: u16 = 0xFFF7;
/// Start of Scan marker.
pub(crate) const SOS: u16 = 0xFFDA;
/// Define Number of Lines marker.
pub(crate) const DNL: u16 = 0xFFDC;
/// Define Restart Interval marker.
pub(crate) const DRI: u16 = 0xFFDD;
/// JPEG-LS Application Specification (LSE) marker.
pub(crate) const LSE: u16 = 0xFFF0;
/// End of Image marker.
pub(crate) const EOI: u16 = 0xFFD9;

// ─── Prediction Modes (compatibility API) ─────────────────────────────────────

/// JPEG-LS predictor field values for DICOM interoperability.
///
/// Used in tests and compatibility checks; internally maps to [`scan::Predictor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Prediction {
    None       = 0,
    Left       = 1,
    Up         = 2,
    AvgLeftUp  = 3,
    Paeth      = 4,
}

impl Prediction {
    /// Parse a predictor byte from the SOS header.
    ///
    /// Valid values: 0–4 (DICOM PS 3.5 restricts to 0–4 for JPEG-LS Lossless).
    pub(crate) fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Left),
            2 => Ok(Self::Up),
            3 => Ok(Self::AvgLeftUp),
            4 => Ok(Self::Paeth),
            _ => bail!("Invalid JPEG-LS prediction mode: {}", v),
        }
    }
}

// ─── Component Info (test-visible type) ───────────────────────────────────────

/// Per-component decoder metadata (populated during SOF55 header parsing).
pub(crate) struct ComponentInfo {
    /// Component identifier byte from SOF55.
    #[allow(dead_code)]
    pub(crate) id: u8,
    /// Per-component predictor byte from SOS (kept for multi-component future use).
    #[allow(dead_code)]
    pub(crate) predictor: u8,
    /// Context states (365 per component); created fresh during decode.
    #[allow(dead_code)]
    pub(crate) context: [ContextState; 365],
}

// ─── JpegLsDecoder (header state container) ───────────────────────────────────

/// JPEG-LS decoder state populated by header parsing.
///
/// `decode_fragment()` reads these fields and dispatches to the ISO 14495-1 scan decoder.
pub(crate) struct JpegLsDecoder {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) bits_per_sample: u32,
    pub(crate) components: Vec<ComponentInfo>,
    pub(crate) prediction: Prediction,
    /// NEAR parameter (must be 0 for DICOM lossless JPEG-LS, DICOM PS 3.5 §8.2.3).
    pub(crate) near: u32,
    #[allow(dead_code)]
    pub(crate) restart_interval: u32,
    /// LSE-specified thresholds (0,0,0 = use defaults).
    t1: i32,
    t2: i32,
    t3: i32,
}

impl JpegLsDecoder {
    /// Create a decoder with default (uninitialized) fields.
    pub(crate) fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            bits_per_sample: 8,
            components: Vec::new(),
            prediction: Prediction::Left,
            near: 0,
            restart_interval: 0,
            t1: 0,
            t2: 0,
            t3: 0,
        }
    }

    /// Decode a JPEG-LS scan starting from `data` (scan data bytes, after SOS header).
    ///
    /// # Pre-conditions
    /// - `self.width > 0` and `self.height > 0` (set by SOF55 header)
    /// - `self.near == 0` (DICOM lossless constraint)
    /// - `self.components.len() == 1` (non-interleaved DICOM frame)
    ///
    /// # Returns
    /// Raw decoded bytes in DICOM pixel byte order (little-endian 16-bit for 16-bpp).
    pub(crate) fn decode_fragment(&self, data: &[u8]) -> Result<Vec<u8>> {
        if self.width == 0 || self.height == 0 {
            bail!("JPEG-LS fragment has invalid dimensions ({}×{})", self.width, self.height);
        }
        if self.near != 0 {
            bail!("JPEG-LS NEAR={} not supported (lossless only, DICOM PS 3.5 §8.2.3)", self.near);
        }
        if self.components.len() != 1 {
            bail!(
                "JPEG-LS multi-component ({}) not supported; use non-interleaved encoding",
                self.components.len()
            );
        }

        // Map Prediction → scan::Predictor
        let predictor = match self.prediction {
            Prediction::None      => Predictor::None,
            Prediction::Left      => Predictor::Left,
            Prediction::Up        => Predictor::Up,
            Prediction::AvgLeftUp => Predictor::UpPlusLeftMinusUpLeft,
            Prediction::Paeth     => Predictor::Adaptive,
        };

        let params = ScanParams {
            rows: self.height,
            cols: self.width,
            bpp: self.bits_per_sample,
            near: self.near,
            predictor,
            t1: self.t1,
            t2: self.t2,
            t3: self.t3,
        };

        let mut reader = BitReader::new(data);
        let mut samples = Vec::with_capacity(self.height * self.width);
        decode_scan(&mut reader, &params, &mut samples)
            .context("JPEG-LS scan decode failed")?;

        // Convert i32 samples to output bytes (little-endian)
        let bytes_per_sample = (self.bits_per_sample as usize + 7) / 8;
        let mut out = vec![0u8; samples.len() * bytes_per_sample];
        for (i, &s) in samples.iter().enumerate() {
            if bytes_per_sample == 1 {
                out[i] = s as u8;
            } else {
                let bytes = (s as u16).to_le_bytes();
                out[i * 2..i * 2 + 2].copy_from_slice(&bytes);
            }
        }
        Ok(out)
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Decode a JPEG-LS encapsulated DICOM frame.
///
/// # Arguments
/// * `fragment` — The complete JPEG-LS frame bytes (SOI … EOI).
/// * `layout`   — DICOM pixel metadata (rows, cols, bpp, rescale slope/intercept).
///
/// # Returns
/// Decoded float32 samples with modality LUT applied (via `decode_native_pixel_bytes_checked`).
pub fn decode_jpeg_ls_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    let mut decoder = JpegLsDecoder::new();
    parse_jpeg_ls_headers(&mut decoder, fragment)
        .context("Failed to parse JPEG-LS headers")?;

    if decoder.width != layout.cols || decoder.height != layout.rows {
        bail!(
            "JPEG-LS dimensions {}×{} do not match DICOM layout {}×{}",
            decoder.width, decoder.height, layout.cols, layout.rows
        );
    }

    let scan_data = find_scan_data(fragment)
        .context("JPEG-LS scan data not found (SOS marker missing or truncated)")?;

    let decoded_bytes = decoder
        .decode_fragment(scan_data)
        .context("JPEG-LS decode failed")?;

    decode_native_pixel_bytes_checked(&decoded_bytes, layout)
}

// ─── Header Parsing ───────────────────────────────────────────────────────────

/// Parse all JPEG-LS markers before the scan data, populating `decoder`.
///
/// Stops at the SOS marker (scan data is located separately via `find_scan_data`).
fn parse_jpeg_ls_headers(decoder: &mut JpegLsDecoder, data: &[u8]) -> Result<()> {
    if data.len() < 2 || u16::from_be_bytes([data[0], data[1]]) != SOI {
        bail!("JPEG-LS fragment does not start with SOI marker (0xFFD8)");
    }
    let mut pos = 2usize;

    while pos + 1 < data.len() {
        let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);

        if marker == EOI || marker == SOS {
            break;
        }

        match marker {
            SOI => {
                pos += 2;
                continue;
            }
            SOF55 => {
                if pos + 9 > data.len() {
                    bail!("Truncated SOF55 marker at offset {}", pos);
                }
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                decoder.bits_per_sample = data[pos + 4] as u32;
                decoder.height = u16::from_be_bytes([data[pos + 5], data[pos + 6]]) as usize;
                decoder.width  = u16::from_be_bytes([data[pos + 7], data[pos + 8]]) as usize;
                let num_comp = if pos + 9 < data.len() { data[pos + 9] } else { 1 };
                decoder.components.clear();
                for i in 0..(num_comp as usize) {
                    let idx = pos + 10 + i * 3;
                    if idx + 2 < data.len() {
                        decoder.components.push(ComponentInfo {
                            id: data[idx],
                            predictor: 0,
                            context: [ContextState::default(); 365],
                        });
                    }
                }
                pos += 2 + length;
            }
            DNL => {
                if pos + 6 > data.len() {
                    bail!("Truncated DNL marker at offset {}", pos);
                }
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                if pos + 4 + length <= data.len() && length >= 2 {
                    decoder.height = u16::from_be_bytes([data[pos + 4], data[pos + 5]]) as usize;
                }
                pos += 2 + length;
            }
            DRI => {
                if pos + 6 > data.len() {
                    bail!("Truncated DRI marker at offset {}", pos);
                }
                decoder.restart_interval =
                    u16::from_be_bytes([data[pos + 4], data[pos + 5]]) as u32;
                pos += 6;
            }
            LSE => {
                // JPEG-LS Preset Parameter Extension (ISO 14495-1 §C.2.4)
                if pos + 4 > data.len() {
                    bail!("Truncated LSE marker at offset {}", pos);
                }
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                if pos + 2 + length <= data.len() && length >= 12 {
                    // Preset ID=1: custom T1, T2, T3, RESET
                    let id = data[pos + 4];
                    if id == 1 {
                        decoder.t1 = u16::from_be_bytes([data[pos + 7], data[pos + 8]]) as i32;
                        decoder.t2 = u16::from_be_bytes([data[pos + 9], data[pos + 10]]) as i32;
                        decoder.t3 = u16::from_be_bytes([data[pos + 11], data[pos + 12]]) as i32;
                    }
                }
                pos += 2 + length;
            }
            _ => {
                // APP markers, COM, unknown: skip by length field
                if pos + 4 <= data.len() {
                    let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                    pos += 2 + length;
                } else {
                    pos += 2;
                }
            }
        }
    }

    // Parse SOS header to extract predictor and NEAR
    if pos + 1 < data.len() {
        let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
        if marker == SOS && pos + 4 < data.len() {
            let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            let ns = if pos + 4 < data.len() { data[pos + 4] as usize } else { 1 };
            // Per-component: Cs (1 byte), Ta|Tb (1 byte)
            // After Ns components: Ss (predictor), Se (NEAR), Ah|Al (point transform)
            let comp_end = pos + 5 + ns * 2;
            if comp_end + 3 <= data.len() {
                let ss = data[comp_end];     // Ss: predictor/ILV select
                let se = data[comp_end + 1]; // Se: NEAR parameter
                // Parse predictor (Ss ∈ [0..7])
                if let Ok(pred) = Prediction::from_u8(ss & 0x0F) {
                    decoder.prediction = pred;
                }
                decoder.near = se as u32;
            }
            pos += 2 + length; // move past SOS header (unused after this)
            let _ = pos;
        }
    }

    Ok(())
}

/// Find the scan data bytes immediately following the SOS header length field.
///
/// Scans for the SOS marker and skips the variable-length header.
fn find_scan_data(data: &[u8]) -> Option<&[u8]> {
    let mut pos = 0usize;
    while pos + 1 < data.len() {
        if data[pos] == 0xFF {
            let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
            if marker == SOS && pos + 3 < data.len() {
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                let scan_start = pos + 2 + length;
                if scan_start < data.len() {
                    return Some(&data[scan_start..]);
                }
                return None;
            }
        }
        pos += 1;
    }
    None
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jpeg_ls_marker_constants_correct() {
        assert_eq!(SOI,   0xFFD8);
        assert_eq!(SOF55, 0xFFF7);
        assert_eq!(SOS,   0xFFDA);
        assert_eq!(EOI,   0xFFD9);
    }

    #[test]
    fn prediction_mode_from_u8_valid() {
        assert!(matches!(Prediction::from_u8(0), Ok(Prediction::None)));
        assert!(matches!(Prediction::from_u8(1), Ok(Prediction::Left)));
        assert!(matches!(Prediction::from_u8(2), Ok(Prediction::Up)));
        assert!(matches!(Prediction::from_u8(3), Ok(Prediction::AvgLeftUp)));
        assert!(matches!(Prediction::from_u8(4), Ok(Prediction::Paeth)));
    }

    #[test]
    fn prediction_mode_from_u8_invalid() {
        assert!(Prediction::from_u8(5).is_err());
        assert!(Prediction::from_u8(255).is_err());
    }

    #[test]
    fn bit_reader_basic() {
        use bitstream::BitReader;
        let data = [0b10110000u8, 0b11001100u8];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bit(), 1);
        assert_eq!(reader.read_bit(), 0);
        assert_eq!(reader.read_bit(), 1);
        assert_eq!(reader.read_bit(), 1);
        assert_eq!(reader.read_bit(), 0);
        assert_eq!(reader.read_bit(), 0);
    }

    #[test]
    fn bit_reader_read_bits() {
        use bitstream::BitReader;
        let data = [0b10110000u8];
        let mut reader = BitReader::new(&data);
        let bits = reader.read_bits(3);
        assert_eq!(bits, 5); // 0b101
    }

    #[test]
    fn decoder_new_initializes_defaults() {
        let decoder = JpegLsDecoder::new();
        assert_eq!(decoder.width, 0);
        assert_eq!(decoder.height, 0);
        assert_eq!(decoder.bits_per_sample, 8);
        assert_eq!(decoder.near, 0);
    }

    #[test]
    fn decode_fragment_rejects_near_nonzero() {
        let decoder = JpegLsDecoder {
            width: 100,
            height: 100,
            bits_per_sample: 8,
            components: vec![ComponentInfo {
                id: 1,
                predictor: 0,
                context: [ContextState::default(); 365],
            }],
            prediction: Prediction::Left,
            near: 1,
            restart_interval: 0,
            t1: 0, t2: 0, t3: 0,
        };
        let result = decoder.decode_fragment(&[]);
        assert!(result.is_err());
        let msg = format!("{:?}", result.unwrap_err());
        assert!(msg.contains("NEAR"), "Expected 'NEAR' in error: {}", msg);
    }

    #[test]
    fn decode_fragment_rejects_zero_dimensions() {
        let decoder = JpegLsDecoder {
            width: 0,
            height: 100,
            bits_per_sample: 8,
            components: vec![ComponentInfo {
                id: 1,
                predictor: 0,
                context: [ContextState::default(); 365],
            }],
            prediction: Prediction::Left,
            near: 0,
            restart_interval: 0,
            t1: 0, t2: 0, t3: 0,
        };
        let result = decoder.decode_fragment(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_fragment_rejects_nonzero_near() {
        let decoder = JpegLsDecoder {
            width: 100,
            height: 100,
            bits_per_sample: 8,
            components: vec![ComponentInfo {
                id: 1,
                predictor: 0,
                context: [ContextState::default(); 365],
            }],
            prediction: Prediction::Left,
            near: 1,
            restart_interval: 0,
            t1: 0, t2: 0, t3: 0,
        };
        let result = decoder.decode_fragment(&[]);
        assert!(result.is_err());
        let msg = format!("{:?}", result.unwrap_err());
        assert!(msg.contains("NEAR"), "Expected 'NEAR' in error: {}", msg);
    }

    #[test]
    fn parse_jpeg_ls_headers_rejects_missing_soi() {
        let mut d = JpegLsDecoder::new();
        let bad_data = [0x00u8, 0x00u8];
        assert!(parse_jpeg_ls_headers(&mut d, &bad_data).is_err());
    }

    #[test]
    fn find_scan_data_returns_none_without_sos() {
        let data = [0xFF, 0xD8]; // SOI only, no SOS
        assert!(find_scan_data(&data).is_none());
    }

    #[test]
    fn find_scan_data_returns_bytes_after_sos_header() {
        // SOI + SOS with length=8 (6 bytes of header) + 3 bytes of "scan" data
        let data: &[u8] = &[
            0xFF, 0xD8,             // SOI
            0xFF, 0xDA,             // SOS
            0x00, 0x08,             // length = 8 (SOS header = 8 bytes)
            0x01,                   // Ns = 1
            0x01, 0x00,             // component 1, table
            0x01, 0x00, 0x00,       // Ss, Se, AhAl
            0xAB, 0xCD, 0xEF,       // scan data
        ];
        let result = find_scan_data(data);
        assert!(result.is_some());
        let sd = result.unwrap();
        assert_eq!(sd[0], 0xAB);
        assert_eq!(sd[1], 0xCD);
        assert_eq!(sd[2], 0xEF);
    }
}
