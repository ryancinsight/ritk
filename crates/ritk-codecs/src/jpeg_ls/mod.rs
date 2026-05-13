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

use crate::{decode_native_pixel_bytes_checked, PixelLayout};
use bitstream::BitReader;
use scan::{decode_scan, Predictor, ScanParams};

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
/// JPEG-LS preset parameter marker (ISO 14495-1 §C.2.4).
pub(crate) const LSE: u16 = 0xFFF8;
/// End of Image marker.
pub(crate) const EOI: u16 = 0xFFD9;

// ─── Component Info (test-visible type) ───────────────────────────────────────

/// Per-component decoder metadata (populated during SOF55 header parsing).
pub(crate) struct ComponentInfo {
    /// Component identifier byte from SOF55.
    #[allow(dead_code)]
    pub(crate) id: u8,
    /// Per-component mapping-table selector byte.
    #[allow(dead_code)]
    pub(crate) mapping_table_selector: u8,
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
    /// NEAR parameter (must be 0 for DICOM lossless JPEG-LS, DICOM PS 3.5 §8.2.3).
    pub(crate) near: u32,
    /// Interleave mode from the SOS header. Single-component scans require 0.
    pub(crate) interleave_mode: u8,
    /// Point transform byte from the SOS header. DICOM lossless image frames require 0.
    pub(crate) point_transform: u8,
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
            near: 0,
            interleave_mode: 0,
            point_transform: 0,
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
            bail!(
                "JPEG-LS fragment has invalid dimensions ({}×{})",
                self.width,
                self.height
            );
        }
        if self.near != 0 {
            bail!(
                "JPEG-LS NEAR={} not supported (lossless only, DICOM PS 3.5 §8.2.3)",
                self.near
            );
        }
        if self.components.len() != 1 {
            bail!(
                "JPEG-LS multi-component ({}) not supported; use non-interleaved encoding",
                self.components.len()
            );
        }
        if self.interleave_mode != 0 {
            bail!(
                "JPEG-LS interleave mode {} not supported for single-component DICOM frames",
                self.interleave_mode
            );
        }
        if self.point_transform != 0 {
            bail!(
                "JPEG-LS point transform {} not supported for DICOM lossless frames",
                self.point_transform
            );
        }

        let params = ScanParams {
            rows: self.height,
            cols: self.width,
            bpp: self.bits_per_sample,
            near: self.near,
            predictor: Predictor::Adaptive,
            t1: self.t1,
            t2: self.t2,
            t3: self.t3,
        };

        let mut reader = BitReader::new(data);
        let mut samples = Vec::with_capacity(self.height * self.width);
        decode_scan(&mut reader, &params, &mut samples).context("JPEG-LS scan decode failed")?;

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
    parse_jpeg_ls_headers(&mut decoder, fragment).context("Failed to parse JPEG-LS headers")?;

    if decoder.width != layout.cols || decoder.height != layout.rows {
        bail!(
            "JPEG-LS dimensions {}×{} do not match DICOM layout {}×{}",
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
                decoder.width = u16::from_be_bytes([data[pos + 7], data[pos + 8]]) as usize;
                let num_comp = if pos + 9 < data.len() {
                    data[pos + 9]
                } else {
                    1
                };
                decoder.components.clear();
                for i in 0..(num_comp as usize) {
                    let idx = pos + 10 + i * 3;
                    if idx + 2 < data.len() {
                        decoder.components.push(ComponentInfo {
                            id: data[idx],
                            mapping_table_selector: data[idx + 2],
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
                if pos + 2 + length <= data.len() && length >= 13 {
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

    // Parse SOS header to extract NEAR, interleave mode, and point transform.
    if pos + 1 < data.len() {
        let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
        if marker == SOS && pos + 4 < data.len() {
            let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            let ns = if pos + 4 < data.len() {
                data[pos + 4] as usize
            } else {
                1
            };
            // Per-component: Cs (1 byte), Ta|Tb (1 byte)
            // After Ns components: NEAR (1 byte), ILV (1 byte), Ah|Al (1 byte).
            let comp_end = pos + 5 + ns * 2;
            if comp_end + 3 <= data.len() {
                decoder.near = data[comp_end] as u32;
                decoder.interleave_mode = data[comp_end + 1];
                decoder.point_transform = data[comp_end + 2];
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
        assert_eq!(SOI, 0xFFD8);
        assert_eq!(SOF55, 0xFFF7);
        assert_eq!(SOS, 0xFFDA);
        assert_eq!(LSE, 0xFFF8);
        assert_eq!(EOI, 0xFFD9);
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
        assert_eq!(decoder.interleave_mode, 0);
        assert_eq!(decoder.point_transform, 0);
    }

    #[test]
    fn decode_fragment_rejects_near_nonzero() {
        let decoder = JpegLsDecoder {
            width: 100,
            height: 100,
            bits_per_sample: 8,
            components: vec![ComponentInfo {
                id: 1,
                mapping_table_selector: 0,
                context: [ContextState::default(); 365],
            }],
            near: 1,
            interleave_mode: 0,
            point_transform: 0,
            restart_interval: 0,
            t1: 0,
            t2: 0,
            t3: 0,
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
                mapping_table_selector: 0,
                context: [ContextState::default(); 365],
            }],
            near: 0,
            interleave_mode: 0,
            point_transform: 0,
            restart_interval: 0,
            t1: 0,
            t2: 0,
            t3: 0,
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
                mapping_table_selector: 0,
                context: [ContextState::default(); 365],
            }],
            near: 1,
            interleave_mode: 0,
            point_transform: 0,
            restart_interval: 0,
            t1: 0,
            t2: 0,
            t3: 0,
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
            0xFF, 0xD8, // SOI
            0xFF, 0xDA, // SOS
            0x00, 0x08, // length = 8 (SOS header = 8 bytes)
            0x01, // Ns = 1
            0x01, 0x00, // component 1, table
            0x00, 0x00, 0x00, // NEAR, ILV, Ah/Al
            0xAB, 0xCD, 0xEF, // scan data
        ];
        let result = find_scan_data(data);
        assert!(result.is_some());
        let sd = result.unwrap();
        assert_eq!(sd[0], 0xAB);
        assert_eq!(sd[1], 0xCD);
        assert_eq!(sd[2], 0xEF);
    }

    // ─── Positive conformance fixtures (ISO 14495-1 §A.3 / §A.6) ─────────────
    //
    // Scan data derivations are fully worked per ISO 14495-1 §A.3 (regular mode)
    // and §A.6 (run mode). a_init = max(2, (RANGE+32)>>6) = 4 for 8-bit images
    // (RANGE=256). k = floor(log2(A/N)), limit = 2*(bpp + max(bpp,2)) = 32 for
    // 8-bit. Golomb-Rice uses: (q leading zeros) + (stop 1) + (k-bit remainder),
    // MSB-first, counted from the MSB of the first scan byte.

    fn layout_8bit(rows: usize, cols: usize, slope: f32, intercept: f32) -> PixelLayout {
        PixelLayout {
            rows,
            cols,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: 0,
            rescale_slope: slope,
            rescale_intercept: intercept,
        }
    }

    /// Build a minimal single-component JPEG-LS 8-bit lossless frame.
    ///
    /// Frame layout:
    ///   SOI | SOF55(bpp,height,width,1 component) | SOS(NEAR=0,ILV=0) | scan_data | EOI
    fn build_jpeg_ls_frame(height: u16, width: u16, scan_data: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(29 + scan_data.len());
        // SOI
        frame.extend_from_slice(&[0xFF, 0xD8]);
        // SOF55: length=11, bpp=8, height, width, 1 component {id=1, sampling=0x11, quant=0}
        frame.extend_from_slice(&[0xFF, 0xF7, 0x00, 0x0B, 0x08]);
        frame.extend_from_slice(&height.to_be_bytes());
        frame.extend_from_slice(&width.to_be_bytes());
        frame.extend_from_slice(&[0x01, 0x01, 0x11, 0x00]);
        // SOS: length=8, Ns=1, comp_id=1, table=0, NEAR=0, ILV=0, Ah/Al=0.
        frame.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00]);
        // Scan data
        frame.extend_from_slice(scan_data);
        // EOI
        frame.extend_from_slice(&[0xFF, 0xD9]);
        frame
    }

    #[test]
    fn jpeg_ls_fragment_2x2_all_zero_decodes_correctly() {
        // 2×2 all-zero frame. Each row is encoded as a complete run of zero
        // samples. `0xF8` provides four run-hit bits followed by padding.
        let frame = build_jpeg_ls_frame(2, 2, &[0xF8]);
        let layout = layout_8bit(2, 2, 1.0, 0.0);
        let result = decode_jpeg_ls_fragment(&frame, layout).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result, vec![0.0f32, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn jpeg_ls_fragment_1x3_constant_value10_decodes_correctly() {
        // 1×3, 8-bit, samples = [10, 10, 10].
        //
        // Scan derivation (a_init=4, limit=32):
        //   (0,0): all causal neighbors 0 (sentinel) → D1=D2=D3=0 → run mode.
        //     run_val=a=0, remaining=3. value=10≠0 → run MISS.
        //     J[0]=0: read bit=0 (miss); no remainder bits; run_len=0.
        //     run_len=0 < remaining=3 → run interrupt at c=0.
        //     ri_ctx initial: a=4,n=1 → k=2. rb=ra=0, |rb-ra|=0<=NEAR=0, px=0.
        //     Need rx=10 → RItype=1, map=false, e_mapped=2×10−1=19.
        //     Golomb(19,k=2): q=19>>2=4 zeros; stop 1; rem=19&3=3 (2 bits "11").
        //     Bits: 0(miss) 0000(q=4) 1(stop) 11(rem) = 8 bits.
        //     run_index stays 0 (decrement on interrupt: 0>0 is false).
        //   (0,1): a=10,b=0,cc=0,d=0 → D3=cc-a=0-10=-10.
        //     quant(-10,3,7,21)=-3 (since -10≤-t2=-7). q3=-3≠0 → regular mode.
        //     sign_normalize(0,0,-3)=(0,0,3,sign=-1). qi=context_index(0,0,3)=3.
        //     row=0,col=1 → predict returns a=10. ctx.c=0 → px=10.
        //     rx=10 → errval=0 → errval_canon=0 → me=0.
        //     ctx.a=4,n=1 → k=2. Golomb(0,2): q=0 stop 1; rem=00. Bits: 1(stop) 00(rem)=3 bits.
        //     update_context(errval=0*sign=-1*0=0): a=4,n=2,b=0,c=0.
        //   (0,2): same gradients as (0,1), same context qi=3.
        //     ctx.a=4,n=2 → k=compute_k(4,2,8)=1. px=10, rx=10, me=0.
        //     Golomb(0,1): q=0 stop 1; rem=0 (1 bit). Bits: 1(stop) 0(rem) = 2 bits.
        //   Total: 8+3+2=13 bits. Padding: 3 zero bits.
        //   Bytes: [0b00000111, 0b10010000] = [0x07, 0x90].
        let frame = build_jpeg_ls_frame(1, 3, &[0x07, 0x90]);
        let layout = layout_8bit(1, 3, 1.0, 0.0);
        let result = decode_jpeg_ls_fragment(&frame, layout).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec![10.0f32, 10.0, 10.0]);
    }

    #[test]
    fn jpeg_ls_fragment_1x1_run_interrupt_with_modality_lut() {
        // 1×1, 8-bit, sample = 2; modality LUT: slope=2.0, intercept=-5.0.
        // Expected output: 2×2.0 + (−5.0) = −1.0.
        //
        // Scan derivation (a_init=4, limit=32):
        //   (0,0): D1=D2=D3=0 → run mode. run_val=a=0, remaining=1. value=2≠0 → run MISS.
        //     J[0]=0: read bit=0 (miss); no remainder bits; run_len=0.
        //     run_len=0 < remaining=1 → run interrupt.
        //     ri_ctx: a=4,n=1 → k=2. rb=ra=0, px=0.
        //     Need rx=2 → RItype=1, map=false, e_mapped=2×2−1=3.
        //     Golomb(3,k=2): q=0; stop 1; rem=3 (2 bits "11").
        //     Bits: 0(miss) 1(stop) 11(rem) = 4 bits.
        //   Total: 4 bits. Padding: 4 zero bits.
        //   Byte: [0b01110000] = [0x70].
        let frame = build_jpeg_ls_frame(1, 1, &[0x70]);
        let layout = layout_8bit(1, 1, 2.0, -5.0);
        let result = decode_jpeg_ls_fragment(&frame, layout).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], -1.0f32);
    }
}
