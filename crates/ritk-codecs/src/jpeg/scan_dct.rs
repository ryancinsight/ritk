//! JPEG Baseline (SOF0/SOF1) sequential DCT scan decode.
//!
//! # Specification
//! ITU-T T.81 §F.2: Sequential entropy decode for Baseline and Extended DCT.
//! Coefficient decode order: DC first (one per block), then AC (up to 63 per block).
//! AC encoding: run-length pairs (run, size) per T.81 §F.1.2.1.
//!
//! After decode, dequantize and apply 2D IDCT to each 8×8 block.
//! Level-shift: add 2^(P-1) (= 128 for P=8) after IDCT, clamp to [0, 2^P − 1].

use anyhow::{bail, Context, Result};

use super::backend::{JpegDecoded, JpegPixelFormat};
use super::color::ycbcr_to_rgb;
use super::constants::{DCT_BLOCK_CELLS, DCT_BLOCK_DIM};
use super::huffman::{receive_and_extend, BitReader};
use super::idct::idct_8x8;
use super::marker::{JpegFrameData, QuantPrecision, TableId, MAX_SCAN_COMPONENTS, SOF0, SOF1};
use crate::dimensions::{checked_pixel_count, checked_sample_count};

/// Largest DC difference magnitude category T.81 Table F.1 defines.
///
/// Baseline uses 0-11; the extended sequential process reaches 15. The value
/// arrives as a decoded HUFFVAL byte, which spans 0-255 on the wire.
const MAX_DC_CATEGORY: u8 = 15;

/// Natural zigzag-to-raster reorder (T.81 §A.3.6).
const ZIGZAG: [usize; DCT_BLOCK_CELLS] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Decode one 8×8 block from the entropy stream into dequantised IDCT output.
///
/// `prev_dc` is updated in-place (DC differential coding).
fn decode_block(
    reader: &mut BitReader<'_>,
    frame: &JpegFrameData,
    dc_table_id: TableId,
    ac_table_id: TableId,
    quant_id: TableId,
    prev_dc: &mut i32,
) -> Result<[i16; 64]> {
    let dc_table = frame.dc_huff[dc_table_id.index()]
        .as_ref()
        .with_context(|| format!("DC Huffman table {dc_table_id} not loaded"))?;
    let ac_table = frame.ac_huff[ac_table_id.index()]
        .as_ref()
        .with_context(|| format!("AC Huffman table {ac_table_id} not loaded"))?;
    let quant = frame.quant[quant_id.index()]
        .as_ref()
        .with_context(|| format!("Quantization table {quant_id} not loaded"))?;
    if quant.precision != QuantPrecision::Bits8 {
        bail!(
            "JPEG DCT quantization table {quant_id} uses 16-bit precision; only 8-bit DQT is supported"
        );
    }

    // Decode DC coefficient (T.81 §F.2.2.1).
    //
    // The category is a decoded HUFFVAL byte, so the wire admits 0-255 while
    // T.81 Table F.1 defines 0-11 for baseline and 0-15 for extended. Rejecting
    // it here names the offending value; `read_bits` also refuses out-of-range
    // counts, but by then the error no longer says which field was wrong.
    let dc_cat = dc_table.decode(reader)?;
    if dc_cat > MAX_DC_CATEGORY {
        bail!(
            "invalid DC difference magnitude category {dc_cat}; T.81 allows 0 to {MAX_DC_CATEGORY}"
        );
    }
    let dc_diff = receive_and_extend(reader, dc_cat)?;
    *prev_dc += dc_diff;
    let dc = *prev_dc;

    // Decode AC coefficients (T.81 §F.2.2.2)
    let mut coeffs_zigzag = [0i16; DCT_BLOCK_CELLS];
    coeffs_zigzag[0] = dc as i16;

    let mut k = 1usize;
    while k < DCT_BLOCK_CELLS {
        let rs = ac_table.decode(reader)?;
        let run = (rs >> 4) as usize;
        let size = rs & 0x0F;
        if size == 0 {
            if run == 15 {
                k += 16; // ZRL: 16 zeros
            } else {
                break; // EOB: rest of coefficients are zero
            }
        } else {
            k += run;
            if k >= DCT_BLOCK_CELLS {
                break;
            }
            let val = receive_and_extend(reader, size)?;
            coeffs_zigzag[k] = val as i16;
            k += 1;
        }
    }

    // Dequantize: multiply by quantization table in zigzag order, place in raster order
    let mut coeffs_raster = [0i16; DCT_BLOCK_CELLS];
    for (zz, &qval) in quant.values.iter().enumerate() {
        let raster = ZIGZAG[zz];
        coeffs_raster[raster] = coeffs_zigzag[zz].saturating_mul(qval as i16);
    }

    Ok(coeffs_raster)
}

/// Apply 8×8 IDCT to a block of quantized coefficients and level-shift.
fn reconstruct_block(coeffs: &[i16; DCT_BLOCK_CELLS], precision: u8) -> [u8; DCT_BLOCK_CELLS] {
    let mut block = [0.0f32; DCT_BLOCK_CELLS];
    for (i, &c) in coeffs.iter().enumerate() {
        block[i] = c as f32;
    }
    idct_8x8(&mut block);
    let level_shift = (1 << (precision - 1)) as f32;
    let maxval = ((1 << precision) - 1) as f32;
    let mut out = [0u8; DCT_BLOCK_CELLS];
    for (i, v) in block.iter().enumerate() {
        let shifted = v + level_shift;
        out[i] = shifted.clamp(0.0, maxval) as u8;
    }
    out
}

/// Decode a JPEG Baseline (SOF0) or Extended Sequential (SOF1) scan.
///
/// Supports:
/// - 1-component (grayscale, L8): `pixel_format = L8`
/// - 3-component YCbCr (H/V sampling 1:1:1 or 4:2:0): `pixel_format = Rgb24`
///
/// Returns interleaved RGB24 or single-plane L8 bytes in raster order.
pub(crate) fn decode_baseline_scan(
    frame: &JpegFrameData,
    entropy_data: &[u8],
) -> Result<JpegDecoded> {
    let marker = frame.sof.sof_marker;
    if marker != SOF0 && marker != SOF1 {
        bail!("decode_baseline_scan called with SOF marker 0x{marker:04X}");
    }
    let precision = frame.sof.precision;
    if precision != 8 {
        bail!("JPEG Baseline: only 8-bit precision supported (got {precision})");
    }
    if frame.sos.ss != 0 || frame.sos.se != 63 || frame.sos.ah != 0 || frame.sos.al != 0 {
        bail!(
            "JPEG sequential DCT scan parameters unsupported: Ss={} Se={} Ah={} Al={}",
            frame.sos.ss,
            frame.sos.se,
            frame.sos.ah,
            frame.sos.al
        );
    }
    let width = frame.sof.width as usize;
    let height = frame.sof.height as usize;
    let ncomp = frame.sof.components.len();

    // Map component id → frame component index
    let comp_by_id = |id: u8| -> Result<usize> {
        frame
            .sof
            .components
            .iter()
            .position(|c| c.id == id)
            .with_context(|| format!("SOS references unknown component id {id}"))
    };

    match ncomp {
        1 => decode_baseline_grayscale(frame, entropy_data, width, height, comp_by_id),
        3 => decode_baseline_ycbcr(frame, entropy_data, width, height, comp_by_id),
        _ => bail!("JPEG Baseline: unsupported component count {ncomp}"),
    }
}

fn decode_baseline_grayscale(
    frame: &JpegFrameData,
    entropy_data: &[u8],
    width: usize,
    height: usize,
    comp_by_id: impl Fn(u8) -> Result<usize>,
) -> Result<JpegDecoded> {
    let scan_comp = &frame.sos.components[0];
    let fc_idx = comp_by_id(scan_comp.id)?;
    let fc = &frame.sof.components[fc_idx];

    let blocks_x = width.div_ceil(DCT_BLOCK_DIM);
    let blocks_y = height.div_ceil(DCT_BLOCK_DIM);
    let mut pixels = vec![0u8; width * height];
    let mut prev_dc = 0i32;
    let mut reader = BitReader::new(entropy_data);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let coeffs = decode_block(
                &mut reader,
                frame,
                scan_comp.dc_table_id,
                scan_comp.ac_table_id,
                fc.quant_id,
                &mut prev_dc,
            )?;
            let block = reconstruct_block(&coeffs, frame.sof.precision);
            // Write 8×8 block into output, clamping to image bounds
            for r in 0..DCT_BLOCK_DIM {
                let py = by * DCT_BLOCK_DIM + r;
                if py >= height {
                    break;
                }
                for c in 0..DCT_BLOCK_DIM {
                    let px = bx * DCT_BLOCK_DIM + c;
                    if px >= width {
                        break;
                    }
                    pixels[py * width + px] = block[r * DCT_BLOCK_DIM + c];
                }
            }
        }
    }

    Ok(JpegDecoded {
        width,
        height,
        pixel_format: JpegPixelFormat::L8,
        pixels,
    })
}

fn decode_baseline_ycbcr(
    frame: &JpegFrameData,
    entropy_data: &[u8],
    width: usize,
    height: usize,
    comp_by_id: impl Fn(u8) -> Result<usize>,
) -> Result<JpegDecoded> {
    // Determine MCU structure from scan component sampling factors.
    // Find max H and V sampling factors across all scan components.
    let mut max_h = 1usize;
    let mut max_v = 1usize;
    for sc in &frame.sos.components {
        let fc_idx = comp_by_id(sc.id)?;
        let fc = &frame.sof.components[fc_idx];
        max_h = max_h.max(fc.h_samp.get());
        max_v = max_v.max(fc.v_samp.get());
    }

    let mcu_width = DCT_BLOCK_DIM * max_h;
    let mcu_height = DCT_BLOCK_DIM * max_v;
    let mcus_x = width.div_ceil(mcu_width);
    let mcus_y = height.div_ceil(mcu_height);
    let total_width = mcus_x * mcu_width;
    let total_height = mcus_y * mcu_height;

    // `RitkJpegDecoder` bounds `width * height` against the SOF, but the buffers
    // allocated here are the MCU-padded size times the *scan* component count,
    // and padding grows with the sampling factors. Re-check the number actually
    // allocated through the same limit rather than assuming the earlier one
    // covered it.
    let plane_cells =
        checked_pixel_count(total_width, total_height).context("MCU-padded frame dimensions")?;
    let ncomp = frame.sos.components.len();
    checked_sample_count(plane_cells, ncomp).context("MCU-padded scan planes")?;
    let mut planes: Vec<Vec<u8>> = (0..ncomp).map(|_| vec![0u8; plane_cells]).collect();

    let mut prev_dc = [0i32; MAX_SCAN_COMPONENTS];
    let mut reader = BitReader::new(entropy_data);

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for (ci, sc) in frame.sos.components.iter().enumerate() {
                let fc_idx = comp_by_id(sc.id)?;
                let fc = &frame.sof.components[fc_idx];

                // Decode h_samp × v_samp blocks for this component per MCU
                for bv in 0..fc.v_samp.get() {
                    for bh in 0..fc.h_samp.get() {
                        let coeffs = decode_block(
                            &mut reader,
                            frame,
                            sc.dc_table_id,
                            sc.ac_table_id,
                            fc.quant_id,
                            &mut prev_dc[ci],
                        )?;
                        let block = reconstruct_block(&coeffs, frame.sof.precision);

                        // Block position in the padded component plane.
                        // Component (ci) has sampling h_samp:max_h, v_samp:max_v.
                        // Each MCU has max_h*8 × max_v*8 pixels at full resolution.
                        // Component ci's sub-blocks map to that MCU region.
                        let base_x = mcu_x * max_h * DCT_BLOCK_DIM + bh * DCT_BLOCK_DIM;
                        let base_y = mcu_y * max_v * DCT_BLOCK_DIM + bv * DCT_BLOCK_DIM;

                        for r in 0..DCT_BLOCK_DIM {
                            for c in 0..DCT_BLOCK_DIM {
                                let px = base_x + c;
                                let py = base_y + r;
                                if px < total_width && py < total_height {
                                    planes[ci][py * total_width + px] =
                                        block[r * DCT_BLOCK_DIM + c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Upsample chroma and interleave into RGB.
    // For each component, scale position by (max_h/h_samp) × (max_v/v_samp).
    let mut pixels = vec![0u8; width * height * 3];
    for py in 0..height {
        for px in 0..width {
            let mut comps = [0u8; 3];
            for (ci, sc) in frame.sos.components.iter().enumerate() {
                let fc_idx = frame
                    .sof
                    .components
                    .iter()
                    .position(|c| c.id == sc.id)
                    .expect("infallible: validated precondition");
                let fc = &frame.sof.components[fc_idx];
                let scale_x = max_h / fc.h_samp.get();
                let scale_y = max_v / fc.v_samp.get();
                let cp_x = px / scale_x;
                let cp_y = py / scale_y;
                comps[ci] = planes[ci][cp_y * total_width + cp_x];
            }
            let (r, g, b) = ycbcr_to_rgb(comps[0] as i32, comps[1] as i32, comps[2] as i32);
            let out = &mut pixels[(py * width + px) * 3..];
            out[0] = r;
            out[1] = g;
            out[2] = b;
        }
    }

    Ok(JpegDecoded {
        width,
        height,
        pixel_format: JpegPixelFormat::Rgb24,
        pixels,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::jpeg::marker::parse_jpeg;

    /// Hand-crafted three-component baseline DCT JPEG: 8×8, YCbCr, 1×1 sampling.
    ///
    /// The committed fixtures were all single-component lossless (SOF3), so no
    /// test reached this decoder at all — which is why the SOS component count,
    /// the DC category and the MCU-padded allocation each went unbounded until
    /// an audit found them by reading. This is the smallest stream that
    /// exercises the three-component DCT path end to end.
    ///
    /// Every block codes DC category 0 (difference 0) followed immediately by
    /// EOB, so all coefficients are zero and each component reconstructs to the
    /// level shift, 128. Y=Cb=Cr=128 is mid-grey in RGB.
    ///
    /// Both Huffman tables hold a single one-bit code `0`: DC symbol 0 is
    /// category 0, AC symbol 0x00 is EOB. Each block is therefore two bits, and
    /// three blocks fill six, padded to `0b000000_11` per T.81 §F.1.2.3.
    pub(crate) fn baseline_ycbcr_fixture() -> Vec<u8> {
        baseline_fixture(3)
    }

    /// Single-component baseline DCT JPEG, otherwise identical.
    ///
    /// The grayscale scan path is a separate function reached only when SOF
    /// declares one component, so the three-component fixture never exercises
    /// it — and that is the path holding the `sos.components[0]` index.
    pub(crate) fn baseline_grayscale_fixture() -> Vec<u8> {
        baseline_fixture(1)
    }

    /// Build a baseline fixture with `components` components, 1x1 sampled.
    fn baseline_fixture(components: usize) -> Vec<u8> {
        let mut stream = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xDB, // DQT
            0x00, 0x43, // length 67 = 2 + 1 + 64
            0x00, // Pq=0 (8-bit), Tq=0
        ];
        stream.extend(std::iter::repeat_n(0x01, 64)); // flat quantisation

        let sof_marker_at = stream.len();
        stream.extend_from_slice(&[
            0xFF, 0xC0, // SOF0 (baseline DCT)
            0x00, 0x00, // length patched below
            0x08, // precision 8
            0x00, 0x08, // height 8
            0x00, 0x08, // width 8
        ]);
        stream.push(components as u8);
        for id in 1..=components as u8 {
            stream.extend_from_slice(&[id, 0x11, 0x00]); // 1x1 sampling, quant table 0
        }

        // DC table 0 and AC table 0, each one code of length 1 mapping to
        // symbol 0.
        for class_and_id in [0x00u8, 0x10] {
            stream.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, class_and_id]);
            stream.push(0x01); // BITS[1] = one code of length 1
            stream.extend(std::iter::repeat_n(0x00, 15)); // BITS[2..=16] = 0
            stream.push(0x00); // HUFFVAL[0] = 0
        }

        stream.extend_from_slice(&[]);

        // SOF0 length: 2 + precision + 2*dimension + component count + 3 each.
        let sof_len = (8 + 3 * components) as u16;
        let sof_len_at = sof_marker_at + 2;
        stream[sof_len_at..sof_len_at + 2].copy_from_slice(&sof_len.to_be_bytes());

        stream.extend_from_slice(&[0xFF, 0xDA]); // SOS
        let sos_len = (6 + 2 * components) as u16;
        stream.extend_from_slice(&sos_len.to_be_bytes());
        stream.push(components as u8);
        for id in 1..=components as u8 {
            stream.extend_from_slice(&[id, 0x00]); // DC table 0, AC table 0
        }
        stream.extend_from_slice(&[
            0x00, // Ss = 0
            0x3F, // Se = 63
            0x00, // Ah = 0, Al = 0
        ]);
        // Two bits per block (DC category 0, then EOB), padded with 1-bits per
        // T.81 §F.1.2.3.
        let used = 2 * components;
        let entropy = (0xFFu16 >> used) as u8;
        stream.push(entropy);
        stream.extend_from_slice(&[0xFF, 0xD9]); // EOI
        stream
    }

    /// Decode the fragment the way the public entry point does.
    ///
    /// Returns `None` for any stream the header stage rejects, so a sweep can
    /// tell "refused at the header" from "reached the scan".
    fn decode_fragment(fragment: &[u8]) -> Option<Result<JpegDecoded>> {
        let frame = parse_jpeg(fragment).ok()?;
        let entropy = fragment.get(frame.scan_data_start..)?;
        match frame.sof.sof_marker {
            SOF0 | SOF1 => Some(decode_baseline_scan(&frame, entropy)),
            _ => None,
        }
    }

    /// Every truncation of the baseline stream errors rather than panicking.
    ///
    /// The header sweeps in `marker.rs` cover their own stage thoroughly, but
    /// their fixtures are single-component lossless, so nothing exercised the
    /// entropy decoder against a short stream. The entropy stage pads with
    /// 1-bits at end of data by design, so a truncated scan decodes to
    /// *something*; what matters is that it stays inside its buffers while
    /// doing so.
    #[test]
    fn truncating_the_baseline_stream_never_panics() {
        for fixture in [baseline_grayscale_fixture(), baseline_ycbcr_fixture()] {
            let channels = usize::from(fixture[9]); // SOF component count
            for cut in 0..fixture.len() {
                let Some(result) = decode_fragment(&fixture[..cut]) else {
                    continue; // rejected at the header stage, which is its own sweep
                };
                if let Ok(decoded) = result {
                    assert_eq!(
                        decoded.pixels.len(),
                        decoded.width * decoded.height * decoded.pixel_format.pixel_bytes(),
                        "prefix of {cut} bytes produced a buffer inconsistent with its dimensions"
                    );
                }
            }
            assert!(
                decode_fragment(&fixture)
                    .expect("intact stream reaches the scan")
                    .is_ok(),
                "the intact {channels}-component fixture must decode"
            );
        }
    }

    /// Single-byte corruption either fails or yields a self-consistent image.
    ///
    /// The scan stage indexes plane buffers by MCU geometry derived from SOF
    /// dimensions and sampling factors, and steps Huffman tables by decoded
    /// symbols. Substituting the extremes at every offset reaches each of those
    /// without needing a corpus.
    ///
    /// Both halves of the assertion matter. Arriving here at all requires no
    /// panic, and any image that comes back must have a buffer matching the
    /// dimensions it reports — so a decoder that wrote a plane sized from one
    /// component count and reported another fails here.
    #[test]
    fn single_byte_corruption_of_the_baseline_stream_stays_self_consistent() {
        for fixture in [baseline_grayscale_fixture(), baseline_ycbcr_fixture()] {
            for offset in 0..fixture.len() {
                for byte in [0x00u8, 0x0F, 0xF0, 0xFF] {
                    let mut corrupt = fixture.clone();
                    corrupt[offset] = byte;
                    let Some(Ok(decoded)) = decode_fragment(&corrupt) else {
                        continue;
                    };
                    assert_eq!(
                    decoded.pixels.len(),
                    decoded.width * decoded.height * decoded.pixel_format.pixel_bytes(),
                    "byte {byte:#04X} at offset {offset} produced a buffer inconsistent                      with its {}x{} dimensions",
                    decoded.width,
                    decoded.height
                );
                }
            }
        }
    }

    /// The fixture must decode, or every sweep built on it proves nothing.
    #[test]
    fn the_baseline_fixture_decodes_to_mid_grey() {
        let fixture = baseline_ycbcr_fixture();
        let frame = parse_jpeg(&fixture).expect("the fixture is a conforming baseline stream");
        assert_eq!(frame.sof.sof_marker, SOF0);
        assert_eq!(frame.sof.components.len(), 3);
        assert_eq!(frame.sos.components.len(), 3);

        let decoded = decode_baseline_scan(&frame, &fixture[frame.scan_data_start..])
            .expect("all-zero coefficients decode");
        assert_eq!(decoded.pixel_format, JpegPixelFormat::Rgb24);
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.pixels.len(), 8 * 8 * 3);
        // Y=Cb=Cr=128 is achromatic, so every RGB channel lands on the level
        // shift. One off-by-one either way is the YCbCr rounding.
        for (i, &sample) in decoded.pixels.iter().enumerate() {
            assert!(
                sample.abs_diff(128) <= 1,
                "sample {i} is {sample}, expected the 128 level shift"
            );
        }
    }
}
