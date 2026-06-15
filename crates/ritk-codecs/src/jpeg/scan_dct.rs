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
use super::marker::{JpegFrameData, QuantPrecision, SOF0, SOF1};

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
    dc_table_id: u8,
    ac_table_id: u8,
    quant_id: usize,
    prev_dc: &mut i32,
) -> Result<[i16; 64]> {
    let dc_table = frame.dc_huff[dc_table_id as usize]
        .as_ref()
        .with_context(|| format!("DC Huffman table {dc_table_id} not loaded"))?;
    let ac_table = frame.ac_huff[ac_table_id as usize]
        .as_ref()
        .with_context(|| format!("AC Huffman table {ac_table_id} not loaded"))?;
    let quant = frame.quant[quant_id]
        .as_ref()
        .with_context(|| format!("Quantization table {quant_id} not loaded"))?;
    if quant.precision != QuantPrecision::Bits8 {
        bail!(
            "JPEG DCT quantization table {quant_id} uses 16-bit precision; only 8-bit DQT is supported"
        );
    }

    // Decode DC coefficient (T.81 §F.2.2.1)
    let dc_cat = dc_table.decode(reader)?;
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
                fc.quant_id as usize,
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
    let mut max_h = 1u8;
    let mut max_v = 1u8;
    for sc in &frame.sos.components {
        let fc_idx = comp_by_id(sc.id)?;
        let fc = &frame.sof.components[fc_idx];
        if fc.h_samp > max_h {
            max_h = fc.h_samp;
        }
        if fc.v_samp > max_v {
            max_v = fc.v_samp;
        }
    }

    let mcu_width = DCT_BLOCK_DIM * max_h as usize;
    let mcu_height = DCT_BLOCK_DIM * max_v as usize;
    let mcus_x = width.div_ceil(mcu_width);
    let mcus_y = height.div_ceil(mcu_height);
    let total_width = mcus_x * mcu_width;
    let total_height = mcus_y * mcu_height;

    // Component plane buffers (full padded size)
    let ncomp = frame.sos.components.len();
    let mut planes: Vec<Vec<u8>> = (0..ncomp)
        .map(|_| vec![0u8; total_width * total_height])
        .collect();

    let mut prev_dc = [0i32; 4];
    let mut reader = BitReader::new(entropy_data);

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for (ci, sc) in frame.sos.components.iter().enumerate() {
                let fc_idx = comp_by_id(sc.id)?;
                let fc = &frame.sof.components[fc_idx];

                // Decode h_samp × v_samp blocks for this component per MCU
                for bv in 0..(fc.v_samp as usize) {
                    for bh in 0..(fc.h_samp as usize) {
                        let coeffs = decode_block(
                            &mut reader,
                            frame,
                            sc.dc_table_id,
                            sc.ac_table_id,
                            fc.quant_id as usize,
                            &mut prev_dc[ci],
                        )?;
                        let block = reconstruct_block(&coeffs, frame.sof.precision);

                        // Block position in the padded component plane.
                        // Component (ci) has sampling h_samp:max_h, v_samp:max_v.
                        // Each MCU has max_h*8 × max_v*8 pixels at full resolution.
                        // Component ci's sub-blocks map to that MCU region.
                        let base_x = mcu_x * max_h as usize * DCT_BLOCK_DIM + bh * DCT_BLOCK_DIM;
                        let base_y = mcu_y * max_v as usize * DCT_BLOCK_DIM + bv * DCT_BLOCK_DIM;

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
                    .unwrap();
                let fc = &frame.sof.components[fc_idx];
                let scale_x = max_h as usize / fc.h_samp as usize;
                let scale_y = max_v as usize / fc.v_samp as usize;
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
