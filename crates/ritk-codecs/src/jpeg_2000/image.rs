//! Full J2K codestream decoder and pixel extractor.
//!
//! # Pipeline (ISO 15444-1)
//! 1. Parse main header (SIZ, COD, QCD) via `codestream`.
//! 2. Locate tile-part data (SOT → SOD).
//! 3. Decode each tile-component via `packet::decode_tile_part`.
//! 4. Apply DC level un-shift for unsigned components (ISO 15444-1 §G.1.2).
//! 5. Validate decoded dimensions against `PixelLayout`.
//! 6. Apply DICOM modality LUT: `output = stored_integer × slope + intercept`.

use anyhow::{bail, Context, Result};

use super::codestream::{parse_main_header, parse_sot};
use super::ebcot::SubbandOrientation;
use super::marker;
use super::packet::{decode_tile_part, TileCodingParams};
use crate::PixelLayout;

/// Decode a DICOM-encapsulated J2K codestream, returning rescaled `f32` pixel values.
///
/// # Specification
/// - Transfer syntax 1.2.840.10008.1.2.4.90 (lossless) and .91 (lossy or lossless).
/// - The fragment must start with the SOC marker (0xFF 0x4F).
/// - DC level shift reversed for unsigned components per ISO 15444-1 §G.1.2.
/// - Modality LUT applied: `output = stored_integer × slope + intercept`.
pub fn decode_j2k_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    if !is_soc(fragment) {
        bail!(
            "J2K: fragment does not begin with SOC 0xFF4F \
             (first 2 bytes: {:02X?})",
            &fragment[..fragment.len().min(2)]
        );
    }

    let (header, mut pos) = parse_main_header(fragment).context("J2K: parse main header")?;

    let siz = &header.siz;
    let cod = &header.cod;
    let qcd = &header.qcd;

    let num_guard_bits = qcd.num_guard_bits();

    // Validate layout consistency.
    let expected_comps = layout.samples_per_pixel;
    if siz.csiz as usize != expected_comps {
        bail!(
            "J2K: Csiz={} does not match layout samples_per_pixel={}",
            siz.csiz,
            expected_comps
        );
    }

    let img_w = siz.width() as usize;
    let img_h = siz.height() as usize;
    if img_w != layout.cols || img_h != layout.rows {
        bail!(
            "J2K: image dimensions {}×{} do not match layout {}×{}",
            img_w,
            img_h,
            layout.cols,
            layout.rows
        );
    }

    let num_tiles = siz.num_tiles_x() * siz.num_tiles_y();
    if num_tiles == 0 {
        bail!("J2K: image has 0 tiles");
    }

    // For the RITK DICOM use case, single-tile images are the norm (DICOM
    // encapsulates one frame per fragment with a single tile).  Multi-tile
    // images are handled by reconstructing each tile into the correct region
    // of the output buffer.
    //
    // Currently only num_decomp_levels = 0 is supported.  Images with
    // num_decomp_levels > 0 report a diagnostic error.
    if cod.num_decomp_levels > 0 {
        bail!(
            "J2K: num_decomp_levels={} > 0: DWT not yet supported in the RITK-native decoder. \
             This will be addressed in a future sprint.",
            cod.num_decomp_levels
        );
    }

    // Decode all tile-parts.
    // We allocate the full output and write each tile into its region.
    let total_pixels = layout.rows * layout.cols * layout.samples_per_pixel;
    let mut out = vec![0f32; total_pixels];

    // State machine: walk the tile-part markers.
    let mut tiles_decoded = 0u32;
    let data = fragment;

    while pos + 2 <= data.len() {
        let m = marker::read_u16(data, pos)?;
        match m {
            marker::SOT => {
                let (sot, after_sot) = parse_sot(data, pos).context("J2K: parse SOT")?;
                pos = after_sot;

                // Locate SOD inside this tile-part.
                let sod_pos =
                    find_sod(data, pos).with_context(|| "J2K: SOD not found in tile-part")?;
                let tile_data_start = sod_pos + 2; // skip SOD marker

                // Determine tile-part byte extent.
                let tile_end = if sot.psot > 0 {
                    // psot is from start of SOT marker.
                    let sot_start = pos - 12; // pos advanced past SOT segment
                    sot_start + sot.psot as usize
                } else {
                    // psot=0: extends to next SOT or EOC.
                    find_next_sot_or_eoc(data, tile_data_start).unwrap_or(data.len())
                };

                let tile_end = tile_end.min(data.len());
                let tile_data = &data[tile_data_start..tile_end];

                // Compute tile (tx, ty) from Isot.
                let isot = sot.isot as u32;
                let ntx = siz.num_tiles_x();
                let tx = isot % ntx;
                let ty = isot / ntx;

                let tw = siz.tile_width(tx) as usize;
                let th = siz.tile_height(ty) as usize;
                let tile_x0 = (siz.xto_siz + tx * siz.xt_siz).saturating_sub(siz.xo_siz) as usize;
                let tile_y0 = (siz.yto_siz + ty * siz.yt_siz).saturating_sub(siz.yo_siz) as usize;

                // Decode each component.
                for ci in 0..siz.csiz as usize {
                    let comp_spec = &siz.components[ci];
                    let c_prec = comp_spec.precision();
                    let c_signed = comp_spec.is_signed();

                    let tile_comp = decode_tile_part(
                        tile_data,
                        tw,
                        th,
                        TileCodingParams {
                            num_guard_bits,
                            precision: c_prec,
                            num_decomp_levels: cod.num_decomp_levels,
                            num_layers: cod.num_layers.max(1),
                            orient: SubbandOrientation::LlOrLh,
                        },
                    )
                    .with_context(|| format!("J2K: decode tile {isot} component {ci}"))?;

                    // Write reconstructed samples into the output buffer.
                    for py in 0..th {
                        for px in 0..tw {
                            let img_x = tile_x0 + px;
                            let img_y = tile_y0 + py;
                            if img_x >= img_w || img_y >= img_h {
                                continue;
                            }
                            let dc_shifted = tile_comp.samples[py * tw + px];
                            // Reverse DC level shift for unsigned components
                            // (ISO 15444-1 §G.1.2).
                            let raw = if c_signed {
                                dc_shifted
                            } else {
                                dc_shifted + (1i32 << (c_prec - 1))
                            };
                            // Apply DICOM modality LUT.
                            let rescaled = raw as f64 * f64::from(layout.rescale_slope)
                                + f64::from(layout.rescale_intercept);
                            let out_idx = (img_y * img_w + img_x) * layout.samples_per_pixel + ci;
                            if out_idx < out.len() {
                                out[out_idx] = rescaled as f32;
                            }
                        }
                    }
                }

                tiles_decoded += 1;
                pos = tile_end;
            }
            marker::EOC => break,
            // Skip optional tile-part header markers (e.g., PPT, PLT).
            marker::PPT | marker::COM => {
                pos += 2;
                if pos + 2 > data.len() {
                    break;
                }
                let len = marker::read_u16(data, pos)? as usize;
                pos += len;
            }
            _ => {
                // Unknown marker: try to skip.
                pos += 2;
                if pos + 2 <= data.len() {
                    let len = marker::read_u16(data, pos).unwrap_or(2) as usize;
                    pos += len;
                } else {
                    break;
                }
            }
        }
    }

    if tiles_decoded == 0 {
        bail!("J2K: no tile-parts were decoded");
    }

    Ok(out)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns `true` if `fragment` begins with the J2K SOC marker (0xFF 0x4F).
#[inline]
pub fn is_soc(fragment: &[u8]) -> bool {
    fragment.len() >= 2 && fragment[0] == 0xFF && fragment[1] == 0x4F
}

/// Find the SOD marker within `data[start..]` and return its offset.
fn find_sod(data: &[u8], start: usize) -> Option<usize> {
    (start..data.len().saturating_sub(1)).find(|&i| data[i] == 0xFF && data[i + 1] == 0x93)
}

/// Find the next SOT (0xFF90) or EOC (0xFFD9) marker after `start`.
fn find_next_sot_or_eoc(data: &[u8], start: usize) -> Option<usize> {
    (start..data.len().saturating_sub(1))
        .find(|&i| data[i] == 0xFF && (data[i + 1] == 0x90 || data[i + 1] == 0xD9))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelSignedness;

    fn make_layout(rows: usize, cols: usize, bits: u16, signed: PixelSignedness) -> PixelLayout {
        PixelLayout {
            rows,
            cols,
            samples_per_pixel: 1,
            bits_allocated: bits,
            pixel_representation: signed,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        }
    }

    #[test]
    fn is_soc_accepts_valid_prefix() {
        assert!(is_soc(&[0xFF, 0x4F, 0x00]));
    }

    #[test]
    fn is_soc_rejects_jpeg_baseline_prefix() {
        assert!(!is_soc(&[0xFF, 0xD8, 0xFF, 0xE0]));
    }

    #[test]
    fn is_soc_rejects_empty_slice() {
        assert!(!is_soc(&[]));
    }

    #[test]
    fn decode_j2k_fragment_rejects_non_soc() {
        let data = [0xFF_u8, 0xD8, 0x00];
        let err = decode_j2k_fragment(&data, make_layout(1, 1, 8, PixelSignedness::Unsigned))
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("SOC") || format!("{err:#}").contains("0xFF4F"),
            "error must mention SOC; got: {err:#}"
        );
    }
}
