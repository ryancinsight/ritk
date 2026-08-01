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
use std::ops::Range;

use super::codestream::{parse_main_header, parse_sot};
use super::marker;
use super::packet::{decode_tile_part, TileCodingParams, WaveletTransform};
use crate::dimensions::{checked_pixel_count, checked_sample_count};
use crate::PixelLayout;

#[derive(Debug)]
struct TilePartRange {
    isot: u16,
    data: Range<usize>,
}

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

    let (header, pos) = parse_main_header(fragment).context("J2K: parse main header")?;

    let siz = &header.siz;
    let cod = &header.cod;
    let qcd = &header.qcd;

    let num_guard_bits = qcd.num_guard_bits();
    let qcd_exponents = qcd.exponents();
    let qcd_mantissas = qcd.mantissas();
    // The packet reader currently traverses one grayscale LRCP packet stream.
    // Replaying that stream once per component would silently duplicate the
    // first component, so reject the unsupported shape before allocating or
    // decoding any output.
    if siz.csiz != 1 {
        bail!(
            "J2K: native decode currently supports one grayscale component; Csiz={}",
            siz.csiz
        );
    }
    if cod.progression_order != 0 {
        bail!(
            "J2K: native decode supports LRCP progression order 0; found {}",
            cod.progression_order
        );
    }
    if cod.mct != 0 {
        bail!(
            "J2K: native decode does not implement the multiple-component transform; MCT={}",
            cod.mct
        );
    }
    if cod.scod != 0 {
        bail!(
            "J2K: native decode requires default precincts without SOP/EPH; Scod=0x{:02X}",
            cod.scod
        );
    }
    if cod.num_layers == 0 {
        bail!("J2K: COD declares zero quality layers");
    }
    if cod.xcb_o != 4 || cod.ycb_o != 4 {
        bail!(
            "J2K: native decode requires 64x64 nominal code-blocks; found {}x{}",
            cod.cb_width(),
            cod.cb_height()
        );
    }
    if cod.cb_style != 0 {
        bail!(
            "J2K: native decode requires default code-block style; found 0x{:02X}",
            cod.cb_style
        );
    }
    let transform = match cod.wavelet_transform {
        0 => WaveletTransform::Irreversible,
        1 => WaveletTransform::Reversible,
        other => bail!("J2K: unsupported wavelet transform {other}; expected 0 or 1"),
    };

    // Validate layout consistency.
    let expected_comps = layout.samples_per_pixel;
    if siz.csiz as usize != expected_comps {
        bail!(
            "J2K: Csiz={} does not match layout samples_per_pixel={}",
            siz.csiz,
            expected_comps
        );
    }
    if let Some((index, component)) = siz
        .components
        .iter()
        .enumerate()
        .find(|(_, component)| component.xr_siz != 1 || component.yr_siz != 1)
    {
        bail!(
            "J2K: component {index} uses unsupported sampling XRsiz={} YRsiz={}; \
             DICOM interleaved decode requires 1x1 sampling",
            component.xr_siz,
            component.yr_siz
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

    let tile_count = usize::try_from(siz.num_tiles().context("J2K: validate tile grid")?)
        .context("J2K: tile count does not fit the platform address size")?;
    let tile_parts = scan_tile_parts(fragment, pos, tile_count)?;

    // For the RITK DICOM use case, single-tile images are the norm (DICOM
    // encapsulates one frame per fragment with a single tile).  Multi-tile
    // images are handled by reconstructing each tile into the correct region
    // of the output buffer.

    // The complete marker structure is validated before this allocation, so an
    // unsupported progression override or incomplete tile set cannot reserve
    // the DICOM-sized output buffer.
    // Bound the pixel count against a hostile/corrupt header before allocating
    // the full `f32` output (defense-in-depth: SIZ is already required to match
    // the DICOM layout above).
    let pixels = checked_pixel_count(layout.cols, layout.rows).context("J2K image")?;
    let total_samples =
        checked_sample_count(pixels, layout.samples_per_pixel).context("J2K output samples")?;
    let mut out = vec![0f32; total_samples];

    for tile_part in tile_parts {
        let isot = tile_part.isot;
        let bounds = siz
            .tile_bounds(isot)
            .context("J2K: validate SOT tile index")?;
        let tw = bounds.width;
        let th = bounds.height;
        let comp_spec = &siz.components[0];
        let c_prec = comp_spec.precision();
        let c_signed = comp_spec.is_signed();
        let tile_comp = decode_tile_part(
            &fragment[tile_part.data],
            tw,
            th,
            TileCodingParams {
                num_guard_bits,
                precision: c_prec,
                num_decomp_levels: cod.num_decomp_levels,
                num_layers: cod.num_layers,
                exponents: &qcd_exponents,
                mantissas: &qcd_mantissas,
                transform,
            },
        )
        .with_context(|| format!("J2K: decode tile {isot} component 0"))?;

        for py in 0..th {
            for px in 0..tw {
                let img_x = bounds.x0 + px;
                let img_y = bounds.y0 + py;
                if img_x >= img_w || img_y >= img_h {
                    continue;
                }
                let dc_shifted = tile_comp.samples[py * tw + px];
                let raw = if c_signed {
                    dc_shifted
                } else {
                    dc_shifted + (1i32 << (c_prec - 1))
                };
                let rescaled = raw as f64 * f64::from(layout.rescale_slope)
                    + f64::from(layout.rescale_intercept);
                let out_idx = (img_y * img_w + img_x) * layout.samples_per_pixel;
                if out_idx < out.len() {
                    out[out_idx] = rescaled as f32;
                }
            }
        }
    }
    Ok(out)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns `true` if `fragment` begins with the J2K SOC marker (0xFF 0x4F).
#[inline]
pub fn is_soc(fragment: &[u8]) -> bool {
    fragment.len() >= 2 && fragment[0] == 0xFF && fragment[1] == 0x4F
}

/// Locate the terminal EOC, permitting only the single zero byte that DICOM
/// uses to make an odd-length encapsulated item value even.
fn terminal_eoc_position(data: &[u8]) -> Result<usize> {
    if data.ends_with(&[0xFF, 0xD9]) {
        return Ok(data.len() - 2);
    }
    if data.len() >= 3 && data.ends_with(&[0xFF, 0xD9, 0x00]) && (data.len() - 1) % 2 == 1 {
        return Ok(data.len() - 3);
    }
    bail!(
        "J2K: EOC must terminate the codestream, followed only by an optional DICOM even-length zero pad"
    )
}

/// Return the first byte after a marker segment whose two-byte length includes
/// the length field itself.
fn marker_segment_end(
    data: &[u8],
    marker_pos: usize,
    marker_code: u16,
    limit: usize,
) -> Result<usize> {
    let length_pos = marker_pos
        .checked_add(2)
        .context("J2K: marker position overflows the platform address size")?;
    let length = usize::from(marker::read_u16(data, length_pos).with_context(|| {
        format!("J2K: marker 0x{marker_code:04X} at byte {marker_pos} has no length field")
    })?);
    if length < 2 {
        bail!(
            "J2K: marker 0x{marker_code:04X} at byte {marker_pos} has invalid length {length}; expected at least 2"
        );
    }
    let end = length_pos
        .checked_add(length)
        .context("J2K: marker segment end overflows the platform address size")?;
    if end > limit {
        bail!(
            "J2K: marker 0x{marker_code:04X} at byte {marker_pos} ends at byte {end}, beyond the byte limit {limit}"
        );
    }
    Ok(end)
}

fn scan_tile_parts(data: &[u8], start: usize, tile_count: usize) -> Result<Vec<TilePartRange>> {
    let terminal_eoc = terminal_eoc_position(data)?;
    let mut pos = start;
    let mut tile_parts = Vec::with_capacity(tile_count);
    let mut seen_tiles = vec![false; tile_count];
    let mut saw_eoc = false;

    while pos < data.len() {
        let marker_code = marker::read_u16(data, pos)
            .with_context(|| format!("J2K: truncated marker at byte {pos}"))?;
        match marker_code {
            marker::SOT => {
                let sot_start = pos;
                let (sot, after_sot) = parse_sot(data, pos).context("J2K: parse SOT")?;
                if sot.tpsot != 0 || sot.tnsot > 1 {
                    bail!(
                        "J2K: native decode supports one tile-part per tile; Isot={} TPsot={} TNsot={}",
                        sot.isot,
                        sot.tpsot,
                        sot.tnsot
                    );
                }
                let seen = seen_tiles.get_mut(usize::from(sot.isot)).with_context(|| {
                    format!(
                        "J2K: SOT tile index Isot={} is outside 0..{}",
                        sot.isot,
                        tile_count.saturating_sub(1)
                    )
                })?;
                if *seen {
                    bail!("J2K: tile {} has more than one tile-part", sot.isot);
                }

                let tile_end = if sot.psot > 0 {
                    let declared_end = sot_start
                        .checked_add(
                            usize::try_from(sot.psot)
                                .context("J2K: Psot does not fit the platform address size")?,
                        )
                        .context("J2K: Psot overflows the platform address size")?;
                    if declared_end > terminal_eoc {
                        bail!(
                            "J2K: tile-part Psot={} ends at byte {declared_end}, beyond terminal EOC at byte {terminal_eoc}",
                            sot.psot,
                        );
                    }
                    declared_end
                } else {
                    // ISO 15444-1 defines Psot=0 as extending the final
                    // tile-part to EOC. Using the validated terminal marker
                    // avoids treating marker-looking COM payload bytes as a
                    // tile boundary.
                    terminal_eoc
                };
                let tile_data_start = parse_tile_header(data, after_sot, tile_end)?;
                tile_parts.push(TilePartRange {
                    isot: sot.isot,
                    data: tile_data_start..tile_end,
                });
                *seen = true;
                pos = tile_end;
            }
            marker::EOC => {
                if pos != terminal_eoc {
                    bail!("J2K: EOC at byte {pos} precedes terminal EOC at byte {terminal_eoc}");
                }
                saw_eoc = true;
                break;
            }
            other => {
                marker_segment_end(data, pos, other, data.len())?;
                bail!("J2K: unexpected marker 0x{other:04X} at byte {pos} between tile-parts");
            }
        }
    }

    if !saw_eoc {
        bail!("J2K: EOC marker missing after tile data");
    }
    if let Some(missing_tile) = seen_tiles.iter().position(|&seen| !seen) {
        bail!(
            "J2K: EOC reached before tile {missing_tile} of {} was decoded",
            seen_tiles.len()
        );
    }
    Ok(tile_parts)
}

fn parse_tile_header(data: &[u8], mut pos: usize, tile_end: usize) -> Result<usize> {
    while pos < tile_end {
        let marker_code = marker::read_u16(data, pos)
            .with_context(|| format!("J2K: truncated tile-header marker at byte {pos}"))?;
        match marker_code {
            marker::SOD => return Ok(pos + 2),
            marker::COM | marker::PLT => {
                pos = marker_segment_end(data, pos, marker_code, tile_end)?;
            }
            marker::COD | marker::COC | marker::QCD | marker::QCC | marker::RGN | marker::POC => {
                bail!(
                "J2K: tile-header coding override 0x{marker_code:04X} at byte {pos} is unsupported"
            )
            }
            marker::PPT => {
                bail!("J2K: packed tile-part packet headers (PPT) are unsupported at byte {pos}")
            }
            marker::SOT | marker::EOC => {
                bail!("J2K: marker 0x{marker_code:04X} reached before SOD in tile-part header")
            }
            other => {
                marker_segment_end(data, pos, other, tile_end)?;
                bail!("J2K: unsupported tile-header marker 0x{other:04X} at byte {pos}");
            }
        }
    }
    bail!("J2K: SOD not found in tile-part")
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
