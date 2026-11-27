//! Full J2K codestream decoder and pixel extractor.
//!
//! # Pipeline (ISO 15444-1)
//! 1. Parse main header (SIZ, COD, QCD) via `codestream`.
//! 2. Locate tile-part data (SOT → SOD).
//! 3. Decode each tile-component via `packet::reader::decode_tile_part`.
//! 4. Apply DC level un-shift for unsigned components (ISO 15444-1 §G.1.2).
//! 5. Validate decoded dimensions against `PixelLayout`.
//! 6. Apply DICOM modality LUT: `output = stored_integer × slope + intercept`.

use anyhow::{bail, Context, Result};

use super::marker;
use super::packet::{
    reader::{decode_tile_part, TileCodingParams, TileCoefficient},
    WaveletTransform,
};
use super::Jpeg2000DecodeError;
use crate::PixelLayout;

/// Decode a DICOM-encapsulated J2K codestream, returning rescaled `f32` pixel values.
///
/// # Specification
/// - Transfer syntax 1.2.840.10008.1.2.4.90 (lossless) and .91 (lossy or lossless).
/// - The fragment must start with the SOC marker (0xFF 0x4F).
/// - DC level shift reversed for unsigned components per ISO 15444-1 §G.1.2.
/// - Modality LUT applied: `output = stored_integer × slope + intercept`.
pub(super) fn decode_j2k_fragment<T: Copy + Default>(
    fragment: &[u8],
    layout: PixelLayout,
    plan: &super::organization::Organization,
    convert: impl Fn(i64) -> T,
) -> Result<Vec<T>> {
    let siz = &plan.header.siz;
    let cod = &plan.header.cod;
    let qcd = &plan.header.qcd;
    let num_guard_bits = qcd.num_guard_bits();
    let qcd_exponents = qcd.exponents();
    let qcd_mantissas = qcd.mantissas();
    let [component] = siz.components.as_slice() else {
        bail!("expected a scalar tile component")
    };
    let c_prec = component.precision();
    let c_signed = component.is_signed();
    let transform = if cod.wavelet_transform == 1 {
        WaveletTransform::Reversible
    } else {
        WaveletTransform::Irreversible
    };
    let img_w = layout.cols;
    let samples = layout
        .samples_per_frame()
        .context("J2K: invalid DICOM pixel layout")?;
    let mut out = Vec::new();
    out.try_reserve_exact(samples).map_err(|source| {
        anyhow::Error::new(Jpeg2000DecodeError::OutputAllocation { samples, source })
    })?;
    out.resize(samples, T::default());

    for (index, tile) in plan.tiles.iter().enumerate() {
        let isot = u16::try_from(index)?;
        let bounds = siz
            .tile_bounds(isot)
            .context("J2K: validate SOT tile index")?;
        let coding = TileCodingParams {
            progression: cod.progression_order,
            high_throughput: cod.cb_style == 0x40,
            num_guard_bits,
            precision: c_prec,
            num_decomp_levels: cod.num_decomp_levels,
            num_layers: cod.num_layers,
            exponents: &qcd_exponents,
            mantissas: &qcd_mantissas,
            transform,
        };
        let bytes = tile.bytes(fragment);
        let data = bytes.as_ref();
        let mut put = |index, centered| {
            out[index] = convert(restore_component_sample(centered, c_prec, c_signed));
        };
        // Bound the narrow wavelet sums, including quantizer guard bits. A
        // more conservative precision is inexpensive for rare wide images.
        if c_prec > 24
            || qcd_exponents.iter().copied().max().unwrap_or(0) + u32::from(num_guard_bits) > 28
        {
            copy_tile::<i64>(data, coding, bounds, img_w, &mut put)
        } else {
            copy_tile::<i32>(data, coding, bounds, img_w, &mut put)
        }
        .with_context(|| format!("J2K: decode tile {isot} component 0"))?;
    }
    Ok(out)
}

fn copy_tile<C: TileCoefficient>(
    data: &[u8],
    coding: TileCodingParams<'_>,
    bounds: super::codestream::TileBounds,
    stride: usize,
    mut put: impl FnMut(usize, i64),
) -> Result<()> {
    let tile = decode_tile_part::<C>(data, bounds.width, bounds.height, coding)?;
    for (y, row) in tile.samples.chunks_exact(bounds.width).enumerate() {
        let start = (bounds.y0 + y) * stride + bounds.x0;
        for (x, &value) in row.iter().enumerate() {
            put(start + x, value.into());
        }
    }
    Ok(())
}

fn restore_component_sample(dc_shifted: i64, precision: u32, signed: bool) -> i64 {
    debug_assert!((1..=38).contains(&precision));
    let half = 1i64 << (precision - 1);
    let centered = dc_shifted.clamp(-half, half - 1);
    if signed {
        centered
    } else {
        centered + half
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns `true` if `fragment` begins with the J2K SOC marker (0xFF 0x4F).
#[inline]
pub fn is_soc(fragment: &[u8]) -> bool {
    fragment.len() >= 2 && fragment[0] == 0xFF && fragment[1] == 0x4F
}

/// Locate the terminal EOC, permitting only the single zero byte that DICOM
/// uses to make an odd-length encapsulated item value even.
pub(super) fn terminal_eoc_position(data: &[u8]) -> Result<usize> {
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
pub(super) fn marker_segment_end(
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

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;
    use crate::PixelSignedness;

    fn make_layout(rows: usize, cols: usize, bits: u16, signed: PixelSignedness) -> PixelLayout {
        PixelLayout {
            rows,
            cols,
            samples_per_pixel: 1,
            bits_allocated: if bits <= 8 { 8 } else { 16 },
            bits_stored: bits,
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
        let err = crate::jpeg_2000::decode_jpeg2000_fragment(
            &data,
            make_layout(1, 1, 8, PixelSignedness::Unsigned),
            crate::ImageReadBudget::DEFAULT,
        )
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("SOC") || format!("{err:#}").contains("0xFF4F"),
            "error must mention SOC; got: {err:#}"
        );
    }

    #[test]
    fn unsigned_dc_restoration_covers_complete_32_bit_range() {
        assert_eq!(restore_component_sample(i64::from(i32::MIN), 32, false), 0);
        assert_eq!(
            restore_component_sample(i64::from(i32::MAX), 32, false),
            i64::from(u32::MAX)
        );
        assert_eq!(
            restore_component_sample(i64::from(i32::MIN), 32, true),
            i64::from(i32::MIN)
        );
    }
}
