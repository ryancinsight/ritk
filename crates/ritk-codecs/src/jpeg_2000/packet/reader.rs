use anyhow::{bail, Context, Result};

use crate::dimensions::checked_pixel_count;
use crate::jpeg_2000::ebcot::decode_code_block;
use crate::jpeg_2000::quantization::{dequantize, step_size};
use crate::jpeg_2000::subband::{resolution_band_range, subband_layout, Subband};
use crate::jpeg_2000::tag_tree::TagTree;
use crate::jpeg_2000::wavelet::inverse_dwt_5_3;
use crate::jpeg_2000::wavelet_9_7::inverse_dwt_9_7;

use super::{band_cblks, cblk_grid, lblock_extra_bits, CblkRef, WaveletTransform};

/// Read individual bits (MSB first) from a borrowed §B.10.1 bit-stuffed
/// header: after a 0xFF byte, the following byte contributes only its low
/// 7 bits (= OpenJPEG `opj_bio_bytein`).
pub struct BitReader<'a> {
    bytes: &'a [u8],
    /// Next unread byte index.
    pos: usize,
    /// 16-bit sliding window (high byte = previously consumed byte).
    buf: u32,
    /// Unread bits remaining in the low byte of `buf`.
    ct: u8,
}

impl<'a> BitReader<'a> {
    /// Create a `BitReader` over a raw packet-header byte slice.
    pub fn new(raw: &'a [u8]) -> Self {
        Self {
            bytes: raw,
            pos: 0,
            buf: 0,
            ct: 0,
        }
    }

    fn bytein(&mut self) -> Result<()> {
        self.buf = (self.buf << 8) & 0xFFFF;
        self.ct = if self.buf == 0xFF00 { 7 } else { 8 };
        if self.pos < self.bytes.len() {
            self.buf |= u32::from(self.bytes[self.pos]);
            self.pos += 1;
            Ok(())
        } else {
            bail!(
                "J2K: truncated packet header while reading byte {}",
                self.pos
            )
        }
    }

    #[inline]
    pub fn read_bit(&mut self) -> Result<u32> {
        if self.ct == 0 {
            self.bytein()?;
        }
        self.ct -= 1;
        Ok((self.buf >> self.ct) & 1)
    }

    /// Read at most 32 packet-header bits, rejecting truncated input.
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        if u32::from(n) > u32::BITS {
            bail!("J2K: packet-header field width {n} exceeds 32-bit decode limit");
        }
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read_bit()?;
        }
        Ok(value)
    }

    /// Align past the header to the packet body and return its RAW byte
    /// offset (= OpenJPEG `opj_bio_inalign`): if the last consumed header
    /// byte is 0xFF, the mandatory stuffed follow byte is skipped too.
    pub fn byte_pos(&mut self) -> Result<usize> {
        if self.buf & 0xFF == 0xFF {
            self.bytein()?;
        }
        self.ct = 0;
        Ok(self.pos)
    }
}

/// Decode the number-of-passes code from a `BitReader` (ISO 15444-1 Table B.4).
pub(crate) fn read_num_passes(br: &mut BitReader<'_>) -> Result<u32> {
    if br.read_bit()? == 0 {
        return Ok(1);
    }
    if br.read_bit()? == 0 {
        return Ok(2);
    }
    let n = br.read_bits(2)?;
    if n != 3 {
        return Ok(3 + n);
    }
    let n = br.read_bits(5)?;
    if n != 31 {
        return Ok(6 + n);
    }
    Ok(37 + br.read_bits(7)?)
}

/// Per-code-block decode state accumulated across quality layers.
#[derive(Debug, Clone, Default)]
struct CblkState {
    /// Concatenated EBCOT bytes across all layers.
    data: Vec<u8>,
    /// Accumulated coding passes.
    num_passes: u32,
    /// Missing MSBs signalled at first inclusion.
    msbs: u32,
    /// Lblock state (§B.10.7.1), persists across layers.
    lblock: u8,
    /// Whether this code-block has been included in an earlier layer.
    included_before: bool,
}

/// Per-band tag trees (inclusion + missing MSBs), persistent across layers.
struct BandTrees {
    incl: TagTree,
    msbs: TagTree,
}

fn band_trees(bands: &[Subband]) -> Vec<Option<BandTrees>> {
    bands
        .iter()
        .map(|b| {
            if b.w == 0 || b.h == 0 {
                None
            } else {
                let (gw, gh) = cblk_grid(b.w, b.h);
                Some(BandTrees {
                    incl: TagTree::new(gw, gh),
                    msbs: TagTree::new(gw, gh),
                })
            }
        })
        .collect()
}

/// Decoded samples for one complete component of one tile.
#[allow(dead_code)]
pub struct TileComponentSamples {
    pub samples: Vec<i32>,
    pub width: usize,
    pub height: usize,
}

/// Tile coding parameters extracted from the COD/QCD main-header segments.
#[derive(Clone, Copy, Debug)]
pub struct TileCodingParams<'a> {
    /// Guard bits from QCD (ISO 15444-1 §A.6.4).
    pub num_guard_bits: u8,
    /// Component bit precision (Ssiz + 1).
    pub precision: u32,
    /// DWT decomposition levels from COD (§A.6.1).
    pub num_decomp_levels: u8,
    /// Quality layers from COD; must be ≥ 1.
    pub num_layers: u16,
    /// Per-subband quantizer exponents ε_b from QCD in codestream subband
    /// order; when empty (or too short) the reversible default
    /// `precision + gain_b` is used.
    pub exponents: &'a [u32],
    /// Per-subband quantizer mantissas μ_b from QCD (scalar style only); empty
    /// for the no-quantization style, where μ_b = 0.
    pub mantissas: &'a [u32],
    /// Wavelet transform family (from COD); selects the inverse DWT and whether
    /// coefficients are dequantized.
    pub transform: WaveletTransform,
}

/// Decode the tile-part body starting immediately after the SOD marker.
///
/// Supports the LRCP progression with one precinct per resolution/band,
/// 64×64 nominal code-blocks, any number of 5/3 decomposition levels, and
/// multiple quality layers (per-code-block pass accumulation).
///
/// # Errors
/// Returns an error when the tile dimensions exceed the shared decode bound,
/// a packet header is truncated or carries an unsupported length width, a
/// signalled packet-body length exceeds the available tile data, or the
/// inverse DWT geometry is inconsistent.
pub fn decode_tile_part(
    tile_data: &[u8],
    width: usize,
    height: usize,
    coding: TileCodingParams<'_>,
) -> Result<TileComponentSamples> {
    let sample_count =
        checked_pixel_count(width, height).context("J2K tile-component dimensions")?;
    let bands = subband_layout(width, height, coding.num_decomp_levels);
    let cblks: Vec<CblkRef> = bands
        .iter()
        .enumerate()
        .flat_map(|(bi, b)| band_cblks(bi, b))
        .collect();
    let mut states: Vec<CblkState> = cblks
        .iter()
        .map(|_| CblkState {
            lblock: 3,
            ..CblkState::default()
        })
        .collect();
    let mut trees = band_trees(&bands);
    // Per-band index range into the flat `cblks` list.
    let mut band_ranges = Vec::with_capacity(bands.len());
    let mut start = 0usize;
    for b in &bands {
        let n = if b.w == 0 || b.h == 0 {
            0
        } else {
            let (gw, gh) = cblk_grid(b.w, b.h);
            gw * gh
        };
        band_ranges.push(start..start + n);
        start += n;
    }

    let mut pos = 0usize;
    for layer in 0..u32::from(coding.num_layers.max(1)) {
        for r in 0..=usize::from(coding.num_decomp_levels) {
            if pos >= tile_data.len() {
                bail!(
                    "J2K: tile data ended before LRCP packet header for layer {layer}, resolution {r}"
                );
            }
            let (s, e) = resolution_band_range(r);
            let mut br = BitReader::new(&tile_data[pos..]);
            // (flat code-block index, body length) included in this packet.
            let mut included: Vec<(usize, usize)> = Vec::new();
            if br.read_bit()? == 1 {
                for bi in s..e {
                    let Some(t) = trees[bi].as_mut() else {
                        continue;
                    };
                    for ci in band_ranges[bi].clone() {
                        let c = cblks[ci];
                        let st = &mut states[ci];
                        // Inclusion (§B.10.4): tag tree before first
                        // inclusion, a single raw bit afterwards.
                        let included_now = if st.included_before {
                            br.read_bit()? == 1
                        } else {
                            t.incl.decode(&mut br, c.gx, c.gy, layer + 1)?
                        };
                        if !included_now {
                            continue;
                        }
                        if !st.included_before {
                            st.msbs = t.msbs.decode_value(&mut br, c.gx, c.gy)?;
                            st.included_before = true;
                        }
                        let np = read_num_passes(&mut br)?;
                        st.num_passes = st
                            .num_passes
                            .checked_add(np)
                            .context("J2K packet coding-pass count overflow")?;
                        while br.read_bit()? == 1 {
                            st.lblock = st
                                .lblock
                                .checked_add(1)
                                .context("J2K packet Lblock unary value overflow")?;
                        }
                        let bits = st
                            .lblock
                            .checked_add(lblock_extra_bits(np))
                            .context("J2K packet length field width overflow")?;
                        let len = br.read_bits(bits)? as usize;
                        if len == 0 {
                            bail!(
                                "J2K: included code-block declares {np} coding passes with a zero-length packet body"
                            );
                        }
                        included.push((ci, len));
                    }
                }
            }
            pos = pos
                .checked_add(br.byte_pos()?)
                .filter(|&header_end| header_end <= tile_data.len())
                .context("J2K packet-header offset exceeds tile data")?;
            for (ci, len) in included {
                let end = pos.checked_add(len).filter(|&e2| e2 <= tile_data.len());
                let Some(end) = end else {
                    bail!(
                        "J2K: packet body length {len} at offset {pos} exceeds tile data {}",
                        tile_data.len()
                    );
                };
                states[ci].data.extend_from_slice(&tile_data[pos..end]);
                pos = end;
            }
        }
    }

    // EBCOT-decode each code-block into the Mallat coefficient plane. `bp_plane`
    // records, per coefficient, the lowest bit-plane its code-block decoded so
    // the irreversible reconstruction can place the dequantized value at the
    // midpoint of the still-undecoded interval (ISO 15444-1 §E.1.1.2).
    let mut mallat = vec![0i32; sample_count];
    let mut bp_plane = vec![0u32; sample_count];
    for (ci, c) in cblks.iter().enumerate() {
        let b = &bands[c.band];
        let st = &states[ci];
        let exponent = coding
            .exponents
            .get(c.band)
            .copied()
            .unwrap_or(coding.precision + b.gain);
        // Mb = ε_b + G − 1 (ISO 15444-1 §E.1).
        let total_bp = (u32::from(coding.num_guard_bits) + exponent).saturating_sub(1);
        let num_bit_planes = if st.included_before {
            total_bp.saturating_sub(st.msbs)
        } else {
            0
        };
        let block = decode_code_block(
            &st.data,
            c.w,
            c.h,
            num_bit_planes as u8,
            st.num_passes,
            b.orient,
        )
        .with_context(|| format!("J2K: decode code-block {ci}"))?;
        for y in 0..c.h {
            let off = (b.y0 + c.y0 + y) * width + b.x0 + c.x0;
            mallat[off..off + c.w].copy_from_slice(&block.samples[y * c.w..(y + 1) * c.w]);
            for x in 0..c.w {
                bp_plane[off + x] = block.lowest_bitplane;
            }
        }
    }

    match coding.transform {
        WaveletTransform::Reversible => {
            inverse_dwt_5_3(&mut mallat, width, height, coding.num_decomp_levels)?;
        }
        WaveletTransform::Irreversible => {
            // Dequantize each subband (Δ_b from the QCD ε_b/μ_b relative to
            // R_b = precision + gain_b), inverse 9/7, then round to integers.
            // With ≥ 1 decomposition level the coefficients are continuous 9/7
            // outputs (sub-step uncertainty → midpoint reconstruction). With
            // zero levels the single LL band is exact only when Δ = 1; any
            // coarser quantizer still carries interval uncertainty.
            let mut coeffs = vec![0f32; sample_count];
            for (bi, b) in bands.iter().enumerate() {
                let r_b = coding.precision + b.gain;
                let exponent = coding.exponents.get(bi).copied().unwrap_or(r_b);
                let mantissa = coding.mantissas.get(bi).copied().unwrap_or(0);
                let delta = step_size(r_b, exponent, mantissa);
                let exact_integer =
                    coding.num_decomp_levels == 0 && exponent == r_b && mantissa == 0;
                let continuous = !exact_integer;
                for y in 0..b.h {
                    for x in 0..b.w {
                        let idx = (b.y0 + y) * width + b.x0 + x;
                        coeffs[idx] = dequantize(mallat[idx], delta, bp_plane[idx], continuous);
                    }
                }
            }
            inverse_dwt_9_7(&mut coeffs, width, height, coding.num_decomp_levels)?;
            for (m, &c) in mallat.iter_mut().zip(coeffs.iter()) {
                *m = c.round() as i32;
            }
        }
    }

    Ok(TileComponentSamples {
        samples: mallat,
        width,
        height,
    })
}
