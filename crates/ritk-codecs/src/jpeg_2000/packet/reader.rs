use anyhow::{bail, Context, Result};

use crate::jpeg_2000::ebcot::decode_code_block;
use crate::jpeg_2000::quantization::{dequantize, reversible_midpoint, step_size};
use crate::jpeg_2000::subband::{resolution_band_range, subband_layout, Subband};
use crate::jpeg_2000::tag_tree::TagTree;
use crate::jpeg_2000::wavelet::{inverse_dwt_5_3, LiftingInteger};
use crate::jpeg_2000::wavelet_9_7::{inverse_dwt_9_7, LiftingFloat};

use super::{
    band_cblks, cblk_grid, lblock_extra_bits, total_bit_planes, CblkRef, WaveletTransform,
};

mod bits;
pub(crate) use bits::{read_num_passes, BitReader};

/// Per-code-block decode state accumulated across quality layers.
#[derive(Debug, Clone, Default)]
struct CblkState {
    /// Separate cleanup and refinement segment boundary for Part 15.
    cleanup_length: u32,
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

/// Keep ordinary CT/MR tiles at their original working precision. Wide components
/// use the same packet traversal and inverse transforms with wider coefficients.
pub(in crate::jpeg_2000) trait TileCoefficient:
    LiftingInteger + Into<i64> + TryFrom<i64>
{
    type Float: LiftingFloat;
    fn floating(value: f64) -> Self::Float;
    fn rounded(value: Self::Float) -> Self;
}
impl TileCoefficient for i32 {
    type Float = f32;
    fn floating(value: f64) -> f32 {
        value as f32
    }
    fn rounded(value: f32) -> Self {
        value.round() as i32
    }
}
impl TileCoefficient for i64 {
    type Float = f64;
    fn floating(value: f64) -> f64 {
        value
    }
    fn rounded(value: f64) -> Self {
        value.round() as i64
    }
}

/// Progressive tag-tree and code-block state for one tile component.
struct PacketState {
    cblks: Vec<CblkRef>,
    states: Vec<CblkState>,
    trees: Vec<Option<BandTrees>>,
    band_ranges: Vec<std::ops::Range<usize>>,
}
impl PacketState {
    fn new(bands: &[Subband]) -> Self {
        let cblks: Vec<CblkRef> = bands
            .iter()
            .enumerate()
            .flat_map(|(bi, b)| band_cblks(bi, b))
            .collect();
        let states: Vec<CblkState> = cblks
            .iter()
            .map(|_| CblkState {
                lblock: 3,
                ..CblkState::default()
            })
            .collect();
        let trees = band_trees(bands);
        // Per-band index range into the flat `cblks` list.
        let mut band_ranges = Vec::with_capacity(bands.len());
        let mut start = 0usize;
        for b in bands {
            let n = if b.w == 0 || b.h == 0 {
                0
            } else {
                let (gw, gh) = cblk_grid(b.w, b.h);
                gw * gh
            };
            band_ranges.push(start..start + n);
            start += n;
        }

        Self {
            cblks,
            states,
            trees,
            band_ranges,
        }
    }
    fn read(
        &mut self,
        tile_data: &[u8],
        r: usize,
        layer: u32,
        high_throughput: bool,
    ) -> Result<usize> {
        let Self {
            cblks,
            states,
            trees,
            band_ranges,
        } = self;
        let mut pos = 0usize;
        if pos >= tile_data.len() {
            bail!("J2K: tile data ended before packet header for layer {layer}, resolution {r}");
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
                    let len = if high_throughput {
                        let omitted_sets = (np - 1) / 3;
                        let passes = np - 3 * omitted_sets;
                        st.num_passes = passes;
                        st.msbs = st
                            .msbs
                            .checked_add(omitted_sets)
                            .context("HT missing bit-plane overflow")?;
                        let bits = st
                            .lblock
                            .checked_add(lblock_extra_bits(3 * omitted_sets + 1))
                            .context("HT segment length width overflow")?;
                        st.cleanup_length = br.read_bits(bits)?;
                        anyhow::ensure!(
                            (2..65535).contains(&st.cleanup_length),
                            "invalid HT cleanup segment length"
                        );
                        let refinement = if passes == 1 {
                            0
                        } else {
                            br.read_bits(
                                st.lblock
                                    .checked_add(u8::from(passes == 3))
                                    .context("HT refinement length width overflow")?,
                            )?
                        };
                        anyhow::ensure!(refinement < 2047, "invalid HT refinement segment length");
                        usize::try_from(st.cleanup_length + refinement)?
                    } else {
                        let bits = st
                            .lblock
                            .checked_add(lblock_extra_bits(np))
                            .context("J2K packet length field width overflow")?;
                        usize::try_from(br.read_bits(bits)?)?
                    };
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
        Ok(pos)
    }
}

/// Validate each component packet, including zero-contribution packets. A
/// missing trailing color component must never be synthesized as a neutral channel.
pub(in crate::jpeg_2000) fn validate_component_packets(
    data: &[u8],
    width: usize,
    height: usize,
    cod: &crate::jpeg_2000::codestream::CodMarker,
    components: usize,
) -> Result<()> {
    let bands = subband_layout(width, height, cod.num_decomp_levels);
    let mut states: Vec<_> = (0..components).map(|_| PacketState::new(&bands)).collect();
    let l = usize::from(cod.num_layers);
    let r = usize::from(cod.num_decomp_levels) + 1;
    let mut at = 0;
    for n in 0..l * r * components {
        let (layer, res, component) = match cod.progression_order {
            0 => (n / (r * components), (n / components) % r, n % components),
            1 => ((n / components) % l, n / (l * components), n % components),
            2 => (n % l, n / (components * l), (n / l) % components),
            3 | 4 => (n % l, (n / l) % r, n / (r * l)),
            _ => bail!("invalid progression order"),
        };
        at += states[component].read(
            &data[at..],
            res,
            u32::try_from(layer)?,
            cod.cb_style == 0x40,
        )?;
    }
    anyhow::ensure!(at == data.len(), "trailing component packet data");
    Ok(())
}

/// Decoded samples for one complete component of one tile.
pub struct TileComponentSamples<C> {
    pub samples: Vec<C>,
    #[cfg(test)]
    pub width: usize,
    #[cfg(test)]
    pub height: usize,
}

/// Tile coding parameters extracted from the COD/QCD main-header segments.
#[derive(Clone, Copy, Debug)]
pub struct TileCodingParams<'a> {
    pub progression: u8,
    pub high_throughput: bool,
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
/// Returns an error when the tile dimensions overflow `usize`,
/// a packet header is truncated or carries an unsupported length width, a
/// signalled packet-body length exceeds the available tile data, or the
/// inverse DWT geometry is inconsistent.
pub(in crate::jpeg_2000) fn decode_tile_part<C: TileCoefficient>(
    tile_data: &[u8],
    width: usize,
    height: usize,
    coding: TileCodingParams<'_>,
) -> Result<TileComponentSamples<C>> {
    let sample_count = width
        .checked_mul(height)
        .context("J2K tile-component dimensions overflow usize")?;
    let bands = subband_layout(width, height, coding.num_decomp_levels);
    let mut packets = PacketState::new(&bands);
    let mut pos = 0usize;
    let layers = usize::from(coding.num_layers.max(1));
    let resolutions = usize::from(coding.num_decomp_levels) + 1;
    // With a single component and one precinct per resolution, RLCP, RPCL,
    // PCRL and CPRL all visit layers inside each resolution.
    for packet in 0..layers * resolutions {
        let (layer, r) = if coding.progression == 0 {
            (packet / resolutions, packet % resolutions)
        } else {
            (packet % layers, packet / layers)
        };
        let layer = u32::try_from(layer)?;
        pos += packets.read(&tile_data[pos..], r, layer, coding.high_throughput)?;
    }
    anyhow::ensure!(
        pos == tile_data.len(),
        "unconsumed JPEG 2000 tile packet bytes"
    );

    // Only the irreversible transform needs a per-coefficient plane for dequantization.
    let mut mallat = vec![C::default(); sample_count];
    let mut bp_plane = match coding.transform {
        WaveletTransform::Reversible => Vec::new(),
        WaveletTransform::Irreversible => vec![0u32; sample_count],
    };
    for (ci, c) in packets.cblks.iter().enumerate() {
        let b = &bands[c.band];
        let st = &packets.states[ci];
        let exponent = coding
            .exponents
            .get(c.band)
            .copied()
            .unwrap_or(coding.precision + b.gain);
        // Mb = ε_b + G − 1 (ISO 15444-1 §E.1).
        let total_bp = total_bit_planes(coding.num_guard_bits, exponent);
        let num_bit_planes = if st.included_before {
            total_bp.saturating_sub(st.msbs)
        } else {
            0
        };
        if coding.high_throughput {
            if st.included_before {
                let samples = super::ht_block::decode(
                    &st.data,
                    c.w,
                    c.h,
                    total_bp,
                    st.msbs,
                    st.num_passes,
                    st.cleanup_length,
                )?;
                for (y, row) in samples.chunks_exact(c.w).enumerate() {
                    let off = (b.y0 + c.y0 + y) * width + b.x0 + c.x0;
                    for (dst, &v) in mallat[off..off + c.w].iter_mut().zip(row) {
                        *dst = C::try_from(v)
                            .ok()
                            .context("HT coefficient exceeds tile precision")?;
                    }
                }
            }
            continue;
        }
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
            for (x, &q) in block.samples[y * c.w..(y + 1) * c.w].iter().enumerate() {
                let plane = block.lowest_decoded_plane(q);
                match coding.transform {
                    WaveletTransform::Reversible => {
                        mallat[off + x] = C::try_from(reversible_midpoint(q, plane))
                            .ok()
                            .context("J2K coefficient exceeds tile precision")?
                    }
                    WaveletTransform::Irreversible => {
                        mallat[off + x] = C::try_from(q)
                            .ok()
                            .context("J2K coefficient exceeds tile precision")?;
                        bp_plane[off + x] = plane;
                    }
                }
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
            let mut coeffs = vec![C::Float::default(); sample_count];
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
                        coeffs[idx] = C::floating(dequantize(
                            mallat[idx].into(),
                            delta,
                            bp_plane[idx],
                            continuous,
                        ));
                    }
                }
            }
            inverse_dwt_9_7(&mut coeffs, width, height, coding.num_decomp_levels)?;
            for (m, &c) in mallat.iter_mut().zip(coeffs.iter()) {
                *m = C::rounded(c);
            }
        }
    }

    Ok(TileComponentSamples {
        samples: mallat,
        #[cfg(test)]
        width,
        #[cfg(test)]
        height,
    })
}
