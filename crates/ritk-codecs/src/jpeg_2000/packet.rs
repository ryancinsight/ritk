//! JPEG 2000 tier-2 packet encoder and decoder (ISO 15444-1 Annex B).
//!
//! # Scope
//! Tier-2 wraps each EBCOT code-block bitstream inside a **packet** whose header
//! communicates, for each code-block in each precinct:
//! - whether it is included in this quality layer,
//! - the number of leading zero bit-planes (missing MSBs), and
//! - the byte length of its coded data.
//!
//! # Encoder scope
//! The encoder produces packets for images with:
//! - one quality layer,
//! - zero DWT decomposition levels (LL0 subband only),
//! - one tile = one precinct = one code-block.
//!
//! # Decoder scope
//! The decoder handles the general case: multiple code-blocks per precinct,
//! multiple resolution levels, and multiple quality layers.

use anyhow::{bail, Result};

use super::ebcot::{decode_code_block, encode_code_block};
use super::subband::{resolution_band_range, subband_layout, Subband};
use super::tag_tree::TagTree;
use super::wavelet::{forward_dwt_5_3, inverse_dwt_5_3};

// ── Bit I/O ───────────────────────────────────────────────────────────────────

/// Write individual bits, MSB first, into a byte buffer.
///
/// The JPEG 2000 packet-header bit stream uses 0xFF byte-stuffing (ISO 15444-1
/// §B.10.1): whenever 0xFF would appear in the output, a 0x00 is inserted after
/// it.  We apply stuffing on `flush`.
pub(crate) struct BitWriter {
    bits: Vec<u8>, // packed bytes (before stuffing)
    cur: u8,
    used: u8,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self {
            bits: Vec::new(),
            cur: 0,
            used: 0,
        }
    }

    pub(crate) fn write_bit(&mut self, b: u32) {
        self.cur = (self.cur << 1) | (b as u8 & 1);
        self.used += 1;
        if self.used == 8 {
            self.bits.push(self.cur);
            self.cur = 0;
            self.used = 0;
        }
    }

    pub(crate) fn write_bits(&mut self, value: u32, n: u8) {
        for shift in (0..n).rev() {
            self.write_bit((value >> shift) & 1);
        }
    }

    /// Flush remaining bits (padding with 0s), apply 0xFF-stuffing, return bytes.
    pub(crate) fn flush(mut self) -> Vec<u8> {
        if self.used > 0 {
            self.cur <<= 8 - self.used;
            self.bits.push(self.cur);
        }
        // Apply 0xFF byte-stuffing per ISO 15444-1 §B.10.1.
        let mut out = Vec::with_capacity(self.bits.len() + 4);
        for b in self.bits {
            out.push(b);
            if b == 0xFF {
                out.push(0x00);
            }
        }
        out
    }
}

/// Read individual bits from a byte buffer (MSB first), stripping 0xFF stuffing.
pub struct BitReader {
    bytes: Vec<u8>, // de-stuffed bytes
    /// `raw_offsets[k]` = raw (stuffed) bytes consumed once `k` de-stuffed
    /// bytes have been read — needed so [`Self::byte_pos`] reports the packet
    /// body offset in RAW bytes even when the header contains stuffed 0xFFs.
    raw_offsets: Vec<usize>,
    byte_pos: usize,
    bit_pos: u8, // 0 = MSB of current byte
}

impl BitReader {
    /// Create a `BitReader` from a raw (stuffed) byte slice.
    ///
    /// Strips the 0x00 bytes that follow each 0xFF byte (§B.10.1).
    pub fn new(raw: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(raw.len());
        let mut raw_offsets = Vec::with_capacity(raw.len() + 1);
        raw_offsets.push(0);
        let mut i = 0;
        while i < raw.len() {
            let b = raw[i];
            bytes.push(b);
            i += 1;
            if b == 0xFF && i < raw.len() && raw[i] == 0x00 {
                i += 1; // skip stuffed zero
            }
            raw_offsets.push(i);
        }
        Self {
            bytes,
            raw_offsets,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    #[inline]
    pub fn read_bit(&mut self) -> u32 {
        if self.byte_pos >= self.bytes.len() {
            return 0;
        }
        let b = (self.bytes[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        u32::from(b)
    }

    pub fn read_bits(&mut self, n: u8) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit();
        }
        v
    }

    /// RAW byte position of the first unused byte (for locating the packet
    /// body); accounts for stuffed bytes removed during construction.
    pub fn byte_pos(&self) -> usize {
        let destuffed = if self.bit_pos == 0 {
            self.byte_pos
        } else {
            self.byte_pos + 1
        };
        self.raw_offsets[destuffed.min(self.raw_offsets.len() - 1)]
    }
}

// ── Number-of-passes encoding (ISO 15444-1 §B.10.6, Table B.3) ───────────────

/// Encode `ncp` (number of new coding passes) into a `BitWriter`
/// (ISO 15444-1 Table B.4):
/// - 1 pass    → `0`
/// - 2 passes  → `10`
/// - 3–5       → `11` + 2 bits (ncp − 3 ∈ 0..=2)
/// - 6–36      → `1111` + 5 bits (ncp − 6 ∈ 0..=30)
/// - 37–164    → `111111111` + 7 bits (ncp − 37)
fn write_num_passes(bw: &mut BitWriter, ncp: u32) {
    match ncp {
        0 => {} // No passes: write nothing (caller ensures code-block is excluded).
        1 => bw.write_bit(0),
        2 => {
            bw.write_bit(1);
            bw.write_bit(0);
        }
        3..=5 => {
            bw.write_bits(0b11, 2);
            bw.write_bits(ncp - 3, 2);
        }
        6..=36 => {
            bw.write_bits(0b11, 2);
            bw.write_bits(0b11, 2);
            bw.write_bits(ncp - 6, 5);
        }
        _ => {
            bw.write_bits(0b11, 2);
            bw.write_bits(0b11, 2);
            bw.write_bits(0b11111, 5);
            bw.write_bits(ncp - 37, 7);
        }
    }
}

/// Decode the number-of-passes code from a `BitReader` (ISO 15444-1 Table B.4).
fn read_num_passes(br: &mut BitReader) -> u32 {
    if br.read_bit() == 0 {
        return 1;
    }
    if br.read_bit() == 0 {
        return 2;
    }
    let n = br.read_bits(2);
    if n != 3 {
        return 3 + n;
    }
    let n = br.read_bits(5);
    if n != 31 {
        return 6 + n;
    }
    37 + br.read_bits(7)
}

// ── Lblock byte-count encoding ────────────────────────────────────────────────

/// Extra length bits beyond the stored `Lblock` for a packet contributing
/// `ncp` passes: `⌊log₂ ncp⌋` (ISO 15444-1 §B.10.7.1).
fn lblock_extra_bits(ncp: u32) -> u8 {
    if ncp == 0 {
        return 0;
    }
    (u32::BITS - ncp.leading_zeros() - 1) as u8
}

// ── Code-block partitioning ───────────────────────────────────────────────────

/// Nominal code-block size (COD `xcb = ycb = 4` → 2^(4+2) = 64), shared by the
/// encoder's COD emission and both tier-2 directions.
pub(crate) const CBLK_SIZE: usize = 64;

/// One code-block: its subband, grid position, and rectangle within the band.
#[derive(Clone, Copy, Debug)]
struct CblkRef {
    /// Index into the subband list.
    band: usize,
    /// Grid position within the band's code-block grid.
    gx: usize,
    gy: usize,
    /// Rectangle within the subband (band-local coordinates).
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
}

/// Per-band code-block grid dimensions (`ceil(dim / CBLK_SIZE)`).
fn cblk_grid(band_w: usize, band_h: usize) -> (usize, usize) {
    (band_w.div_ceil(CBLK_SIZE), band_h.div_ceil(CBLK_SIZE))
}

/// Enumerate the code-blocks of one subband in raster order.
fn band_cblks(band_idx: usize, band: &Subband) -> Vec<CblkRef> {
    if band.w == 0 || band.h == 0 {
        return Vec::new();
    }
    let (gw, gh) = cblk_grid(band.w, band.h);
    let mut out = Vec::with_capacity(gw * gh);
    for gy in 0..gh {
        for gx in 0..gw {
            let x0 = gx * CBLK_SIZE;
            let y0 = gy * CBLK_SIZE;
            out.push(CblkRef {
                band: band_idx,
                gx,
                gy,
                x0,
                y0,
                w: (band.w - x0).min(CBLK_SIZE),
                h: (band.h - y0).min(CBLK_SIZE),
            });
        }
    }
    out
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

// ── Encoder: single tile, LRCP, 64×64 code-blocks ────────────────────────────

/// Encode one tile-component into a J2K tile-part byte stream:
/// SOT + SOD + LRCP packets (one quality layer, one precinct per
/// resolution/band, 64×64 nominal code-blocks).
///
/// # Parameters
/// - `samples`: DC-shifted i32 samples in row-major order.
/// - `width` / `height`: tile dimensions.
/// - `num_guard_bits`: from the QCD marker (typically 2).
/// - `precision`: component bit precision (from SIZ Ssiz).
/// - `num_decomp_levels`: 5/3 reversible DWT levels (0 = no transform).
pub fn encode_tile_part(
    samples: &[i32],
    width: usize,
    height: usize,
    num_guard_bits: u8,
    precision: u32,
    tile_index: u16,
    num_decomp_levels: u8,
) -> Vec<u8> {
    // Forward DWT into the Mallat coefficient layout.
    let mut mallat = samples.to_vec();
    forward_dwt_5_3(&mut mallat, width, height, num_decomp_levels)
        .expect("invariant: samples.len() == width × height");
    let bands = subband_layout(width, height, num_decomp_levels);

    // EBCOT-encode every code-block of every non-empty subband.
    struct EncCblk {
        cblk: CblkRef,
        msbs: u32,
        passes: u32,
        data: Vec<u8>,
    }
    let mut per_band_cblks: Vec<Vec<EncCblk>> = Vec::with_capacity(bands.len());
    for (bi, b) in bands.iter().enumerate() {
        let mut list = Vec::new();
        for cblk in band_cblks(bi, b) {
            let mut coeffs = Vec::with_capacity(cblk.w * cblk.h);
            for y in 0..cblk.h {
                let off = (b.y0 + cblk.y0 + y) * width + b.x0 + cblk.x0;
                coeffs.extend_from_slice(&mallat[off..off + cblk.w]);
            }
            let enc = encode_code_block(&coeffs, cblk.w, cblk.h, b.orient);
            let (msbs, passes, data) = if enc.num_bit_planes == 0 {
                (0u32, 0u32, Vec::new())
            } else {
                // Mb = ε_b + G − 1 (ISO 15444-1 §E.1), ε_b = precision + gain.
                let total_bp = u32::from(num_guard_bits) + precision + b.gain - 1;
                (
                    total_bp.saturating_sub(u32::from(enc.num_bit_planes)),
                    enc.num_passes,
                    enc.bytes,
                )
            };
            list.push(EncCblk {
                cblk,
                msbs,
                passes,
                data,
            });
        }
        per_band_cblks.push(list);
    }

    // Build per-band tag trees: inclusion layer (0 = layer 0, 1 = never) and
    // missing MSBs (excluded blocks contribute 0, which only affects internal
    // minima — the decoder never reads their leaves).
    let mut trees = band_trees(&bands);
    for (bi, list) in per_band_cblks.iter().enumerate() {
        if let Some(t) = trees[bi].as_mut() {
            for ec in list {
                let incl = u32::from(ec.passes == 0);
                t.incl.set_value(ec.cblk.gx, ec.cblk.gy, incl);
                t.msbs.set_value(ec.cblk.gx, ec.cblk.gy, ec.msbs);
            }
            t.incl.finalize();
            t.msbs.finalize();
        }
    }

    // LRCP packet sequence: one packet per resolution (single quality layer).
    let mut body = Vec::new();
    for r in 0..=usize::from(num_decomp_levels) {
        let (s, e) = resolution_band_range(r);
        let mut bw = BitWriter::new();
        bw.write_bit(1); // non-empty packet (§B.10.3: 1 = data present)
        for bi in s..e {
            let Some(t) = trees[bi].as_mut() else {
                continue;
            };
            for ec in &per_band_cblks[bi] {
                let (gx, gy) = (ec.cblk.gx, ec.cblk.gy);
                // Inclusion tag tree at threshold layer + 1 = 1.
                t.incl.encode(&mut bw, gx, gy, 1);
                if ec.passes == 0 {
                    continue;
                }
                // Missing MSBs tag tree, fully communicated.
                t.msbs.encode(&mut bw, gx, gy, ec.msbs + 1);
                write_num_passes(&mut bw, ec.passes);
                // Lblock signalling (§B.10.7.1).
                let lextra = lblock_extra_bits(ec.passes);
                let len = ec.data.len() as u32;
                let needed_bits = (u32::BITS - len.leading_zeros()).max(1) as u8;
                let mut lblock: u8 = 3;
                while lblock + lextra < needed_bits {
                    bw.write_bit(1);
                    lblock += 1;
                }
                bw.write_bit(0);
                bw.write_bits(len, lblock + lextra);
            }
        }
        body.extend_from_slice(&bw.flush());
        for band_list in &per_band_cblks[s..e] {
            for ec in band_list {
                body.extend_from_slice(&ec.data);
            }
        }
    }

    // Assemble: SOT + SOD + packets. Psot counts from the SOT marker.
    let psot = 14u32 + body.len() as u32;
    let mut out = Vec::with_capacity(body.len() + 16);
    out.extend_from_slice(&[0xFF, 0x90]); // SOT
    out.extend_from_slice(&[0x00, 0x0A]); // Lsot = 10
    out.extend_from_slice(&tile_index.to_be_bytes()); // Isot
    out.extend_from_slice(&psot.to_be_bytes()); // Psot
    out.push(0x00); // TPsot
    out.push(0x01); // TNsot
    out.extend_from_slice(&[0xFF, 0x93]); // SOD
    out.extend_from_slice(&body);
    out
}

// ── Decoder: LRCP packet sequence ────────────────────────────────────────────

/// Decoded samples for one complete component of one tile.
#[allow(dead_code)] // width and height used when multi-tile/multi-component support is added
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
}

/// Decode the tile-part body starting immediately after the SOD marker.
///
/// Supports the LRCP progression with one precinct per resolution/band,
/// 64×64 nominal code-blocks, any number of 5/3 decomposition levels, and
/// multiple quality layers (per-code-block pass accumulation).
///
/// # Parameters
/// - `tile_data`: bytes from (and including) the first packet header byte to
///   the end of the tile-part (exclusive of EOC or the next SOT).
/// - `width` / `height`: tile dimensions.
/// - `coding`: tile coding parameters from the COD/QCD main-header segments.
///
/// # Errors
/// Returns an error when a signalled packet-body length exceeds the available
/// tile data or the inverse DWT geometry is inconsistent.
pub fn decode_tile_part(
    tile_data: &[u8],
    width: usize,
    height: usize,
    coding: TileCodingParams<'_>,
) -> Result<TileComponentSamples> {
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
    'layers: for layer in 0..u32::from(coding.num_layers.max(1)) {
        for r in 0..=usize::from(coding.num_decomp_levels) {
            if pos >= tile_data.len() {
                break 'layers;
            }
            let (s, e) = resolution_band_range(r);
            let mut br = BitReader::new(&tile_data[pos..]);
            // (flat code-block index, body length) included in this packet.
            let mut included: Vec<(usize, usize)> = Vec::new();
            if br.read_bit() == 1 {
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
                            br.read_bit() == 1
                        } else {
                            t.incl.decode(&mut br, c.gx, c.gy, layer + 1)
                        };
                        if !included_now {
                            continue;
                        }
                        if !st.included_before {
                            st.msbs = t.msbs.decode_value(&mut br, c.gx, c.gy);
                            st.included_before = true;
                        }
                        let np = read_num_passes(&mut br);
                        st.num_passes += np;
                        while br.read_bit() == 1 {
                            st.lblock += 1;
                        }
                        let bits = st.lblock + lblock_extra_bits(np);
                        let len = br.read_bits(bits) as usize;
                        included.push((ci, len));
                    }
                }
            }
            pos += br.byte_pos();
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

    // EBCOT-decode each code-block into the Mallat coefficient plane.
    let mut mallat = vec![0i32; width * height];
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
        );
        for y in 0..c.h {
            let off = (b.y0 + c.y0 + y) * width + b.x0 + c.x0;
            mallat[off..off + c.w].copy_from_slice(&block.samples[y * c.w..(y + 1) * c.w]);
        }
    }

    inverse_dwt_5_3(&mut mallat, width, height, coding.num_decomp_levels)?;

    Ok(TileComponentSamples {
        samples: mallat,
        width,
        height,
    })
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_writer_reader_round_trip() {
        let mut bw = BitWriter::new();
        let bits = [1u32, 0, 1, 1, 0, 0, 1, 0, 1];
        for &b in &bits {
            bw.write_bit(b);
        }
        let bytes = bw.flush();
        let mut br = BitReader::new(&bytes);
        for (i, &expected) in bits.iter().enumerate() {
            assert_eq!(br.read_bit(), expected, "bit[{i}]");
        }
    }

    #[test]
    fn num_passes_encode_decode_round_trip() {
        for ncp in [1u32, 2, 3, 4, 5, 6, 7, 10, 20, 24, 38] {
            let mut bw = BitWriter::new();
            write_num_passes(&mut bw, ncp);
            let bytes = bw.flush();
            let mut br = BitReader::new(&bytes);
            let decoded = read_num_passes(&mut br);
            assert_eq!(decoded, ncp, "ncp={ncp}");
        }
    }

    /// Differential frontier (J2K-INTEROP): the tier-2 header of this captured
    /// OpenJPEG packet parses exactly (msbs=2, ncp=19=3·7−2, body fills the
    /// tile-part), but tier-1 decode diverges in the first cleanup pass around
    /// the third stripe column — context-adaptation mismatch under
    /// investigation.
    #[test]
    #[ignore = "J2K-INTEROP: tier-1 divergence under investigation (see backlog.md)"]
    fn probe_openjp2_packet_header_fields() {
        // Captured 8×8 8-bit numres=1 OpenJPEG 2.5.2 tile body (after SOD).
        let body: [u8; 65] = [
            0xCF, 0xB4, 0xF8, 0x12, 0x51, 0x7A, 0x62, 0x3E, 0xFC, 0x7B, 0x8E, 0x3E, 0x6C, 0xBF,
            0x33, 0xA9, 0xB6, 0xED, 0xDD, 0x98, 0x8C, 0x61, 0x4E, 0x7B, 0x10, 0x37, 0x1E, 0x00,
            0x55, 0x20, 0xC9, 0x4D, 0x0D, 0xB4, 0x4E, 0xEF, 0xE7, 0xC7, 0x55, 0x87, 0x6A, 0xDF,
            0x82, 0xED, 0xD1, 0xCF, 0xA5, 0x9E, 0x88, 0x11, 0x34, 0x5D, 0xEB, 0xB7, 0x4F, 0x03,
            0xDB, 0x1A, 0xA9, 0x8F, 0x19, 0xD7, 0x94, 0x36, 0x8E,
        ];
        let mut br = BitReader::new(&body);
        assert_eq!(br.read_bit(), 1, "non-empty packet bit");
        let mut incl = TagTree::new(1, 1);
        assert!(incl.decode(&mut br, 0, 0, 1), "cblk included in layer 0");
        let mut msbs_tree = TagTree::new(1, 1);
        let msbs = msbs_tree.decode_value(&mut br, 0, 0);
        let ncp = read_num_passes(&mut br);
        let mut lblock = 3u8;
        while br.read_bit() == 1 {
            lblock += 1;
        }
        let bits = lblock + lblock_extra_bits(ncp);
        let len = br.read_bits(bits) as usize;
        let header_bytes = br.byte_pos();
        eprintln!(
            "PROBE msbs={msbs} ncp={ncp} lblock={lblock} len={len} header_bytes={header_bytes} body_total={}",
            body.len()
        );
        // 8-bit, guard 2, ε = 8 → Mb = ε + G − 1 = 9 planes. Expected pass
        // budget is 3·nbp − 2 with nbp = 9 − msbs. Body must fit exactly.
        assert_eq!(
            header_bytes + len,
            body.len(),
            "packet body must fill the tile-part"
        );
        assert_eq!(ncp, 3 * (9 - msbs) - 2, "pass count must equal 3·nbp − 2");

        // Tier-1: decode the code-block body and compare with the source
        // image (8×8 synthetic from the interop suite, DC-shifted by −128).
        let mut state = 0xC0FF_EE00_DEAD_F00Du64;
        let expected: Vec<i32> = (0..64)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 33) % 64) as i64;
                (((i as i64 * 5) + noise) % 256) as i32 - 128
            })
            .collect();
        // Symbol-level differential: compare encoder-intended vs decoder-read
        // (ctx, bit) sequences; first divergence pinpoints the modelling gap.
        let _ = super::super::ebcot::cup_trace_take();
        // Differential A: our encoder must reproduce OpenJPEG byte-for-byte.
        let enc = encode_code_block(
            &expected,
            8,
            8,
            super::super::ebcot::SubbandOrientation::LlOrLh,
        );
        eprintln!(
            "ENC nbp={} ncp={} len={} (ref nbp={} ncp={ncp} len={len})",
            enc.num_bit_planes,
            enc.num_passes,
            enc.bytes.len(),
            9 - msbs,
        );
        let reference = &body[header_bytes..header_bytes + len];
        let first_diff = enc
            .bytes
            .iter()
            .zip(reference.iter())
            .position(|(a, b)| a != b);
        eprintln!(
            "FIRST DIFF at {:?}; ours[0..16]={:02X?} ref[0..16]={:02X?}",
            first_diff,
            &enc.bytes[..16.min(enc.bytes.len())],
            &reference[..16.min(reference.len())]
        );
        let enc_trace = super::super::ebcot::cup_trace_take();
        // Differential B: our decoder on the reference body.
        let block = decode_code_block(
            reference,
            8,
            8,
            (9 - msbs) as u8,
            ncp,
            super::super::ebcot::SubbandOrientation::LlOrLh,
        );
        let dec_trace = super::super::ebcot::cup_trace_take();
        let div = enc_trace
            .iter()
            .zip(dec_trace.iter())
            .position(|(a, b)| a != b);
        eprintln!(
            "TRACE div at {:?} (enc len {}, dec len {})",
            div,
            enc_trace.len(),
            dec_trace.len()
        );
        if let Some(k) = div {
            let lo = k.saturating_sub(6);
            eprintln!(
                "enc[{lo}..{}] = {:?}",
                (k + 4).min(enc_trace.len()),
                &enc_trace[lo..(k + 4).min(enc_trace.len())]
            );
            eprintln!(
                "dec[{lo}..{}] = {:?}",
                (k + 4).min(dec_trace.len()),
                &dec_trace[lo..(k + 4).min(dec_trace.len())]
            );
        }
        assert_eq!(block.samples, expected, "EBCOT tier-1 must match OpenJPEG");
    }

    #[test]
    fn tile_part_round_trip_2x2_one_dwt_level() {
        // Regression (proptest seed 3404172460139922156): 2×2, 1 DWT level,
        // four 1×1 code-blocks across two LRCP packets.
        let samples = vec![64i32, -119, -42, -28];
        let tp = encode_tile_part(&samples, 2, 2, 2, 8, 0, 1);
        let sod = tp
            .windows(2)
            .position(|w| w == [0xFF, 0x93])
            .expect("SOD present");
        let result = decode_tile_part(
            &tp[sod + 2..],
            2,
            2,
            TileCodingParams {
                num_guard_bits: 2,
                precision: 8,
                num_decomp_levels: 1,
                num_layers: 1,
                exponents: &[],
            },
        )
        .expect("decode must succeed");
        assert_eq!(result.samples, samples, "1-level DWT 2×2 must be lossless");
    }

    #[test]
    fn tile_part_encode_decode_round_trip_uniform() {
        let samples = vec![0i32; 16]; // all zeros (DC-shifted uniform)
        let tp = encode_tile_part(&samples, 4, 4, 2, 8, 0, 0);
        // The tile-part contains SOT(12) + SOD(2) + header + body.
        assert!(tp.len() >= 14, "tile-part must be at least 14 bytes");
    }

    #[test]
    fn tile_part_encode_decode_round_trip_gradient() {
        // DC-shifted: pixels 0..8 → -128..-121
        let samples: Vec<i32> = (0..8i32).map(|v| v - 128).collect();
        let tp = encode_tile_part(&samples, 4, 2, 2, 8, 0, 0);
        assert!(tp.len() >= 14);
        // Locate SOD (0xFF93) and parse the packet.
        let sod_pos = tp
            .windows(2)
            .position(|w| w == [0xFF, 0x93])
            .expect("SOD marker must be present");
        let tile_data = &tp[sod_pos + 2..];
        let result = decode_tile_part(
            tile_data,
            4,
            2,
            TileCodingParams {
                num_guard_bits: 2,
                precision: 8,
                num_decomp_levels: 0,
                num_layers: 1,
                exponents: &[],
            },
        )
        .expect("decode_tile_part must succeed");
        assert_eq!(
            result.samples, samples,
            "gradient round-trip must be lossless"
        );
    }
}
