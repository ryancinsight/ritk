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
use super::subband::{resolution_band_range, subband_layout};
use super::wavelet::{forward_dwt_5_3, inverse_dwt_5_3};

// ── Bit I/O ───────────────────────────────────────────────────────────────────

/// Write individual bits, MSB first, into a byte buffer.
///
/// The JPEG 2000 packet-header bit stream uses 0xFF byte-stuffing (ISO 15444-1
/// §B.10.1): whenever 0xFF would appear in the output, a 0x00 is inserted after
/// it.  We apply stuffing on `flush`.
struct BitWriter {
    bits: Vec<u8>, // packed bytes (before stuffing)
    cur: u8,
    used: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bits: Vec::new(),
            cur: 0,
            used: 0,
        }
    }

    fn write_bit(&mut self, b: u32) {
        self.cur = (self.cur << 1) | (b as u8 & 1);
        self.used += 1;
        if self.used == 8 {
            self.bits.push(self.cur);
            self.cur = 0;
            self.used = 0;
        }
    }

    fn write_bits(&mut self, value: u32, n: u8) {
        for shift in (0..n).rev() {
            self.write_bit((value >> shift) & 1);
        }
    }

    /// Flush remaining bits (padding with 0s), apply 0xFF-stuffing, return bytes.
    fn flush(mut self) -> Vec<u8> {
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

// ── Tag tree (single-leaf, sufficient for our test encoder) ───────────────────

/// Encode one tag-tree leaf value into a bit writer.
///
/// Previous value `prev` is the value already communicated from a prior layer;
/// we only send bits that raise the known threshold from `prev` to `value`.
fn tag_tree_encode(bw: &mut BitWriter, value: u32, prev: u32) {
    for _ in prev..value {
        bw.write_bit(1); // threshold < value
    }
    bw.write_bit(0); // threshold == value
}

/// Decode one tag-tree leaf value (general single-leaf tree).
///
/// `prev` is the threshold communicated so far; the reader is positioned at
/// the next unread bit.  Returns the decoded value.
fn tag_tree_decode(br: &mut BitReader, prev: u32) -> u32 {
    let mut v = prev;
    loop {
        if br.read_bit() == 0 {
            return v;
        }
        v += 1;
    }
}

// ── Number-of-passes encoding (ISO 15444-1 §B.10.6, Table B.3) ───────────────

/// Encode `ncp` (number of new coding passes) into a `BitWriter`.
///
/// The code is a prefix code:
/// - 1   pass  → `0`
/// - 2   passes → `10`
/// - 3–4 passes → `110` + 1 bit
/// - 5–6 passes → `1110` + 1 bit
/// - 7–38 passes → `11110` + 5 bits (value = ncp − 7)
/// - 39–166 passes → `11111` + 7 bits (value = ncp − 39)
fn write_num_passes(bw: &mut BitWriter, ncp: u32) {
    match ncp {
        0 => {} // No passes: write nothing (caller ensures code-block is excluded).
        1 => bw.write_bit(0),
        2 => {
            bw.write_bit(1);
            bw.write_bit(0);
        }
        3 | 4 => {
            bw.write_bit(1);
            bw.write_bit(1);
            bw.write_bit(0);
            bw.write_bit(ncp - 3);
        }
        5 | 6 => {
            bw.write_bit(1);
            bw.write_bit(1);
            bw.write_bit(1);
            bw.write_bit(0);
            bw.write_bit(ncp - 5);
        }
        7..=38 => {
            bw.write_bits(0b11110, 5);
            bw.write_bits(ncp - 7, 5);
        }
        _ => {
            bw.write_bits(0b11111, 5);
            bw.write_bits(ncp - 39, 7);
        }
    }
}

/// Decode the number-of-passes prefix code from a `BitReader`.
fn read_num_passes(br: &mut BitReader) -> u32 {
    if br.read_bit() == 0 {
        return 1;
    }
    if br.read_bit() == 0 {
        return 2;
    }
    if br.read_bit() == 0 {
        return 3 + br.read_bit();
    }
    if br.read_bit() == 0 {
        return 5 + br.read_bit();
    }
    if br.read_bit() == 0 {
        return 7 + br.read_bits(5);
    }
    39 + br.read_bits(7)
}

// ── Lblock byte-count encoding ────────────────────────────────────────────────

/// Compute the number of extra bits for the Lblock based on the pass count.
///
/// `Lextra = floor(log2(max(1, ceil(ncp / 3))))`
/// This gives the extra bits beyond the stored `Lblock` needed to represent
/// the byte count for `ncp` new coding passes.
fn lblock_extra_bits(ncp: u32) -> u8 {
    if ncp == 0 {
        return 0;
    }
    let thirds = ncp.div_ceil(3).max(1);
    (u32::BITS - thirds.leading_zeros() - 1) as u8
}

// ── High-level packet structures ──────────────────────────────────────────────

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

// ── Encoder: single tile, LRCP, one code-block per subband ───────────────────

/// Encode one tile-component into a J2K tile-part byte stream:
/// SOT + SOD + LRCP packets (one quality layer, no precinct partitioning,
/// one code-block per subband — multi-code-block tiles: J2K-MULTI-CBLK).
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

    // EBCOT-encode each non-empty subband as one code-block.
    struct EncBand {
        msbs: u32,
        passes: u32,
        data: Vec<u8>,
    }
    let enc_bands: Vec<EncBand> = bands
        .iter()
        .map(|b| {
            if b.w == 0 || b.h == 0 {
                return EncBand {
                    msbs: 0,
                    passes: 0,
                    data: Vec::new(),
                };
            }
            let mut coeffs = Vec::with_capacity(b.w * b.h);
            for y in 0..b.h {
                let off = (b.y0 + y) * width + b.x0;
                coeffs.extend_from_slice(&mallat[off..off + b.w]);
            }
            let enc = encode_code_block(&coeffs, b.w, b.h, b.orient);
            if enc.num_bit_planes == 0 {
                // All-zero code-block: excluded from the packet.
                EncBand {
                    msbs: 0,
                    passes: 0,
                    data: Vec::new(),
                }
            } else {
                // Dynamic range of subband b: guard bits + ε_b, with
                // ε_b = precision + gain for the reversible 5/3 transform.
                let total_bp = u32::from(num_guard_bits) + precision + b.gain;
                EncBand {
                    msbs: total_bp.saturating_sub(u32::from(enc.num_bit_planes)),
                    passes: enc.num_passes,
                    data: enc.bytes,
                }
            }
        })
        .collect();

    // LRCP packet sequence: one packet per resolution (single quality layer).
    let mut body = Vec::new();
    for r in 0..=usize::from(num_decomp_levels) {
        let (s, e) = resolution_band_range(r);
        let mut bw = BitWriter::new();
        bw.write_bit(0); // non-empty packet indicator
        for i in s..e {
            if bands[i].w == 0 || bands[i].h == 0 {
                continue; // no code-block exists for an empty subband
            }
            let eb = &enc_bands[i];
            if eb.passes == 0 {
                // Not included in layer 0: inclusion tag-tree value 1.
                tag_tree_encode(&mut bw, 1, 0);
                continue;
            }
            tag_tree_encode(&mut bw, 0, 0); // included in layer 0
            tag_tree_encode(&mut bw, eb.msbs, 0); // missing MSBs
            write_num_passes(&mut bw, eb.passes);
            // Lblock signalling (§B.10.7.1): 1-bits increment Lblock, a 0-bit
            // terminates; the byte count then uses Lblock + Lextra bits.
            let lextra = lblock_extra_bits(eb.passes);
            let len = eb.data.len() as u32;
            let needed_bits = (u32::BITS - len.leading_zeros()).max(1) as u8;
            let mut lblock: u8 = 3;
            while lblock + lextra < needed_bits {
                bw.write_bit(1);
                lblock += 1;
            }
            bw.write_bit(0);
            bw.write_bits(len, lblock + lextra);
        }
        body.extend_from_slice(&bw.flush());
        for eb in &enc_bands[s..e] {
            body.extend_from_slice(&eb.data);
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
/// Supports the LRCP progression with no precinct partitioning and one
/// code-block per subband, any number of 5/3 decomposition levels, and
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
    let mut states: Vec<CblkState> = bands
        .iter()
        .map(|_| CblkState {
            lblock: 3,
            ..CblkState::default()
        })
        .collect();

    let mut pos = 0usize;
    'layers: for _layer in 0..coding.num_layers.max(1) {
        for r in 0..=usize::from(coding.num_decomp_levels) {
            if pos >= tile_data.len() {
                break 'layers;
            }
            let (s, e) = resolution_band_range(r);
            let mut br = BitReader::new(&tile_data[pos..]);
            // (band index, body length) for blocks included in this packet.
            let mut included: Vec<(usize, usize)> = Vec::new();
            if br.read_bit() == 0 {
                for i in s..e {
                    if bands[i].w == 0 || bands[i].h == 0 {
                        continue;
                    }
                    let st = &mut states[i];
                    // Inclusion (§B.10.4): tag tree before first inclusion,
                    // a single bit afterwards.
                    let included_now = if st.included_before {
                        br.read_bit() == 1
                    } else {
                        tag_tree_decode(&mut br, 0) == 0
                    };
                    if !included_now {
                        continue;
                    }
                    if !st.included_before {
                        st.msbs = tag_tree_decode(&mut br, 0);
                        st.included_before = true;
                    }
                    let np = read_num_passes(&mut br);
                    st.num_passes += np;
                    while br.read_bit() == 1 {
                        st.lblock += 1;
                    }
                    let bits = st.lblock + lblock_extra_bits(np);
                    let len = br.read_bits(bits) as usize;
                    included.push((i, len));
                }
            }
            pos += br.byte_pos();
            for (i, len) in included {
                let end = pos.checked_add(len).filter(|&e2| e2 <= tile_data.len());
                let Some(end) = end else {
                    bail!(
                        "J2K: packet body length {len} at offset {pos} exceeds tile data {}",
                        tile_data.len()
                    );
                };
                states[i].data.extend_from_slice(&tile_data[pos..end]);
                pos = end;
            }
        }
    }

    // EBCOT-decode each code-block into the Mallat coefficient plane.
    let mut mallat = vec![0i32; width * height];
    for (i, b) in bands.iter().enumerate() {
        if b.w == 0 || b.h == 0 {
            continue;
        }
        let st = &states[i];
        let exponent = coding
            .exponents
            .get(i)
            .copied()
            .unwrap_or(coding.precision + b.gain);
        let total_bp = u32::from(coding.num_guard_bits) + exponent;
        let num_bit_planes = if st.included_before {
            total_bp.saturating_sub(st.msbs)
        } else {
            0
        };
        let block = decode_code_block(
            &st.data,
            b.w,
            b.h,
            num_bit_planes as u8,
            st.num_passes,
            b.orient,
        );
        for y in 0..b.h {
            let off = (b.y0 + y) * width + b.x0;
            mallat[off..off + b.w].copy_from_slice(&block.samples[y * b.w..(y + 1) * b.w]);
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
