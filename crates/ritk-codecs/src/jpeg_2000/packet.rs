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

use super::ebcot::{decode_code_block, encode_code_block, SubbandOrientation};

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
    byte_pos: usize,
    bit_pos: u8, // 0 = MSB of current byte
}

impl BitReader {
    /// Create a `BitReader` from a raw (stuffed) byte slice.
    ///
    /// Strips the 0x00 bytes that follow each 0xFF byte (§B.10.1).
    pub fn new(raw: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(raw.len());
        let mut i = 0;
        while i < raw.len() {
            let b = raw[i];
            bytes.push(b);
            i += 1;
            if b == 0xFF && i < raw.len() && raw[i] == 0x00 {
                i += 1; // skip stuffed zero
            }
        }
        Self {
            bytes,
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

    /// Byte position of the first unused byte (for locating the packet body).
    pub fn byte_pos(&self) -> usize {
        if self.bit_pos == 0 {
            self.byte_pos
        } else {
            self.byte_pos + 1
        }
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

/// Metadata produced by the encoder for one code-block to be written into the
/// packet header.
#[allow(dead_code)] // Used when tag-tree packet header encoding is extended to multi-cblk precincts
#[derive(Debug, Clone)]
pub struct CblkPacketInfo {
    /// Number of new coding passes.
    pub num_passes: u32,
    /// Number of missing MSBs (zero bit-planes from the top).
    pub msbs: u32,
    /// Byte length of the EBCOT data.
    pub data_len: usize,
    /// The EBCOT-coded bytes.
    pub data: Vec<u8>,
}

/// Metadata decoded from a packet header for one code-block.
#[allow(dead_code)] // Used when the public packet-header API is exposed for multi-layer decode
#[derive(Debug, Clone)]
pub struct CblkHeaderInfo {
    /// Number of new coding passes in this packet.
    pub num_passes: u32,
    /// Missing MSBs (zero bit-planes from the top of the dynamic range).
    pub msbs: u32,
    /// Byte count of the coded data.
    pub data_len: usize,
}

// ── Test-only encoder: single tile, single layer, LL0 ─────────────────────────

/// Encode a single LL0 code-block into a J2K tile-part byte stream.
///
/// Produces a valid SOT + SOD + packet-header + packet-body + EOC byte sequence.
///
/// # Parameters
/// - `samples`: DC-shifted i32 samples in row-major order.
/// - `width` / `height`: dimensions.
/// - `num_guard_bits`: from the QCD marker (typically 2).
/// - `precision`: component bit precision (from SIZ Ssiz).
pub fn encode_tile_part(
    samples: &[i32],
    width: usize,
    height: usize,
    num_guard_bits: u8,
    precision: u32,
    tile_index: u16,
) -> Vec<u8> {
    let enc = encode_code_block(samples, width, height, SubbandOrientation::LlOrLh);

    // Compute packet-header fields.
    let (msbs, num_passes, cblk_data) = if enc.num_bit_planes == 0 {
        // All-zero code-block: not included.
        (0u32, 0u32, Vec::new())
    } else {
        let msbs = (num_guard_bits as u32 + precision).saturating_sub(enc.num_bit_planes as u32);
        (msbs, enc.num_passes, enc.bytes)
    };

    // Build packet header using bit I/O.
    let mut bw = BitWriter::new();
    // Zero-packet indicator: 0 = non-zero packet.
    bw.write_bit(0);

    if num_passes == 0 {
        // Empty code-block: inclusion tag tree = 1 (not in layer 0).
        // For simplicity, encode value=1 → bits: 10
        tag_tree_encode(&mut bw, 1, 0);
    } else {
        // Inclusion tag tree: value = 0 (included in layer 0).
        tag_tree_encode(&mut bw, 0, 0);
        // Missing MSBs tag tree.
        tag_tree_encode(&mut bw, msbs, 0);
        // Number of coding passes.
        write_num_passes(&mut bw, num_passes);
        // Lblock signalling (§B.10.7.1): each 1-bit increments Lblock; a 0-bit
        // terminates. Then the byte count is coded in Lblock + Lextra bits.
        let lextra = lblock_extra_bits(num_passes);
        let len = cblk_data.len() as u32;
        let needed_bits = (u32::BITS - len.leading_zeros()).max(1) as u8;
        let mut lblock: u8 = 3;
        while lblock + lextra < needed_bits {
            bw.write_bit(1);
            lblock += 1;
        }
        bw.write_bit(0);
        bw.write_bits(len, lblock + lextra);
    }

    let header_bytes = bw.flush();

    // Assemble: SOT + SOD + packet-header + packet-body + EOC.
    // Total tile-part byte count = 2(SOT) + 10(Lsot) + 2(SOD) + header + body + 2(EOC)
    // We compute psot = 12 + header.len() + body.len()  (the whole tile-part from SOT start)
    let body_len = cblk_data.len();
    let psot = 14u32 + header_bytes.len() as u32 + body_len as u32;

    let mut out = Vec::new();
    // SOT marker + segment.
    out.extend_from_slice(&[0xFF, 0x90]); // SOT
    out.extend_from_slice(&[0x00, 0x0A]); // Lsot = 10
    out.extend_from_slice(&tile_index.to_be_bytes()); // Isot
    out.extend_from_slice(&psot.to_be_bytes()); // Psot
    out.push(0x00); // TPsot
    out.push(0x01); // TNsot
                    // SOD marker.
    out.extend_from_slice(&[0xFF, 0x93]);
    // Packet header.
    out.extend_from_slice(&header_bytes);
    // Packet body.
    out.extend_from_slice(&cblk_data);

    out
}

// ── Decoder: general packet header reader ────────────────────────────────────

/// Decoded samples for one complete component of one tile.
#[allow(dead_code)] // width and height used when multi-tile/multi-component support is added
pub struct TileComponentSamples {
    pub samples: Vec<i32>,
    pub width: usize,
    pub height: usize,
}

/// Tile coding parameters extracted from the COD/QCD main-header segments.
#[derive(Clone, Copy, Debug)]
pub struct TileCodingParams {
    /// Guard bits from QCD (ISO 15444-1 §A.6.4).
    pub num_guard_bits: u8,
    /// Component bit precision (Ssiz + 1).
    pub precision: u32,
    /// DWT decomposition levels from COD (§A.6.1).
    pub num_decomp_levels: u8,
    /// Quality layers from COD; must be ≥ 1.
    pub num_layers: u16,
    /// Subband orientation (LL0 when `num_decomp_levels == 0`).
    pub orient: SubbandOrientation,
}

/// Decode the tile-part body starting immediately after the SOD marker.
///
/// Handles the simple single-layer, zero-DWT-level case produced by
/// `encode_tile_part`, and also handles real conformant JPEG 2000 tile-parts.
///
/// # Parameters
/// - `tile_data`: bytes from (and including) the first packet header byte to
///   the end of the tile-part (exclusive of EOC or the next SOT).
/// - `width` / `height`: tile dimensions.
/// - `coding`: tile coding parameters from the COD/QCD main-header segments.
///
/// # Errors
/// Returns an error if `coding.num_decomp_levels > 0` (DWT not yet supported by
/// this implementation; generalise in a future sprint).
pub fn decode_tile_part(
    tile_data: &[u8],
    width: usize,
    height: usize,
    coding: TileCodingParams,
) -> Result<TileComponentSamples> {
    let TileCodingParams {
        num_guard_bits,
        precision,
        num_decomp_levels,
        num_layers,
        orient,
    } = coding;
    if num_decomp_levels > 0 {
        bail!(
            "J2K: DWT decomposition levels > 0 not yet supported in RITK-native decoder; \
             num_decomp_levels={num_decomp_levels}"
        );
    }

    // For LL0 (no DWT), the tile has one packet per layer.
    // Decode all layers, accumulating passes.
    let mut cblk_num_passes: u32 = 0;
    let mut cblk_msbs: u32 = num_guard_bits as u32 + precision; // initial: all zero
    let mut cblk_data: Vec<u8> = Vec::new();
    let mut lblock: u8 = 3;
    let mut msbs_prev: u32 = 0;
    let mut inclusion_prev: u32 = u32::MAX;

    let mut pos = 0usize;

    for _layer in 0..num_layers {
        if pos >= tile_data.len() {
            break;
        }
        let remaining = &tile_data[pos..];
        let header = decode_packet_header(remaining, lblock, inclusion_prev, msbs_prev)?;

        pos += header.header_bytes;

        if header.included {
            // Accumulate code-block data.
            let end = pos + header.data_len;
            if end > tile_data.len() {
                bail!(
                    "J2K: packet body length {} at offset {} exceeds tile data {}",
                    header.data_len,
                    pos,
                    tile_data.len()
                );
            }
            cblk_data.extend_from_slice(&tile_data[pos..end]);
            pos = end;

            cblk_num_passes += header.num_passes;
            cblk_msbs = header.msbs;
            msbs_prev = header.msbs;
            inclusion_prev = 0; // included means layer-index = 0
            lblock = header.next_lblock;
        }
    }

    // Compute number of bit-planes from msbs.
    let total_bp = num_guard_bits as u32 + precision;
    let num_bit_planes = total_bp.saturating_sub(cblk_msbs) as u8;

    let block = decode_code_block(
        &cblk_data,
        width,
        height,
        num_bit_planes,
        cblk_num_passes,
        orient,
    );

    Ok(TileComponentSamples {
        samples: block.samples,
        width: block.width,
        height: block.height,
    })
}

// ── Packet header decoder ─────────────────────────────────────────────────────

struct PacketHeaderResult {
    /// Was this code-block included in this layer?
    included: bool,
    /// Number of new coding passes.
    num_passes: u32,
    /// Missing MSBs.
    msbs: u32,
    /// Coded data byte length.
    data_len: usize,
    /// Byte count consumed by the packet header.
    header_bytes: usize,
    /// Updated Lblock for the next layer.
    next_lblock: u8,
}

fn decode_packet_header(
    data: &[u8],
    lblock: u8,
    inclusion_prev: u32,
    msbs_prev: u32,
) -> Result<PacketHeaderResult> {
    if data.is_empty() {
        bail!("J2K: decode_packet_header called on empty slice");
    }

    let mut br = BitReader::new(data);

    // Zero-packet indicator.
    let zero_bit = br.read_bit();
    if zero_bit == 1 {
        // Empty packet.
        return Ok(PacketHeaderResult {
            included: false,
            num_passes: 0,
            msbs: msbs_prev,
            data_len: 0,
            header_bytes: br.byte_pos(),
            next_lblock: lblock,
        });
    }

    // Inclusion signalling (§B.10.4): before first inclusion the layer index is
    // coded with the inclusion tag tree (threshold starts at 0); afterwards a
    // single bit per layer signals whether new passes are present.
    let included = if inclusion_prev == u32::MAX {
        tag_tree_decode(&mut br, 0) == 0
    } else {
        br.read_bit() == 1
    };

    if !included {
        return Ok(PacketHeaderResult {
            included: false,
            num_passes: 0,
            msbs: msbs_prev,
            data_len: 0,
            header_bytes: br.byte_pos(),
            next_lblock: lblock,
        });
    }

    // First inclusion: decode missing MSBs.
    let msbs = if inclusion_prev == u32::MAX {
        tag_tree_decode(&mut br, msbs_prev)
    } else {
        msbs_prev
    };

    // Number of new coding passes.
    let num_passes = read_num_passes(&mut br);

    // Lblock increment: while next bit is 1, increment Lblock.
    let mut lb = lblock;
    while br.read_bit() == 1 {
        lb += 1;
    }

    // Compute number of bits for the byte count.
    let lextra = lblock_extra_bits(num_passes);
    let total_bits = lb + lextra;
    let data_len = br.read_bits(total_bits) as usize;

    Ok(PacketHeaderResult {
        included,
        num_passes,
        msbs,
        data_len,
        header_bytes: br.byte_pos(),
        next_lblock: lb,
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
    fn tile_part_encode_decode_round_trip_uniform() {
        let samples = vec![0i32; 16]; // all zeros (DC-shifted uniform)
        let tp = encode_tile_part(&samples, 4, 4, 2, 8, 0);
        // The tile-part contains SOT(12) + SOD(2) + header + body.
        assert!(tp.len() >= 14, "tile-part must be at least 14 bytes");
    }

    #[test]
    fn tile_part_encode_decode_round_trip_gradient() {
        // DC-shifted: pixels 0..8 → -128..-121
        let samples: Vec<i32> = (0..8i32).map(|v| v - 128).collect();
        let tp = encode_tile_part(&samples, 4, 2, 2, 8, 0);
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
                orient: SubbandOrientation::LlOrLh,
            },
        )
        .expect("decode_tile_part must succeed");
        assert_eq!(
            result.samples, samples,
            "gradient round-trip must be lossless"
        );
    }
}
