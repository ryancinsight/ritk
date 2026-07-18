use crate::jpeg_2000::ebcot::encode_code_block;
use crate::jpeg_2000::quantization::quantize;
use crate::jpeg_2000::subband::{resolution_band_range, subband_layout};
use crate::jpeg_2000::wavelet::forward_dwt_5_3;
use crate::jpeg_2000::wavelet_9_7::forward_dwt_9_7;

use super::{band_cblks, band_trees, lblock_extra_bits, CblkRef, WaveletTransform};

/// Write individual bits, MSB first, into a byte buffer.
///
/// The JPEG 2000 packet-header bit stream uses **bit**-stuffing (ISO 15444-1
/// Â§B.10.1, = OpenJPEG `opj_bio_byteout`): a byte following 0xFF carries only
/// 7 payload bits (its MSB is a stuffed 0), so 0xFF can never be followed by a
/// byte with the MSB set. This is not byte-stuffing â€” no full 0x00 is inserted.
pub(crate) struct BitWriter {
    out: Vec<u8>,
    /// 16-bit sliding window: bits accumulate in the low byte; the high byte
    /// is the previously completed byte (drives the 7-bit follow rule).
    buf: u32,
    /// Bits still available in the current byte (7 after emitting 0xFF).
    ct: u8 }

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self {
            out: Vec::new(),
            buf: 0,
            ct: 8 }
    }

    fn byteout(&mut self) {
        self.buf = (self.buf << 8) & 0xFFFF;
        self.ct = if self.buf == 0xFF00 { 7 } else { 8 };
        self.out.push((self.buf >> 8) as u8);
    }

    pub(crate) fn write_bit(&mut self, b: u32) {
        if self.ct == 0 {
            self.byteout();
        }
        self.ct -= 1;
        self.buf |= (b & 1) << self.ct;
    }

    pub(crate) fn write_bits(&mut self, value: u32, n: u8) {
        for shift in (0..n).rev() {
            self.write_bit((value >> shift) & 1);
        }
    }

    /// Flush remaining bits (padding with 0s) and return the header bytes
    /// (= OpenJPEG `opj_bio_flush`): if the final byte is 0xFF, one extra
    /// 7-bit byte is emitted so the header never ends on 0xFF.
    pub(crate) fn flush(mut self) -> Vec<u8> {
        self.byteout();
        if self.ct == 7 {
            self.byteout();
        }
        self.out
    }
}

/// Encode `ncp` (number of new coding passes) into a `BitWriter`
/// (ISO 15444-1 Table B.4):
/// - 1 pass    â†’ `0`
/// - 2 passes  â†’ `10`
/// - 3â€“5       â†’ `11` + 2 bits (ncp âˆ’ 3 âˆˆ 0..=2)
/// - 6â€“36      â†’ `1111` + 5 bits (ncp âˆ’ 6 âˆˆ 0..=30)
/// - 37â€“164    â†’ `111111111` + 7 bits (ncp âˆ’ 37)
pub(crate) fn write_num_passes(bw: &mut BitWriter, ncp: u32) {
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

/// Encode one tile-component into a J2K tile-part byte stream:
/// SOT + SOD + LRCP packets (one quality layer, one precinct per
/// resolution/band, 64Ã—64 nominal code-blocks).
///
/// # Parameters
/// - `samples`: DC-shifted i32 samples in row-major order.
/// - `width` / `height`: tile dimensions.
/// - `num_guard_bits`: from the QCD marker (typically 2).
/// - `precision`: component bit precision (from SIZ Ssiz).
/// - `num_decomp_levels`: 5/3 reversible DWT levels (0 = no transform).
#[allow(clippy::too_many_arguments)]
pub fn encode_tile_part(
    samples: &[i32],
    width: usize,
    height: usize,
    num_guard_bits: u8,
    precision: u32,
    tile_index: u16,
    num_decomp_levels: u8,
    transform: WaveletTransform,
) -> Vec<u8> {
    // Forward DWT into the Mallat coefficient layout.  The irreversible 9/7
    // path transforms in floating point and then dead-zone quantizes with a
    // unit step (Î”_b = 1, i.e. Îµ_b = R_b = precision + gain_b, Î¼_b = 0) so the
    // quantized integer coefficients reuse the same Mb = G + Îµ_b âˆ’ 1 bit-plane
    // budget and entropy-coding path as the reversible 5/3 transform.
    let mallat = match transform {
        WaveletTransform::Reversible => {
            let mut m = samples.to_vec();
            forward_dwt_5_3(&mut m, width, height, num_decomp_levels)
                .expect("invariant: samples.len() == width Ã— height");
            m
        }
        WaveletTransform::Irreversible => {
            let mut f: Vec<f32> = samples.iter().map(|&v| v as f32).collect();
            forward_dwt_9_7(&mut f, width, height, num_decomp_levels)
                .expect("invariant: samples.len() == width Ã— height");
            f.iter().map(|&c| quantize(c, 1.0)).collect()
        }
    };
    let bands = subband_layout(width, height, num_decomp_levels);

    // EBCOT-encode every code-block of every non-empty subband.
    struct EncCblk {
        cblk: CblkRef,
        msbs: u32,
        passes: u32,
        data: Vec<u8> }
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
                // Mb = Îµ_b + G âˆ’ 1 (ISO 15444-1 Â§E.1), Îµ_b = precision + gain.
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
                data });
        }
        per_band_cblks.push(list);
    }

    // Build per-band tag trees: inclusion layer (0 = layer 0, 1 = never) and
    // missing MSBs (excluded blocks contribute 0, which only affects internal
    // minima â€” the decoder never reads their leaves).
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
        bw.write_bit(1); // non-empty packet (Â§B.10.3: 1 = data present)
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
                // Lblock signalling (Â§B.10.7.1).
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
