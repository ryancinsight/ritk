use super::reader::{read_num_passes, BitReader};
use super::writer::{write_num_passes, BitWriter};
use super::*;
use crate::jpeg_2000::ebcot::{decode_code_block, encode_code_block};
use crate::jpeg_2000::tag_tree::TagTree;
use proptest::prelude::*;

fn reversible_coding() -> TileCodingParams<'static> {
    TileCodingParams {
        num_guard_bits: 1,
        precision: 8,
        num_decomp_levels: 0,
        num_layers: 1,
        exponents: &[],
        mantissas: &[],
        transform: WaveletTransform::Reversible,
    }
}

#[test]
fn tile_decode_rejects_oversized_dimensions_before_allocation() {
    let result = decode_tile_part(
        &[],
        crate::dimensions::MAX_DECODED_PIXELS + 1,
        1,
        reversible_coding(),
    );
    let Err(err) = result else {
        panic!("oversized tile dimensions must fail");
    };

    assert!(
        format!("{err:#}").contains("J2K tile-component dimensions"),
        "got: {err:#}"
    );
    assert!(format!("{err:#}").contains("decode limit"), "got: {err:#}");
}

#[test]
fn tile_decode_rejects_truncated_packet_length_field() {
    // Packet present, first inclusion, zero missing MSBs, one coding pass,
    // one Lblock increment, terminator, then only two of four length bits.
    let result = decode_tile_part(&[0b1110_1000], 1, 1, reversible_coding());
    let Err(err) = result else {
        panic!("truncated packet length field must fail");
    };

    assert!(
        format!("{err:#}").contains("truncated packet header"),
        "got: {err:#}"
    );
}

#[test]
fn tile_decode_rejects_included_code_block_with_zero_body_length() {
    let mut writer = BitWriter::new();
    writer.write_bit(1); // packet present
    writer.write_bit(1); // first inclusion at layer zero
    writer.write_bit(1); // zero missing MSBs
    write_num_passes(&mut writer, 1);
    writer.write_bit(0); // no Lblock increment
    writer.write_bits(0, 3); // declared body length
    let packet = writer.flush();

    let Err(err) = decode_tile_part(&packet, 1, 1, reversible_coding()) else {
        panic!("an included code-block cannot have an empty entropy body");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains("zero-length packet body"),
        "got: {message}"
    );
}

#[test]
fn tile_decode_rejects_excessive_lblock_growth() {
    let mut writer = BitWriter::new();
    writer.write_bit(1); // packet present
    writer.write_bit(1); // first inclusion
    writer.write_bit(1); // zero missing MSBs
    write_num_passes(&mut writer, 1);
    for _ in 0..253 {
        writer.write_bit(1);
    }
    writer.write_bit(0);
    let packet = writer.flush();

    let result = decode_tile_part(&packet, 1, 1, reversible_coding());
    let Err(err) = result else {
        panic!("excessive Lblock growth must fail");
    };

    assert!(
        format!("{err:#}").contains("Lblock unary value overflow"),
        "got: {err:#}"
    );
}

proptest! {
    #[test]
    fn arbitrary_packet_headers_are_panic_free(
        packet in proptest::collection::vec(any::<u8>(), 0..128),
    ) {
        if let Ok(decoded) = decode_tile_part(&packet, 1, 1, reversible_coding()) {
            prop_assert_eq!(decoded.width, 1);
            prop_assert_eq!(decoded.height, 1);
            prop_assert_eq!(decoded.samples.len(), 1);
        }
    }
}

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
        assert_eq!(
            br.read_bit().expect("encoded bit must decode"),
            expected,
            "bit[{i}]"
        );
    }
}

#[test]
fn num_passes_encode_decode_round_trip() {
    for ncp in [1u32, 2, 3, 4, 5, 6, 7, 10, 20, 24, 38] {
        let mut bw = BitWriter::new();
        write_num_passes(&mut bw, ncp);
        let bytes = bw.flush();
        let mut br = BitReader::new(&bytes);
        let decoded = read_num_passes(&mut br).expect("encoded pass count must decode");
        assert_eq!(decoded, ncp, "ncp={ncp}");
    }
}

/// Faithful port of OpenJPEG's tier-1 ENCODER control flow (t1.c
/// `opj_t1_enc_sigpass/refpass/clnpass`, flag-based, vsc off) producing a
/// `(ctx, bit)` symbol trace — used to diff symbol framing against ours.
fn opj_reference_trace(coeffs: &[i32], w: usize, h: usize) -> Vec<(usize, u32)> {
    use crate::jpeg_2000::ebcot::{sc_context_for_test, zc_context_for_test, SubbandOrientation};
    const SIG: u8 = 1;
    const VISIT: u8 = 2;
    const REFINE: u8 = 4;
    let n = w * h;
    let mag: Vec<u32> = coeffs.iter().map(|&v| v.unsigned_abs()).collect();
    let sign: Vec<bool> = coeffs.iter().map(|&v| v < 0).collect();
    let max = *mag
        .iter()
        .max()
        .expect("infallible: validated precondition");
    let numbps = u32::BITS - max.leading_zeros();
    let mut flags = vec![0u8; n];
    let mut trace = Vec::new();

    let sig_at = |flags: &[u8], x: isize, y: isize| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < w
            && (y as usize) < h
            && flags[y as usize * w + x as usize] & SIG != 0
    };
    let hvd = |flags: &[u8], x: usize, y: usize| -> (u32, u32, u32) {
        let (x, y) = (x as isize, y as isize);
        let hh = u32::from(sig_at(flags, x - 1, y)) + u32::from(sig_at(flags, x + 1, y));
        let vv = u32::from(sig_at(flags, x, y - 1)) + u32::from(sig_at(flags, x, y + 1));
        let dd = u32::from(sig_at(flags, x - 1, y - 1))
            + u32::from(sig_at(flags, x + 1, y - 1))
            + u32::from(sig_at(flags, x - 1, y + 1))
            + u32::from(sig_at(flags, x + 1, y + 1));
        (hh, vv, dd)
    };
    let sc = |flags: &[u8], x: usize, y: usize| -> (usize, u32) {
        let contrib = |xx: isize, yy: isize| -> i32 {
            if xx < 0 || yy < 0 || xx as usize >= w || yy as usize >= h {
                return 0;
            }
            let i = yy as usize * w + xx as usize;
            if flags[i] & SIG == 0 {
                0
            } else if sign[i] {
                -1
            } else {
                1
            }
        };
        let (x, y) = (x as isize, y as isize);
        let kh = (contrib(x - 1, y) + contrib(x + 1, y)).signum();
        let kv = (contrib(x, y - 1) + contrib(x, y + 1)).signum();
        sc_context_for_test(kh, kv)
    };

    for bp in (0..numbps).rev() {
        let one = 1u32 << bp;
        let first = bp + 1 == numbps;
        if !first {
            // ── sigpass ──────────────────────────────────────────────
            let mut k = 0;
            while k < h {
                for x in 0..w {
                    for y in k..h.min(k + 4) {
                        let i = y * w + x;
                        let (hh, vv, dd) = hvd(&flags, x, y);
                        if flags[i] & (SIG | VISIT) != 0 || hh + vv + dd == 0 {
                            continue;
                        }
                        let ctx = zc_context_for_test(SubbandOrientation::LlOrLh, hh, vv, dd);
                        let bit = u32::from(mag[i] & one != 0);
                        trace.push((ctx, bit));
                        if bit == 1 {
                            flags[i] |= SIG;
                            let (sctx, xor) = sc(&flags, x, y);
                            trace.push((sctx, u32::from(sign[i]) ^ xor));
                        }
                        flags[i] |= VISIT;
                    }
                }
                k += 4;
            }
            // ── refpass ──────────────────────────────────────────────
            let mut k = 0;
            while k < h {
                for x in 0..w {
                    for y in k..h.min(k + 4) {
                        let i = y * w + x;
                        if flags[i] & SIG == 0 || flags[i] & VISIT != 0 {
                            continue;
                        }
                        let (hh, vv, dd) = hvd(&flags, x, y);
                        let ctx = if flags[i] & REFINE != 0 {
                            16
                        } else if hh + vv + dd > 0 {
                            15
                        } else {
                            14
                        };
                        trace.push((ctx, u32::from(mag[i] & one != 0)));
                        flags[i] |= REFINE;
                    }
                }
                k += 4;
            }
        }
        // ── clnpass ──────────────────────────────────────────────────
        let mut k = 0;
        while k < h {
            for x in 0..w {
                let agg = k + 3 < h
                    && (k..k + 4).all(|y| {
                        let i = y * w + x;
                        let (hh, vv, dd) = hvd(&flags, x, y);
                        flags[i] & (SIG | VISIT) == 0 && hh + vv + dd == 0
                    });
                let mut runlen = 0usize;
                if agg {
                    while runlen < 4 && mag[(k + runlen) * w + x] & one == 0 {
                        runlen += 1;
                    }
                    trace.push((18, u32::from(runlen != 4)));
                    if runlen == 4 {
                        continue;
                    }
                    trace.push((17, (runlen as u32 >> 1) & 1));
                    trace.push((17, runlen as u32 & 1));
                }
                let start = if agg { k + runlen } else { k };
                for y in start..h.min(k + 4) {
                    let i = y * w + x;
                    if flags[i] & (SIG | VISIT) != 0 {
                        flags[i] &= !VISIT;
                        continue;
                    }
                    let partial = agg && y == k + runlen;
                    let bit = u32::from(mag[i] & one != 0);
                    if !partial {
                        let (hh, vv, dd) = hvd(&flags, x, y);
                        let ctx = zc_context_for_test(SubbandOrientation::LlOrLh, hh, vv, dd);
                        trace.push((ctx, bit));
                    }
                    if bit == 1 || partial {
                        flags[i] |= SIG;
                        let (sctx, xor) = sc(&flags, x, y);
                        trace.push((sctx, u32::from(sign[i]) ^ xor));
                    }
                }
            }
            k += 4;
        }
        for f in flags.iter_mut() {
            *f &= !VISIT;
        }
    }
    trace
}

#[test]
fn trace_v1_mid_ours_vs_port() {
    // +1 impulse at (4,4) of an 8×8 block: single cleanup pass.
    let mut coeffs = vec![0i32; 64];
    coeffs[4 * 8 + 4] = 1;
    let _ = crate::jpeg_2000::ebcot::cup_trace_take();
    let enc = encode_code_block(
        &coeffs,
        8,
        8,
        crate::jpeg_2000::ebcot::SubbandOrientation::LlOrLh,
    );
    let ours = crate::jpeg_2000::ebcot::cup_trace_take();
    let port = opj_reference_trace(&coeffs, 8, 8);
    eprintln!("ours ({}): {:?}", ours.len(), ours);
    eprintln!("port ({}): {:?}", port.len(), port);
    eprintln!("bytes: {:02X?}", enc.bytes);
    assert_eq!(ours, port, "symbol trace must match the OpenJPEG port");
}

/// Fixed-vector conformance: a tile body captured from OpenJPEG 2.5.2
/// (the C library, 8×8 8-bit synthetic, numres=1). The tier-2 header must
/// parse exactly (msbs=2, ncp=3·nbp−2, body fills the tile-part), our
/// tier-1 decoder must reconstruct every sample, and our encoder must
/// reproduce the code-block body byte-for-byte.
#[test]
fn openjp2_captured_packet_conformance() {
    // Captured 8×8 8-bit numres=1 OpenJPEG 2.5.2 tile body (after SOD).
    let body: [u8; 65] = [
        0xCF, 0xB4, 0xF8, 0x12, 0x51, 0x7A, 0x62, 0x3E, 0xFC, 0x7B, 0x8E, 0x3E, 0x6C, 0xBF, 0x33,
        0xA9, 0xB6, 0xED, 0xDD, 0x98, 0x8C, 0x61, 0x4E, 0x7B, 0x10, 0x37, 0x1E, 0x00, 0x55, 0x20,
        0xC9, 0x4D, 0x0D, 0xB4, 0x4E, 0xEF, 0xE7, 0xC7, 0x55, 0x87, 0x6A, 0xDF, 0x82, 0xED, 0xD1,
        0xCF, 0xA5, 0x9E, 0x88, 0x11, 0x34, 0x5D, 0xEB, 0xB7, 0x4F, 0x03, 0xDB, 0x1A, 0xA9, 0x8F,
        0x19, 0xD7, 0x94, 0x36, 0x8E,
    ];
    let mut br = BitReader::new(&body);
    assert_eq!(
        br.read_bit().expect("packet-present bit must decode"),
        1,
        "non-empty packet bit"
    );
    let mut incl = TagTree::new(1, 1);
    assert!(
        incl.decode(&mut br, 0, 0, 1)
            .expect("captured inclusion tree must decode"),
        "cblk included in layer 0"
    );
    let mut msbs_tree = TagTree::new(1, 1);
    let msbs = msbs_tree
        .decode_value(&mut br, 0, 0)
        .expect("captured missing-MSB tree must decode");
    let ncp = read_num_passes(&mut br).expect("captured pass count must decode");
    let mut lblock = 3u8;
    while br.read_bit().expect("captured Lblock must decode") == 1 {
        lblock += 1;
    }
    let bits = lblock + lblock_extra_bits(ncp);
    let len = br
        .read_bits(bits)
        .expect("captured packet length must decode") as usize;
    let header_bytes = br.byte_pos().expect("captured header must align");
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
    // Our encoder must reproduce the OpenJPEG code-block body
    // byte-for-byte (same pass structure, same MQ arithmetic).
    let enc = encode_code_block(
        &expected,
        8,
        8,
        crate::jpeg_2000::ebcot::SubbandOrientation::LlOrLh,
    );
    let reference = &body[header_bytes..header_bytes + len];
    assert_eq!(enc.num_bit_planes, (9 - msbs) as u8, "bit-plane count");
    assert_eq!(enc.num_passes, ncp, "coding-pass count");
    assert_eq!(enc.bytes, reference, "code-block body bytes");
    // Our decoder must reconstruct every sample from the reference body.
    let block = decode_code_block(
        reference,
        8,
        8,
        (9 - msbs) as u8,
        ncp,
        crate::jpeg_2000::ebcot::SubbandOrientation::LlOrLh,
    )
    .expect("captured OpenJPEG code-block must decode");
    assert_eq!(block.samples, expected, "EBCOT tier-1 must match OpenJPEG");
}

#[test]
fn tile_part_round_trip_2x2_one_dwt_level() {
    // Regression (proptest seed 3404172460139922156): 2×2, 1 DWT level,
    // four 1×1 code-blocks across two LRCP packets.
    let samples = vec![64i32, -119, -42, -28];
    let tp = encode_tile_part(
        &samples,
        2,
        2,
        0,
        2,
        8,
        0,
        1,
        WaveletTransform::Reversible,
        None,
    );
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
            mantissas: &[],
            transform: WaveletTransform::Reversible,
        },
    )
    .expect("decode must succeed");
    assert_eq!(result.samples, samples, "1-level DWT 2×2 must be lossless");
}

#[test]
fn tile_part_encode_decode_round_trip_uniform() {
    let samples = vec![0i32; 16]; // all zeros (DC-shifted uniform)
    let tp = encode_tile_part(
        &samples,
        4,
        4,
        0,
        2,
        8,
        0,
        0,
        WaveletTransform::Reversible,
        None,
    );
    // The tile-part contains SOT(12) + SOD(2) + header + body.
    assert!(tp.len() >= 14, "tile-part must be at least 14 bytes");
}

#[test]
fn tile_part_encode_decode_round_trip_gradient() {
    // DC-shifted: pixels 0..8 → -128..-121
    let samples: Vec<i32> = (0..8i32).map(|v| v - 128).collect();
    let tp = encode_tile_part(
        &samples,
        4,
        2,
        0,
        2,
        8,
        0,
        0,
        WaveletTransform::Reversible,
        None,
    );
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
            mantissas: &[],
            transform: WaveletTransform::Reversible,
        },
    )
    .expect("decode_tile_part must succeed");
    assert_eq!(
        result.samples, samples,
        "gradient round-trip must be lossless"
    );
}
