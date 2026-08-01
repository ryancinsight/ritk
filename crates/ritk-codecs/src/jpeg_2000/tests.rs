use super::*;
use crate::PixelSignedness;
use encoder::{encode_grayscale_j2k, WaveletTransform};

fn layout(rows: usize, cols: usize, bits: u16, signed: PixelSignedness) -> PixelLayout {
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

fn single_pixel_codestream() -> Vec<u8> {
    encode_grayscale_j2k(
        &[17],
        1,
        1,
        8,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid single-pixel image must encode")
}

fn insert_before_eoc(codestream: &mut Vec<u8>, bytes: &[u8]) {
    let eoc = codestream
        .windows(2)
        .rposition(|window| window == [0xFF, 0xD9])
        .expect("encoder output must end with EOC");
    codestream.splice(eoc..eoc, bytes.iter().copied());
}

fn insert_before_first_sot(codestream: &mut Vec<u8>, bytes: &[u8]) {
    let sot = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x90])
        .expect("encoder output must contain SOT");
    codestream.splice(sot..sot, bytes.iter().copied());
}

fn insert_tile_header_segment(codestream: &mut Vec<u8>, bytes: &[u8]) {
    let sot = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x90])
        .expect("encoder output must contain SOT");
    let old_psot = u32::from_be_bytes([
        codestream[sot + 6],
        codestream[sot + 7],
        codestream[sot + 8],
        codestream[sot + 9],
    ]);
    assert_ne!(old_psot, 0, "test encoder must emit a bounded tile-part");
    let inserted = u32::try_from(bytes.len()).expect("test segment length must fit u32");
    codestream.splice(sot + 12..sot + 12, bytes.iter().copied());
    codestream[sot + 6..sot + 10].copy_from_slice(
        &old_psot
            .checked_add(inserted)
            .expect("test tile-part length must not overflow")
            .to_be_bytes(),
    );
}

// ── Marker constant tests ────────────────────────────────────────────────

#[test]
fn soc_marker_constant_matches_iso_15444_1() {
    assert_eq!(SOC, 0xFF4F, "SOC must equal 0xFF4F per ISO 15444-1 §A.3.1");
    assert_eq!(SOC >> 8, 0xFF, "SOC high byte must be 0xFF");
    assert_eq!(SOC & 0xFF, 0x4F, "SOC low byte must be 0x4F");
}

#[test]
fn soi_constant_matches_jpeg_start_of_image() {
    assert_eq!(SOI, 0xFFD8, "SOI must equal 0xFFD8");
    assert_ne!(SOI, SOC, "SOI and SOC must be distinct markers");
}

// ── Codestream detection ─────────────────────────────────────────────────

#[test]
fn is_jpeg2000_codestream_detects_soc_at_byte_0() {
    assert!(is_jpeg2000_codestream(&[0xFF_u8, 0x4F, 0x00]));
}

#[test]
fn is_jpeg2000_codestream_rejects_jpeg_ls_prefix() {
    assert!(!is_jpeg2000_codestream(&[0xFF_u8, 0xD8, 0xFF, 0xF7]));
}

#[test]
fn is_jpeg2000_codestream_rejects_rle_prefix() {
    assert!(!is_jpeg2000_codestream(&[0x00_u8, 0x00, 0x00, 0x01]));
}

#[test]
fn is_jpeg2000_codestream_rejects_empty_and_single_byte() {
    assert!(!is_jpeg2000_codestream(&[]));
    assert!(!is_jpeg2000_codestream(&[0xFF]));
}

// ── Error-path tests ─────────────────────────────────────────────────────

#[test]
fn decode_returns_error_for_non_soc_prefix() {
    let fragment = [0xFF_u8, 0xD8, 0xFF, 0xF7, 0x00, 0x0B];
    let err = decode_jpeg2000_fragment(&fragment, layout(2, 2, 8, PixelSignedness::Unsigned))
        .unwrap_err();
    let msg = format!("{:#}", err);
    assert!(
        msg.contains("SOC") || msg.contains("0xFF4F") || msg.contains("FF4F"),
        "error must mention SOC marker; got: {msg}"
    );
}

#[test]
fn decode_returns_error_for_truncated_codestream() {
    let truncated = [0xFF_u8, 0x4F, 0x00];
    let err = decode_jpeg2000_fragment(&truncated, layout(4, 4, 8, PixelSignedness::Unsigned))
        .unwrap_err();
    let msg = format!("{:#}", err);
    assert!(
        msg.contains("parse")
            || msg.contains("JPEG 2000")
            || msg.contains("J2K")
            || msg.contains("SIZ")
            || msg.contains("SOC"),
        "truncated J2K codestream error must be descriptive; got: {msg}"
    );
}

#[test]
fn decode_rejects_out_of_range_sot_before_packet_decode() {
    let pixels = [17i32];
    let mut j2k = encode_grayscale_j2k(
        &pixels,
        1,
        1,
        8,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");
    let sot = j2k
        .windows(2)
        .position(|bytes| bytes == [0xFF, 0x90])
        .expect("encoder output must contain SOT");
    j2k[sot + 4..sot + 6].copy_from_slice(&1u16.to_be_bytes());

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("single-tile image must reject Isot=1");
    let msg = format!("{err:#}");
    assert!(msg.contains("Isot=1"), "got: {msg}");
    assert!(msg.contains("outside 0..0"), "got: {msg}");
}

#[test]
fn decode_rejects_tile_part_length_beyond_codestream() {
    let pixels = [17i32];
    let mut j2k = encode_grayscale_j2k(
        &pixels,
        1,
        1,
        8,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");
    let sot = j2k
        .windows(2)
        .position(|bytes| bytes == [0xFF, 0x90])
        .expect("encoder output must contain SOT");
    j2k[sot + 6..sot + 10].copy_from_slice(&u32::MAX.to_be_bytes());

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("Psot beyond the codestream must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("Psot=4294967295"), "got: {msg}");
    assert!(msg.contains("beyond terminal EOC"), "got: {msg}");
}

#[test]
fn decode_rejects_missing_eoc_after_complete_tile() {
    let mut j2k = single_pixel_codestream();
    assert_eq!(j2k.pop(), Some(0xD9));
    assert_eq!(j2k.pop(), Some(0xFF));

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("a complete tile without EOC must not return partial output");
    let message = format!("{err:#}");
    assert!(message.contains("EOC"), "got: {message}");
    assert!(message.contains("terminate"), "got: {message}");
}

#[test]
fn decode_rejects_marker_without_length_after_complete_tile() {
    let mut j2k = single_pixel_codestream();
    assert_eq!(j2k.pop(), Some(0xD9));
    assert_eq!(j2k.pop(), Some(0xFF));
    j2k.extend_from_slice(&[0xFF, 0x64]);

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("COM without Lcom must not terminate parsing successfully");
    let message = format!("{err:#}");
    assert!(message.contains("EOC"), "got: {message}");
    assert!(message.contains("terminate"), "got: {message}");
}

#[test]
fn decode_rejects_marker_length_beyond_codestream() {
    let mut j2k = single_pixel_codestream();
    insert_before_eoc(&mut j2k, &[0xFF, 0x64, 0x00, 0x08, 0x01]);

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("a marker segment extending beyond the codestream must fail");
    let message = format!("{err:#}");
    assert!(message.contains("0xFF64"), "got: {message}");
    assert!(message.contains("beyond"), "got: {message}");
}

#[test]
fn decode_rejects_marker_length_smaller_than_field() {
    let mut j2k = single_pixel_codestream();
    insert_before_eoc(&mut j2k, &[0xFF, 0x64, 0x00, 0x01]);

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("a marker length smaller than its length field must fail");
    let message = format!("{err:#}");
    assert!(message.contains("invalid length 1"), "got: {message}");
}

#[test]
fn decode_rejects_multicomponent_stream_before_packet_decode() {
    let mut j2k = single_pixel_codestream();
    let siz = j2k
        .windows(2)
        .position(|window| window == [0xFF, 0x51])
        .expect("encoder output must contain SIZ");
    let old_length = u16::from_be_bytes([j2k[siz + 2], j2k[siz + 3]]);
    let first_component = [j2k[siz + 40], j2k[siz + 41], j2k[siz + 42]];
    let siz_end = siz + 2 + usize::from(old_length);
    j2k.splice(
        siz_end..siz_end,
        first_component.into_iter().chain(first_component),
    );
    j2k[siz + 2..siz + 4].copy_from_slice(&(old_length + 6).to_be_bytes());
    j2k[siz + 38..siz + 40].copy_from_slice(&3u16.to_be_bytes());

    let mut rgb_layout = layout(1, 1, 8, PixelSignedness::Unsigned);
    rgb_layout.samples_per_pixel = 3;
    let err = decode_jpeg2000_fragment(&j2k, rgb_layout)
        .expect_err("unsupported component packet traversal must not duplicate channel zero");
    let message = format!("{err:#}");
    assert!(
        message.contains("one grayscale component"),
        "got: {message}"
    );
    assert!(message.contains("Csiz=3"), "got: {message}");
}

#[test]
fn decode_rejects_unsupported_progression_and_component_transform() {
    let j2k = single_pixel_codestream();
    let cod = j2k
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("encoder output must contain COD");

    let mut progression = j2k.clone();
    progression[cod + 5] = 1;
    let err = decode_jpeg2000_fragment(&progression, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("non-LRCP progression must not be decoded as LRCP");
    assert!(
        format!("{err:#}").contains("progression order 0"),
        "got: {err:#}"
    );

    let mut mct = j2k;
    mct[cod + 8] = 1;
    let err = decode_jpeg2000_fragment(&mct, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("MCT codestream must not bypass inverse component transformation");
    assert!(format!("{err:#}").contains("MCT=1"), "got: {err:#}");
}

#[test]
fn decode_rejects_unsupported_cod_packet_profiles() {
    let source = single_pixel_codestream();
    let cod = source
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("encoder output must contain COD");
    let old_length = u16::from_be_bytes([source[cod + 2], source[cod + 3]]);
    let mut custom_precincts = source.clone();
    custom_precincts.insert(cod + 2 + usize::from(old_length), 0xFF);
    custom_precincts[cod + 2..cod + 4].copy_from_slice(&(old_length + 1).to_be_bytes());
    custom_precincts[cod + 4] = 0x01;
    let err = decode_jpeg2000_fragment(
        &custom_precincts,
        layout(1, 1, 8, PixelSignedness::Unsigned),
    )
    .expect_err("custom precincts must not enter the one-precinct packet reader");
    assert!(format!("{err:#}").contains("Scod=0x01"), "got: {err:#}");

    let cases: &[(usize, &[u8], &str)] = &[
        (4, &[0x02], "Scod=0x02"),
        (6, &[0x00, 0x00], "zero quality layers"),
        (10, &[0x03], "32x64"),
        (12, &[0x01], "code-block style"),
        (13, &[0x02], "wavelet transform 2"),
    ];

    for &(offset, replacement, expected) in cases {
        let mut j2k = source.clone();
        j2k[cod + offset..cod + offset + replacement.len()].copy_from_slice(replacement);
        let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
            .expect_err("unsupported COD profile must not enter packet decode");
        let message = format!("{err:#}");
        assert!(
            message.contains(expected),
            "expected {expected:?}; got: {message}"
        );
    }
}

#[test]
fn decode_rejects_main_header_progression_change() {
    let mut j2k = single_pixel_codestream();
    insert_before_first_sot(&mut j2k, &[0xFF, 0x5F, 0x00, 0x02]);

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("POC must not override the validated LRCP order");
    let message = format!("{err:#}");
    assert!(message.contains("0xFF5F"), "got: {message}");
    assert!(message.contains("unsupported"), "got: {message}");
}

#[test]
fn decode_rejects_tile_header_coding_and_progression_overrides() {
    for marker in [[0xFF, 0x52, 0x00, 0x02], [0xFF, 0x5F, 0x00, 0x02]] {
        let mut j2k = single_pixel_codestream();
        insert_tile_header_segment(&mut j2k, &marker);

        let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
            .expect_err("tile-header coding overrides must not reach LRCP packet decode");
        let message = format!("{err:#}");
        assert!(
            message.contains("tile-header coding override"),
            "got: {message}"
        );
    }
}

#[test]
fn decode_validates_tile_header_segments_instead_of_scanning_payload_bytes() {
    let mut j2k = single_pixel_codestream();
    insert_tile_header_segment(&mut j2k, &[0xFF, 0x64, 0x00, 0x06, 0xFF, 0x93, 0x00, 0x00]);

    let decoded = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect("SOD bytes inside a valid COM payload must be skipped structurally");
    assert_eq!(decoded, vec![17.0]);
}

#[test]
fn decode_psot_zero_uses_terminal_eoc_after_structural_tile_header() {
    let mut j2k = single_pixel_codestream();
    insert_tile_header_segment(&mut j2k, &[0xFF, 0x64, 0x00, 0x06, 0xFF, 0xD9, 0x00, 0x00]);
    let sot = j2k
        .windows(2)
        .position(|window| window == [0xFF, 0x90])
        .expect("encoder output must contain SOT");
    j2k[sot + 6..sot + 10].copy_from_slice(&0u32.to_be_bytes());

    let decoded = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect("Psot=0 must extend past marker-looking COM payload bytes to terminal EOC");
    assert_eq!(decoded, vec![17.0]);
}

#[test]
fn decode_rejects_payload_after_eoc() {
    let mut j2k = single_pixel_codestream();
    j2k.extend_from_slice(&[0x12, 0x34]);

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("payload after EOC must not be ignored");
    assert!(format!("{err:#}").contains("EOC must terminate"));
}

#[test]
fn decode_accepts_only_required_dicom_zero_padding_after_eoc() {
    let mut odd_codestream = None;
    for value in 0..=255 {
        let candidate = encode_grayscale_j2k(
            &[value],
            1,
            1,
            8,
            PixelSignedness::Unsigned,
            0,
            WaveletTransform::Reversible,
        )
        .expect("valid single-pixel image must encode");
        if candidate.len() % 2 == 1 {
            odd_codestream = Some((candidate, value));
            break;
        }
    }
    let (mut j2k, expected) = odd_codestream.expect("test corpus must contain an odd codestream");
    j2k.push(0);

    let decoded = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect("one zero byte may pad an odd DICOM fragment value to even length");
    assert_eq!(decoded, vec![expected as f32]);
}

#[test]
fn decode_rejects_tile_header_segment_beyond_tile_part() {
    let mut j2k = single_pixel_codestream();
    insert_tile_header_segment(&mut j2k, &[0xFF, 0x64, 0xFF, 0xFF]);

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("tile-header marker length must stay inside Psot");
    let message = format!("{err:#}");
    assert!(message.contains("0xFF64"), "got: {message}");
    assert!(message.contains("beyond"), "got: {message}");
}

#[test]
fn decode_rejects_empty_packet_data_even_with_eoc() {
    let mut j2k = single_pixel_codestream();
    let sot = j2k
        .windows(2)
        .position(|window| window == [0xFF, 0x90])
        .expect("encoder output must contain SOT");
    let old_psot = u32::from_be_bytes([j2k[sot + 6], j2k[sot + 7], j2k[sot + 8], j2k[sot + 9]]);
    let old_end = sot + usize::try_from(old_psot).expect("test Psot must fit usize");
    let sod = j2k[sot + 12..old_end]
        .windows(2)
        .position(|window| window == [0xFF, 0x93])
        .map(|offset| sot + 12 + offset)
        .expect("encoder output must contain SOD");
    j2k.drain(sod + 2..old_end);
    j2k[sot + 6..sot + 10].copy_from_slice(&14u32.to_be_bytes());

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("EOC cannot make an absent LRCP packet complete");
    let message = format!("{err:#}");
    assert!(message.contains("LRCP packet header"), "got: {message}");
}

#[test]
fn decode_rejects_declared_multiple_tile_parts() {
    let mut j2k = single_pixel_codestream();
    let sot = j2k
        .windows(2)
        .position(|window| window == [0xFF, 0x90])
        .expect("encoder output must contain SOT");
    j2k[sot + 11] = 2;

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 1, 8, PixelSignedness::Unsigned))
        .expect_err("multi-part tiles must be rejected until accumulation is implemented");
    let message = format!("{err:#}");
    assert!(message.contains("one tile-part per tile"), "got: {message}");
    assert!(message.contains("TNsot=2"), "got: {message}");
}

#[test]
fn decode_rejects_missing_siz_declared_tile() {
    let mut j2k = single_pixel_codestream();
    let siz = j2k
        .windows(2)
        .position(|window| window == [0xFF, 0x51])
        .expect("encoder output must contain SIZ");
    j2k[siz + 6..siz + 10].copy_from_slice(&2u32.to_be_bytes());

    let err = decode_jpeg2000_fragment(&j2k, layout(1, 2, 8, PixelSignedness::Unsigned))
        .expect_err("EOC must not accept an omitted second tile");
    let message = format!("{err:#}");
    assert!(message.contains("tile 1 of 2"), "got: {message}");
}

// ── Lossless round-trip tests ────────────────────────────────────────────

#[test]
fn decode_jpeg2000_lossless_round_trip_4x4_uniform() {
    let rows = 4u32;
    let cols = 4u32;
    let pixel_value = 128i32;
    let pixels = vec![pixel_value; (rows * cols) as usize];
    let j2k = encode_grayscale_j2k(
        &pixels,
        rows,
        cols,
        8,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");

    assert!(
        is_jpeg2000_codestream(&j2k),
        "encoded output must start with SOC 0xFF4F; first bytes: {:02X?}",
        &j2k[..j2k.len().min(4)]
    );

    let decoded = decode_jpeg2000_fragment(&j2k, layout(4, 4, 8, PixelSignedness::Unsigned))
        .expect("lossless JPEG 2000 round-trip must succeed");

    assert_eq!(decoded.len(), (rows * cols) as usize);
    for (i, &value) in decoded.iter().enumerate() {
        assert_eq!(
            value, pixel_value as f32,
            "pixel[{i}] must round-trip exactly"
        );
    }
}

#[test]
fn decode_jpeg2000_lossless_round_trip_gradient_2x4() {
    let rows = 2u32;
    let cols = 4u32;
    let pixels: Vec<i32> = (0..8).collect();
    let j2k = encode_grayscale_j2k(
        &pixels,
        rows,
        cols,
        8,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");

    let decoded = decode_jpeg2000_fragment(&j2k, layout(2, 4, 8, PixelSignedness::Unsigned))
        .expect("gradient round-trip must succeed");

    assert_eq!(decoded.len(), pixels.len());
    for (i, (&raw, &decoded_val)) in pixels.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(decoded_val, raw as f32, "gradient pixel[{i}] must be exact");
    }
}

#[test]
fn decode_jpeg2000_signed_samples_round_trip() {
    let pixels = [-4i32, -1, 0, 3];
    let j2k = encode_grayscale_j2k(
        &pixels,
        2,
        2,
        8,
        PixelSignedness::Signed,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");

    let decoded = decode_jpeg2000_fragment(&j2k, layout(2, 2, 8, PixelSignedness::Signed))
        .expect("signed lossless JPEG 2000 round-trip must succeed");

    assert_eq!(decoded, vec![-4.0f32, -1.0, 0.0, 3.0]);
}

#[test]
fn decode_jpeg2000_lossless_rescale_applied_correctly() {
    let pixels = [100i32];
    let j2k = encode_grayscale_j2k(
        &pixels,
        1,
        1,
        8,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");
    let mut pixel_layout = layout(1, 1, 8, PixelSignedness::Unsigned);
    pixel_layout.rescale_slope = 2.0;
    pixel_layout.rescale_intercept = -1024.0;

    let decoded = decode_jpeg2000_fragment(&j2k, pixel_layout)
        .expect("single-pixel rescale test must succeed");

    assert_eq!(decoded, vec![-824.0f32]); // 100 × 2 + (−1024) = −824
}

#[test]
fn decode_jpeg2000_lossless_round_trip_unsigned_16bit() {
    let pixels: Vec<i32> = vec![
        0, 256, 512, 1024, 2048, 3071, 3584, 3840, 100, 200, 400, 800, 1600, 2400, 3000, 4095,
    ];
    let j2k = encode_grayscale_j2k(
        &pixels,
        4,
        4,
        16,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");
    let decoded = decode_jpeg2000_fragment(&j2k, layout(4, 4, 16, PixelSignedness::Unsigned))
        .expect("16-bit lossless round-trip must succeed");
    let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    assert_eq!(decoded, expected, "16-bit samples must round-trip exactly");
}

proptest::proptest! {
    /// Lossless invariant (ISO 15444-1, 5/3 reversible, 0 DWT levels):
    /// for any image and precision, |decoded − original| = 0 exactly.
    #[test]
    fn decode_jpeg2000_lossless_round_trip_random(
        rows in 1u32..9,
        cols in 1u32..9,
        precision in proptest::sample::select(vec![8u32, 12, 16]),
        signed in proptest::bool::ANY,
        num_decomp_levels in 0u8..4,
        seed in proptest::num::u64::ANY,
    ) {
        let n = (rows * cols) as usize;
        let mut state = seed | 1;
        let pixels: Vec<i32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let raw = (state >> 33) as i64;
                if signed {
                    let half = 1i64 << (precision - 1);
                    ((raw % (2 * half)) - half) as i32
                } else {
                    (raw % (1i64 << precision)) as i32
                }
            })
            .collect();
        let signedness = if signed { PixelSignedness::Signed } else { PixelSignedness::Unsigned };
        let maximum_levels =
            u8::try_from(u32::BITS - (rows.max(cols) - 1).leading_zeros())
                .expect("invariant: u32 bit width fits u8");
        let num_decomp_levels = num_decomp_levels.min(maximum_levels);
        let j2k = encode_grayscale_j2k(&pixels, rows, cols, precision, signedness, num_decomp_levels, WaveletTransform::Reversible)
            .expect("generated valid image must encode");
        let decoded = decode_jpeg2000_fragment(
            &j2k,
            layout(rows as usize, cols as usize, precision as u16, signedness),
        )
        .expect("random lossless round-trip must succeed");
        let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
        proptest::prop_assert_eq!(decoded, expected);
    }
}

/// Deterministic CT-like content: gradient + LCG noise.
fn synthetic(rows: u32, cols: u32, amplitude: i32) -> Vec<i32> {
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    (0..rows as usize * cols as usize)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 33) % 32) as i32;
            ((i as i32 * 7) + noise) % amplitude
        })
        .collect()
}

#[test]
fn decode_jpeg2000_multi_codeblock_zero_levels() {
    let (rows, cols) = (70u32, 130u32);
    let pixels = synthetic(rows, cols, 256);
    let j2k = encode_grayscale_j2k(
        &pixels,
        rows,
        cols,
        8,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");
    let decoded = decode_jpeg2000_fragment(&j2k, layout(70, 130, 8, PixelSignedness::Unsigned))
        .expect("multi-code-block LL0 round-trip must succeed");
    let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    assert_eq!(decoded, expected, "multi-code-block LL0 must be lossless");
}

#[test]
fn decode_jpeg2000_multi_codeblock_two_levels_16bit() {
    let (rows, cols) = (100u32, 150u32);
    let pixels = synthetic(rows, cols, 4096);
    let j2k = encode_grayscale_j2k(
        &pixels,
        rows,
        cols,
        16,
        PixelSignedness::Unsigned,
        2,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");
    let decoded = decode_jpeg2000_fragment(&j2k, layout(100, 150, 16, PixelSignedness::Unsigned))
        .expect("multi-code-block 2-level round-trip must succeed");
    let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    assert_eq!(decoded, expected, "multi-code-block DWT must be lossless");
}

#[test]
fn decode_jpeg2000_lossless_round_trip_two_dwt_levels_16bit() {
    let rows = 8u32;
    let cols = 12u32;
    let pixels: Vec<i32> = (0..96).map(|i| (i * 631) % 4096).collect();
    let j2k = encode_grayscale_j2k(
        &pixels,
        rows,
        cols,
        16,
        PixelSignedness::Unsigned,
        2,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");
    let decoded = decode_jpeg2000_fragment(&j2k, layout(8, 12, 16, PixelSignedness::Unsigned))
        .expect("2-level DWT lossless round-trip must succeed");
    let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    assert_eq!(decoded, expected, "2-level DWT samples must be exact");
}

#[test]
fn decode_jpeg2000_lossless_round_trip_three_dwt_levels_signed_odd_dims() {
    let rows = 7u32;
    let cols = 9u32;
    let pixels: Vec<i32> = (0..63).map(|i| ((i * 37) % 256) - 128).collect();
    let j2k = encode_grayscale_j2k(
        &pixels,
        rows,
        cols,
        8,
        PixelSignedness::Signed,
        3,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");
    let decoded = decode_jpeg2000_fragment(&j2k, layout(7, 9, 8, PixelSignedness::Signed))
        .expect("3-level DWT signed odd-dims round-trip must succeed");
    let expected: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    assert_eq!(decoded, expected, "3-level DWT samples must be exact");
}

#[test]
fn ritk_native_decoder_replaces_openjp2_backend() {
    let pixels: Vec<i32> = (0..16i32).map(|v| v * 10).collect();
    let j2k = encode_grayscale_j2k(
        &pixels,
        4,
        4,
        8,
        PixelSignedness::Unsigned,
        0,
        WaveletTransform::Reversible,
    )
    .expect("valid image must encode");
    let decoded = decode_jpeg2000_fragment(&j2k, layout(4, 4, 8, PixelSignedness::Unsigned))
        .expect("native codec round-trip must succeed");
    let max_err = pixels
        .iter()
        .zip(decoded.iter())
        .map(|(&p, &d)| (p as f32 - d).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_err, 0.0,
        "RITK-native J2K round-trip max error must be 0; got {max_err}"
    );
}

// ── Lossy 9/7 irreversible round-trips ───────────────────────────────────

#[test]
fn decode_jpeg2000_lossy_9_7_round_trip_structured_8bit() {
    let (rows, cols) = (32u32, 32u32);
    let pixels: Vec<i32> = (0..rows * cols)
        .map(|i| {
            let (x, y) = (i % cols, i / cols);
            let ramp = (x * 5 + y * 3) % 200;
            let bump = if (x as i32 - 16).pow(2) + (y as i32 - 16).pow(2) < 25 {
                50
            } else {
                0
            };
            (ramp as i32 + bump).min(255)
        })
        .collect();

    let j2k = encode_grayscale_j2k(
        &pixels,
        rows,
        cols,
        8,
        PixelSignedness::Unsigned,
        2,
        WaveletTransform::Irreversible,
    )
    .expect("valid image must encode");

    let decoded = decode_jpeg2000_fragment(
        &j2k,
        layout(rows as usize, cols as usize, 8, PixelSignedness::Unsigned),
    )
    .expect("9/7 lossy round-trip must decode");
    assert_eq!(decoded.len(), pixels.len());

    let mse: f64 = pixels
        .iter()
        .zip(&decoded)
        .map(|(&p, &d)| {
            let e = p as f64 - d as f64;
            e * e
        })
        .sum::<f64>()
        / pixels.len() as f64;
    let max_err = pixels
        .iter()
        .zip(&decoded)
        .map(|(&p, &d)| (p as f64 - d as f64).abs())
        .fold(0.0, f64::max);
    let psnr = if mse > 0.0 {
        10.0 * (255.0f64.powi(2) / mse).log10()
    } else {
        f64::INFINITY
    };
    assert!(
        psnr >= 48.0,
        "9/7 near-lossless PSNR {psnr:.2} dB too low (mse {mse:.4}, max_err {max_err})"
    );
}

#[test]
fn decode_jpeg2000_lossy_9_7_round_trip_signed_16bit() {
    let (rows, cols) = (16u32, 24u32);
    let pixels: Vec<i32> = (0..rows * cols)
        .map(|i| (i as i32 * 37 % 4000) - 2000)
        .collect();
    let j2k = encode_grayscale_j2k(
        &pixels,
        rows,
        cols,
        16,
        PixelSignedness::Signed,
        1,
        WaveletTransform::Irreversible,
    )
    .expect("valid image must encode");
    let decoded = decode_jpeg2000_fragment(
        &j2k,
        layout(rows as usize, cols as usize, 16, PixelSignedness::Signed),
    )
    .expect("signed 16-bit 9/7 round-trip must decode");
    let max_err = pixels
        .iter()
        .zip(&decoded)
        .map(|(&p, &d)| (p as f64 - d as f64).abs())
        .fold(0.0, f64::max);
    assert!(
        max_err <= 8.0,
        "signed 16-bit 9/7 max error {max_err} exceeds tolerance"
    );
}
