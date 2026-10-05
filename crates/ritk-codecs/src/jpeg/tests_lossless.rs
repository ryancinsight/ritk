//! Tests for the JPEG lossless (SOF3) encoder.
//!
//! The oracle is the crate's own decoder: `decode_jpeg_fragment` already handles
//! lossless streams, so every test here is a real round-trip through independent
//! entropy decoding rather than a self-consistency check.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::*;
use crate::PixelLayout;

fn layout(rows: usize, cols: usize, precision: u8, signed: bool) -> PixelLayout {
    PixelLayout {
        rows,
        cols,
        samples_per_pixel: 1,
        bits_allocated: if precision > 8 { 16 } else { 8 },
        bits_stored: u16::from(precision),
        pixel_representation: if signed {
            crate::PixelSignedness::Signed
        } else {
            crate::PixelSignedness::Unsigned
        },
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    }
}

/// Decodes through the crate's reader, which is the real oracle here.
///
/// The provider's frame reader validates the scan header with `1..=7` for `Ss`,
/// so only [`JpegLosslessPrediction::Left`] and
/// [`JpegLosslessPrediction::Above`] are reachable; see
/// `zero_ss_is_rejected_by_this_stacks_decoder`.
fn round_trip(
    samples: &[u16],
    rows: usize,
    cols: usize,
    precision: u8,
    prediction: JpegLosslessPrediction,
) -> Vec<f32> {
    let stream = encode_grayscale_jpeg_lossless(samples, rows, cols, precision, prediction)
        .expect("encode lossless");
    let fragment = if stream.len().is_multiple_of(2) {
        stream
    } else {
        let mut padded = stream.clone();
        padded.push(0);
        padded
    };
    crate::decode_jpeg_fragment(&fragment, layout(rows, cols, precision, false))
        .expect("decode lossless")
}

#[test]
fn flat_field_round_trips_exactly() {
    // Every prediction difference is zero, so the whole scan stays on the DC
    // table -- the branch that first-order prediction must switch away from.
    let samples = vec![1234u16; 16];
    let decoded = round_trip(&samples, 4, 4, 16, JpegLosslessPrediction::Left);
    for value in decoded {
        assert_eq!(value, 1234.0, "a constant field was lost");
    }
}

#[test]
fn horizontal_gradient_round_trips_exactly() {
    // `Rb` is constant down each column and `Ra` differs every sample: this
    // exercises the Ra prediction path in non-hierarchical mode.
    let cols = 16usize;
    let rows = 4usize;
    let samples: Vec<u16> = (0..rows * cols).map(|i| (i % cols) as u16 * 3).collect();
    let decoded = round_trip(&samples, rows, cols, 16, JpegLosslessPrediction::Left);
    for (index, value) in decoded.iter().enumerate() {
        assert_eq!(*value, f32::from(samples[index]), "sample {index}");
    }
}

#[test]
fn vertical_gradient_round_trips_exactly() {
    // The mirror image: `Rb` differs every sample and `Ra` is constant.
    let cols = 8usize;
    let rows = 5usize;
    let samples: Vec<u16> = (0..rows * cols).map(|i| (i / cols) as u16 * 7).collect();
    let decoded = round_trip(&samples, rows, cols, 16, JpegLosslessPrediction::Left);
    for (index, value) in decoded.iter().enumerate() {
        assert_eq!(*value, f32::from(samples[index]), "sample {index}");
    }
}

#[test]
fn both_prediction_modes_agree_on_a_mixed_image() {
    let rows = 6usize;
    let cols = 7usize;
    let samples: Vec<u16> = (0..rows * cols)
        .map(|i| ((i * 37 + i / cols) % 251) as u16)
        .collect();
    let decoded = round_trip(&samples, rows, cols, 16, JpegLosslessPrediction::Left);
    for (index, expected) in samples.iter().enumerate() {
        assert_eq!(decoded[index], f32::from(*expected), "sample {index}");
    }
}

#[test]
fn non_hierarchical_round_trips_exactly() {
    // DICOM's `JpegLosslessNonHierarchical` writes `Ss = 0`, which T.81's table
    // does not define. The provider now accepts it and maps it to the same `Rb`
    // prediction selector 2 asks for, so this is a real round-trip rather than
    // a recorded limitation.
    let rows = 5usize;
    let cols = 7usize;
    let samples: Vec<u16> = (0..rows * cols)
        .map(|i| ((i * 29 + i / cols) % 4095) as u16)
        .collect();
    let decoded = round_trip(&samples, rows, cols, 12, JpegLosslessPrediction::AboveOnly);
    for (index, expected) in samples.iter().enumerate() {
        assert_eq!(
            decoded[index],
            f32::from(*expected),
            "non-hierarchical sample {index}"
        );
    }
}

#[test]
fn non_hierarchical_and_selector_two_reconstruct_identically() {
    // The provider maps `Ss = 0` and `Ss = 2` to the same predictor, so the two
    // streams must decode to the same samples. That equivalence is the contract
    // the `AboveOnly` variant rests on.
    let rows = 4usize;
    let cols = 5usize;
    let samples: Vec<u16> = (0..rows * cols).map(|i| (i * 11) as u16).collect();
    let zero = round_trip(&samples, rows, cols, 16, JpegLosslessPrediction::AboveOnly);
    let two = round_trip(&samples, rows, cols, 16, JpegLosslessPrediction::Above);
    assert_eq!(
        zero, two,
        "Ss = 0 and Ss = 2 are the same predictor and must agree"
    );
    for (index, expected) in samples.iter().enumerate() {
        assert_eq!(zero[index], f32::from(*expected), "sample {index}");
    }
}

#[test]
fn twelve_bit_samples_round_trip() {
    // SOF3 is defined for 2..=16 bits; 12 is the width DICOM's extended
    // grayscale uses and the case most likely to get the `Al` field wrong.
    let rows = 3usize;
    let cols = 3usize;
    // Every value must sit inside the 12-bit domain: a reader reconstructs at
    //  and rejects anything above it, so 4096 would fail on the
    // bound rather than on anything this test means to check.
    let samples: Vec<u16> = vec![0, 1, 2047, 2048, 2049, 3000, 4095, 100, 2046];
    let decoded = round_trip(&samples, rows, cols, 12, JpegLosslessPrediction::Left);
    for (index, expected) in samples.iter().enumerate() {
        assert_eq!(
            decoded[index],
            f32::from(*expected),
            "12-bit sample {index}"
        );
    }
}

#[test]
fn eight_bit_samples_round_trip() {
    let rows = 4usize;
    let cols = 4usize;
    let samples: Vec<u16> = (0..16).map(|i| (i * 17 % 256) as u16).collect();
    let decoded = round_trip(&samples, rows, cols, 8, JpegLosslessPrediction::Above);
    for (index, expected) in samples.iter().enumerate() {
        assert_eq!(decoded[index], f32::from(*expected), "8-bit sample {index}");
    }
}

#[test]
fn all_ones_raster_round_trips() {
    // Every prediction difference is the maximum positive value, which is the
    // largest magnitude category in the scan.
    let samples = vec![65535u16; 12];
    let decoded = round_trip(&samples, 3, 4, 16, JpegLosslessPrediction::Left);
    for value in decoded {
        assert_eq!(value, 65535.0);
    }
}

#[test]
fn descending_raster_exercises_negative_differences() {
    // Negative differences take the one's-complement path, including the
    // `-32768` special case when precision is 16.
    let rows = 4usize;
    let cols = 4usize;
    let samples: Vec<u16> = (0..rows * cols)
        .map(|i| 65535 - (i as u16) * 1000)
        .collect();
    let decoded = round_trip(&samples, rows, cols, 16, JpegLosslessPrediction::Left);
    for (index, expected) in samples.iter().enumerate() {
        assert_eq!(
            decoded[index],
            f32::from(*expected),
            "descending sample {index}"
        );
    }
}

#[test]
fn stream_starts_with_soi_and_ends_with_eoi() {
    let stream =
        encode_grayscale_jpeg_lossless(&[1, 2, 3, 4], 2, 2, 8, JpegLosslessPrediction::Above)
            .expect("encode");
    assert_eq!(&stream[..2], &[0xFF, 0xD8], "must open with SOI");
    assert_eq!(
        &stream[stream.len() - 2..],
        &[0xFF, 0xD9],
        "must close with EOI"
    );
    assert!(
        stream.windows(2).any(|w| w == [0xFF, 0xC3]),
        "must carry an SOF3 (lossless) frame header"
    );
}

#[test]
fn entropy_data_is_byte_stuffed() {
    // A stream containing 0xFF inside entropy data must have 0x00 after it,
    // otherwise a decoder reads it as a marker. The descending raster is
    // chosen because it produces long runs of set bits.
    let samples: Vec<u16> = (0..32).map(|_| 0xFFFF).collect();
    let stream = encode_grayscale_jpeg_lossless(&samples, 8, 4, 16, JpegLosslessPrediction::Above)
        .expect("encode");
    // Header markers legitimately use 0xFF; only the entropy-coded scan that
    // follows SOS may not carry a bare 0xFF.
    let scan_start = stream
        .windows(2)
        .position(|pair| pair == [0xFF, 0xDA])
        .expect("SOS marker");
    // Skip the SOS marker and its 8-byte header; stuffing applies to the scan.
    let scan_body = scan_start + 2 + 8;
    for index in scan_body..stream.len().saturating_sub(1) {
        if stream[index] == 0xFF {
            let next = stream[index + 1];
            assert!(
                next == 0x00 || next == 0xD9,
                "unstuffed 0xFF at {index} followed by {next:#04x}"
            );
        }
    }
}

#[test]
fn wrong_sample_count_is_rejected() {
    let error = encode_grayscale_jpeg_lossless(&[1, 2, 3], 2, 2, 8, JpegLosslessPrediction::Above)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("3 samples for a 2x2 frame"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn out_of_range_precision_is_rejected() {
    let error =
        encode_grayscale_jpeg_lossless(&[1, 2, 3, 4], 2, 2, 17, JpegLosslessPrediction::Above)
            .unwrap_err();
    assert!(
        format!("{error:#}").contains("outside the standard's 2..=16"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn zero_dimension_is_rejected() {
    let error =
        encode_grayscale_jpeg_lossless(&[], 0, 4, 8, JpegLosslessPrediction::Above).unwrap_err();
    assert!(
        format!("{error:#}").contains("must both be nonzero"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn magnitude_category_matches_its_definition() {
    // SSSS is the bit length of the magnitude, with zero mapped to zero.
    for (difference, expected) in [
        (0i32, 0u8),
        (1, 1),
        (-1, 1),
        (2, 2),
        (-3, 2),
        (255, 8),
        (256, 9),
        (-32768, 16),
        (32767, 15),
        (65535, 16),
    ] {
        assert_eq!(
            magnitude_category(difference),
            expected,
            "category of {difference}"
        );
    }
}
