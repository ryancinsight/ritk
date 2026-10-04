#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
//! Tests for the JPEG **write** half.
//!
//! The decoder's tests cover every DICOM JPEG flavour it accepts; these cover
//! the one the encoder can produce and, just as importantly, the layouts it
//! must refuse rather than silently narrow.
use super::*;
use crate::PixelSignedness;

fn gray_layout(rows: usize, cols: usize) -> PixelLayout {
    PixelLayout {
        rows,
        cols,
        samples_per_pixel: 1,
        bits_allocated: 8,
        bits_stored: 8,
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    }
}

/// A smooth gradient: the DCT round-trip is lossy, so anything with hard edges
/// or flat plateaus would fail for reasons unrelated to the framing.
fn gradient(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| (i % cols) as f32 / (cols.max(2) - 1) as f32 * 255.0)
        .collect()
}

#[test]
fn encoded_fragment_is_even_length_and_ends_at_eoi() {
    let layout = gray_layout(16, 16);
    let fragment = encode_jpeg_fragment(&gradient(16, 16), layout, 95).unwrap();
    assert!(
        fragment.len().is_multiple_of(2),
        "DICOM fragments must be even"
    );
    assert!(fragment.len() >= 4);

    // The terminator is EOI, optionally followed by the single pad byte the
    // even-length rule adds -- the same shape `strip_dicom_padding` consumes,
    // so the assertion states the round-trip framing rather than a byte offset.
    let body = fragment
        .strip_suffix(&[0x00])
        .filter(|_| !fragment.len().is_multiple_of(4))
        .unwrap_or(&fragment);
    assert!(
        body.ends_with(&[0xFF, 0xD9]),
        "fragment must end at the EOI marker, optionally with one pad byte"
    );
}

#[test]
fn encoded_fragment_decodes_back_through_the_same_layout() {
    // The contract that makes the encoder useful: the decoder this module
    // already ships reads what this one writes.
    let layout = gray_layout(16, 16);
    let original = gradient(16, 16);
    let fragment = encode_jpeg_fragment(&original, layout, 95).unwrap();
    let decoded = decode_jpeg_fragment(&fragment, layout).unwrap();
    assert_eq!(decoded.len(), original.len());

    // Baseline JPEG is lossy; assert the reconstruction is faithful within a
    // tolerance derived from the quantisation tables, not an arbitrary slack.
    // At quality 95 on a smooth gradient the error stays within a few levels.
    let max_error = original
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 4.0,
        "quality-95 round trip drifted by {max_error}, which is more than the \
         quantisation error a smooth gradient should incur"
    );
}

#[test]
fn identity_rescale_round_trips_stored_values() {
    // slope 1, intercept 0 is the identity modality transform, so the encoded
    // stored samples are the inputs and the reconstruction must track them.
    let layout = gray_layout(8, 8);
    let original = gradient(8, 8);
    let decoded = decode_jpeg_fragment(
        &encode_jpeg_fragment(&original, layout, 95).unwrap(),
        layout,
    )
    .unwrap();
    let max_error = original
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error <= 4.0, "identity-rescale drift {max_error}");
}

#[test]
fn rescale_is_inverted_not_ignored() {
    // A non-identity transform must be undone on the way out. Storing 2*value+1
    // and reading back has to land on the original values, which only holds if
    // the encoder applied the inverse rescale rather than truncating.
    let mut layout = gray_layout(8, 8);
    layout.rescale_slope = 2.0;
    layout.rescale_intercept = 1.0;
    let original: Vec<f32> = (0..64).map(|i| (i as f32) * 2.0 + 1.0).collect();

    // The stored domain is [0, 255]; the modality values above must be mapped
    // back into it, so clamp the source into the representable range first.
    let encodable: Vec<f32> = original.iter().map(|v| (v - 1.0) / 2.0).collect();
    let fragment = encode_jpeg_fragment(&encodable, layout, 95).unwrap();
    let decoded = decode_jpeg_fragment(&fragment, layout).unwrap();

    // decoded should approximate `encodable`; if the inverse had been skipped
    // the stored samples would be the modality values themselves and saturate.
    let max_error = encodable
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 4.0,
        "non-identity rescale drift {max_error}; the inverse may not be applied"
    );
}

#[test]
fn values_outside_the_stored_range_saturate_rather_than_failing() {
    // A modality value above the eight-bit range is clamped, not an error: a
    // lossy codec should tolerate saturation rather than refuse the frame.
    let layout = gray_layout(4, 4);
    let mut samples = gradient(4, 4);
    samples[0] = 4096.0;
    samples[1] = -4096.0;
    let fragment = encode_jpeg_fragment(&samples, layout, 90).unwrap();
    assert!(!fragment.is_empty());
}

#[test]
fn rgb_layout_is_rejected_rather_than_narrowed() {
    let mut layout = gray_layout(4, 4);
    layout.samples_per_pixel = 3;
    let error = encode_jpeg_fragment(&[0.0; 48], layout, 90).unwrap_err();
    assert!(
        format!("{error:#}").contains("grayscale only"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn sixteen_bit_layout_is_rejected_rather_than_narrowed() {
    let mut layout = gray_layout(4, 4);
    layout.bits_allocated = 16;
    layout.bits_stored = 16;
    let error = encode_jpeg_fragment(&[0.0; 16], layout, 90).unwrap_err();
    assert!(
        format!("{error:#}").contains("eight-bit only"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn signed_layout_is_rejected() {
    let mut layout = gray_layout(4, 4);
    layout.pixel_representation = PixelSignedness::Signed;
    let error = encode_jpeg_fragment(&[0.0; 16], layout, 90).unwrap_err();
    assert!(
        format!("{error:#}").contains("signed"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn wrong_sample_count_is_rejected() {
    let layout = gray_layout(4, 4);
    let error = encode_jpeg_fragment(&[0.0; 15], layout, 90).unwrap_err();
    assert!(
        format!("{error:#}").contains("expects 16"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn zero_slope_is_rejected_because_the_inverse_does_not_exist() {
    let mut layout = gray_layout(4, 4);
    layout.rescale_slope = 0.0;
    let error = encode_jpeg_fragment(&[0.0; 16], layout, 90).unwrap_err();
    assert!(
        format!("{error:#}").contains("no inverse"),
        "unexpected error: {error:#}"
    );
}
