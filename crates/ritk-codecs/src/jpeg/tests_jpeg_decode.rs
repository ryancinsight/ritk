#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;
use crate::PixelSignedness;

fn layout(rows: usize, cols: usize, slope: f32, intercept: f32) -> PixelLayout {
    layout_with_precision(
        rows,
        cols,
        8,
        8,
        PixelSignedness::Unsigned,
        slope,
        intercept,
    )
}

fn layout_with_precision(
    rows: usize,
    cols: usize,
    bits_allocated: u16,
    bits_stored: u16,
    pixel_representation: PixelSignedness,
    slope: f32,
    intercept: f32,
) -> PixelLayout {
    PixelLayout {
        rows,
        cols,
        samples_per_pixel: 1,
        bits_allocated,
        bits_stored,
        pixel_representation,
        rescale_slope: slope,
        rescale_intercept: intercept,
    }
}

fn encode_grayscale_jpeg(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    jpeg::encode_gray(pixels, width, height, 95).expect("test JPEG encode must succeed")
}

fn encode_rgb_jpeg(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    use image::{DynamicImage, RgbImage};
    use std::io::Cursor;

    let rgb = RgbImage::from_raw(width, height, pixels.to_vec())
        .expect("test RGB image dimensions must match sample count");
    let mut jpeg = Vec::with_capacity((width as usize * height as usize * 3) / 8);
    DynamicImage::ImageRgb8(rgb)
        .write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
        .expect("test JPEG encode must succeed");
    jpeg
}

fn reference_grayscale(jpeg: &[u8], slope: f32, intercept: f32) -> Vec<f32> {
    image::load_from_memory(jpeg)
        .expect("independent decoder must read the grayscale fixture")
        .to_luma8()
        .into_raw()
        .into_iter()
        .map(f32::from)
        .map(|sample| sample * slope + intercept)
        .collect()
}

fn reference_rgb(jpeg: &[u8]) -> Vec<f32> {
    image::load_from_memory(jpeg)
        .expect("independent decoder must read the RGB fixture")
        .to_rgb8()
        .into_raw()
        .into_iter()
        .map(f32::from)
        .collect()
}

fn lossless_single_pixel_jpeg_8bit_gray_128() -> Vec<u8> {
    vec![
        0xFF, 0xD8, 0xFF, 0xC3, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01,
        0x00, 0x00, 0x7F, 0xFF, 0xD9,
    ]
}

fn lossless_single_pixel_jpeg_16bit_gray_0x1234() -> Vec<u8> {
    vec![
        0xFF, 0xD8, 0xFF, 0xC3, 0x00, 0x0B, 0x10, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x01,
        0x00, 0x00, 0x12, 0x33, 0xFF, 0xD9,
    ]
}

fn lossless_single_pixel_midpoint(precision: u8) -> Vec<u8> {
    vec![
        0xFF, 0xD8, 0xFF, 0xC3, 0x00, 0x0B, precision, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11,
        0x00, 0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00,
        0x01, 0x00, 0x00, 0x7F, 0xFF, 0xD9,
    ]
}

fn dct_twelve_midpoint(component_ids: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8, 0xff, 0xdb, 0x00, 0x83, 0x10];
    for _ in 0..64 {
        bytes.extend_from_slice(&1_u16.to_be_bytes());
    }
    let frame_length =
        u16::try_from(8 + 3 * component_ids.len()).expect("invariant: test frame length fits u16");
    bytes.extend_from_slice(&[0xff, 0xc1]);
    bytes.extend_from_slice(&frame_length.to_be_bytes());
    bytes.extend_from_slice(&[12, 0, 8, 0, 8]);
    bytes.push(u8::try_from(component_ids.len()).expect("invariant: test component count fits u8"));
    for &id in component_ids {
        bytes.extend_from_slice(&[id, 0x11, 0]);
    }
    bytes.extend_from_slice(&[
        0xff, 0xc4, 0x00, 0x26, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
        0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    let scan_length =
        u16::try_from(6 + 2 * component_ids.len()).expect("invariant: test scan length fits u16");
    bytes.extend_from_slice(&[0xff, 0xda]);
    bytes.extend_from_slice(&scan_length.to_be_bytes());
    bytes.push(u8::try_from(component_ids.len()).expect("invariant: test component count fits u8"));
    for &id in component_ids {
        bytes.extend_from_slice(&[id, 0]);
    }
    bytes.extend_from_slice(&[0, 63, 0]);
    let used_bits = component_ids.len() * 2;
    bytes.push(u8::MAX >> used_bits);
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

#[test]
fn jpeg_baseline_grayscale_fragment_decodes_with_modality_lut() {
    let source = [32u8, 32, 32, 32];
    let jpeg = encode_grayscale_jpeg(2, 2, &source);

    let decoded = decode_jpeg_fragment(&jpeg, layout(2, 2, 2.0, -10.0))
        .expect("infallible: validated precondition");

    assert_eq!(decoded, reference_grayscale(&jpeg, 2.0, -10.0));
}

#[test]
fn jpeg_dimension_mismatch_is_rejected() {
    let jpeg = encode_grayscale_jpeg(2, 2, &[0, 64, 128, 255]);

    let err = decode_jpeg_fragment(&jpeg, layout(1, 4, 1.0, 0.0)).unwrap_err();

    assert!(
        err.to_string().contains("dimensions"),
        "expected dimension validation error, got {err:#}"
    );
}

#[test]
fn jpeg_rejects_component_amplified_dimensions_before_scan_allocation() {
    let mut jpeg = encode_rgb_jpeg(1, 1, &[120, 64, 32]);
    let sof = jpeg
        .windows(2)
        .position(|bytes| bytes == [0xFF, 0xC0])
        .expect("encoded JPEG must contain SOF0");
    jpeg[sof + 5..sof + 7].copy_from_slice(&16_384u16.to_be_bytes());
    jpeg[sof + 7..sof + 9].copy_from_slice(&16_384u16.to_be_bytes());

    let err = decode_jpeg_fragment(&jpeg, layout(1, 1, 1.0, 0.0))
        .expect_err("three-component frame over the pixel cap must fail");
    let source = err
        .downcast_ref::<consus_raster::DecodeError>()
        .expect("decode error must preserve the provider cause");
    assert_eq!(source.kind(), consus_raster::DecodeErrorKind::TooLarge);
}

#[test]
fn jpeg_rgb24_fragment_decodes_interleaved_samples() {
    let source = [120u8, 64, 32, 120, 64, 32];
    let jpeg = encode_rgb_jpeg(2, 1, &source);
    let layout = PixelLayout {
        rows: 1,
        cols: 2,
        samples_per_pixel: 3,
        bits_allocated: 8,
        bits_stored: 8,
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    };

    let decoded = decode_jpeg_fragment(&jpeg, layout).expect("infallible: validated precondition");

    let reference = reference_rgb(&jpeg);
    assert_eq!(decoded.len(), 6);
    assert_eq!(reference.len(), 6);

    // The repeated color produces a DC-only MCU, so both decoders reconstruct
    // identical YCbCr components without AC or interpolation differences. The
    // provider's BT.601 multipliers are 359/256, 88/256, 183/256, and 454/256;
    // the independent decoder uses 1.40200, 0.34414, 0.71414, and 1.77200 in
    // Q20. At the maximum centered chroma magnitude of 128, their respective
    // transform-term differences are below 0.044, 0.050 + 0.091, and 0.184
    // code values. Each is below half a code value, so the final integer
    // roundings can differ by at most one.
    for (actual, reference) in decoded.iter().zip(reference) {
        assert!(
            (*actual - reference).abs() <= 1.0,
            "constant RGB reconstruction differs by more than one code value: {actual} vs {reference}"
        );
    }
}

#[test]
fn jpeg_rgb24_rejects_grayscale_layout() {
    let jpeg = encode_rgb_jpeg(1, 1, &[120, 64, 32]);

    let err = decode_jpeg_fragment(&jpeg, layout(1, 1, 1.0, 0.0)).unwrap_err();

    assert!(
        err.to_string().contains("samples_per_pixel"),
        "expected samples-per-pixel validation error, got {err:#}"
    );
}

#[test]
fn jpeg_lossless_grayscale_fragment_decodes_exact_sample() {
    let jpeg = lossless_single_pixel_jpeg_8bit_gray_128();

    let decoded = decode_jpeg_fragment(&jpeg, layout(1, 1, 1.5, -2.0))
        .expect("infallible: validated precondition");

    assert_eq!(decoded, vec![190.0]);
}

#[test]
fn jpeg_lossless_signed_l8_fragment_decodes_exact_sample() {
    let jpeg = lossless_single_pixel_jpeg_8bit_gray_128();
    let layout = layout_with_precision(1, 1, 16, 8, PixelSignedness::Signed, 2.0, 5.0);

    let decoded = decode_jpeg_fragment(&jpeg, layout).expect("infallible: validated precondition");

    assert_eq!(decoded, vec![-251.0]);
}

#[test]
fn jpeg_lossless_low_precision_uses_codestream_sign_bit() {
    let jpeg = lossless_single_pixel_midpoint(7);
    let unsigned = decode_jpeg_fragment(
        &jpeg,
        layout_with_precision(1, 1, 16, 7, PixelSignedness::Unsigned, 2.0, -4.0),
    )
    .expect("valid unsigned seven-bit lossless sample");
    let signed = decode_jpeg_fragment(
        &jpeg,
        layout_with_precision(1, 1, 16, 7, PixelSignedness::Signed, 2.0, -4.0),
    )
    .expect("valid signed seven-bit lossless sample");

    assert_eq!(unsigned, vec![124.0]);
    assert_eq!(signed, vec![-132.0]);
}

#[test]
fn jpeg_lossless_two_bit_samples_use_sixteen_bit_dicom_container() {
    let jpeg = lossless_single_pixel_midpoint(2);
    let unsigned = decode_jpeg_fragment(
        &jpeg,
        layout_with_precision(1, 1, 16, 2, PixelSignedness::Unsigned, 3.0, 1.0),
    )
    .expect("valid unsigned two-bit lossless sample in a 16-bit container");
    let signed = decode_jpeg_fragment(
        &jpeg,
        layout_with_precision(1, 1, 16, 2, PixelSignedness::Signed, 3.0, 1.0),
    )
    .expect("valid signed two-bit lossless sample in a 16-bit container");

    assert_eq!(unsigned, vec![7.0]);
    assert_eq!(signed, vec![-5.0]);
}

#[test]
fn jpeg_lossless_wide_signed_sample_uses_codestream_sign_bit() {
    let jpeg = lossless_single_pixel_midpoint(12);
    let decoded = decode_jpeg_fragment(
        &jpeg,
        layout_with_precision(1, 1, 16, 12, PixelSignedness::Signed, 1.5, 2.0),
    )
    .expect("valid signed twelve-bit lossless sample");

    assert_eq!(decoded, vec![-3070.0]);
}

#[test]
fn jpeg_extended_twelve_bit_grayscale_preserves_full_precision() {
    let jpeg = dct_twelve_midpoint(&[1]);
    let decoded = decode_jpeg_fragment(
        &jpeg,
        layout_with_precision(8, 8, 16, 12, PixelSignedness::Unsigned, 2.0, -4.0),
    )
    .expect("valid twelve-bit extended grayscale sample");

    assert_eq!(decoded.len(), 64);
    assert_eq!(decoded, vec![4092.0; 64]);
}

#[test]
fn jpeg_extended_twelve_bit_rejects_signed_pixel_representation() {
    let jpeg = dct_twelve_midpoint(&[1]);
    let error = decode_jpeg_fragment(
        &jpeg,
        layout_with_precision(8, 8, 16, 12, PixelSignedness::Signed, 1.0, 0.0),
    )
    .expect_err("lossy JPEG cannot represent signed samples");

    assert!(
        error.to_string().contains("does not support signed"),
        "expected signedness rejection, got {error:#}"
    );
}

#[test]
fn jpeg_extended_twelve_bit_rejects_mismatched_bits_stored() {
    let jpeg = dct_twelve_midpoint(&[1]);
    let error = decode_jpeg_fragment(
        &jpeg,
        layout_with_precision(8, 8, 16, 16, PixelSignedness::Unsigned, 1.0, 0.0),
    )
    .expect_err("JPEG frame precision must match DICOM BitsStored");

    assert!(
        error
            .to_string()
            .contains("does not match DICOM BitsStored=16"),
        "expected BitsStored mismatch, got {error:#}"
    );
}

#[test]
fn jpeg_extended_twelve_bit_rgb_preserves_interleaved_samples() {
    let jpeg = dct_twelve_midpoint(b"RGB");
    let layout = PixelLayout {
        rows: 8,
        cols: 8,
        samples_per_pixel: 3,
        bits_allocated: 16,
        bits_stored: 12,
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    };
    let decoded =
        decode_jpeg_fragment(&jpeg, layout).expect("valid twelve-bit direct RGB extended sample");

    assert_eq!(decoded.len(), 8 * 8 * 3);
    assert_eq!(decoded, vec![2048.0; 8 * 8 * 3]);
}

#[test]
fn jpeg_provider_wide_output_uses_native_endian_contract() {
    let jpeg = lossless_single_pixel_jpeg_16bit_gray_0x1234();

    let decoded = jpeg::decode(
        &jpeg,
        DecodeLimits {
            max_encoded_bytes: jpeg.len(),
            max_dimension: 1,
            max_pixels: 1,
            max_working_bytes: jpeg::working_storage_bound(1, 1)
                .expect("invariant: one-pixel storage bound fits in usize"),
        },
    )
    .expect("infallible: validated precondition");

    assert_eq!(decoded.format(), PixelFormat::GrayWide);
    assert_eq!(decoded.pixels(), 0x1234u16.to_ne_bytes());
}

#[test]
fn jpeg_truncation_is_rejected() {
    let mut jpeg = lossless_single_pixel_jpeg_8bit_gray_128();
    jpeg.truncate(jpeg.len() - 2);

    let error = decode_jpeg_fragment(&jpeg, layout(1, 1, 1.0, 0.0)).unwrap_err();

    let source = error
        .downcast_ref::<consus_raster::DecodeError>()
        .expect("decode error must preserve the provider cause");
    assert_eq!(source.kind(), consus_raster::DecodeErrorKind::Malformed);
}

#[test]
fn jpeg_lossless_l16_fragment_decodes_exact_unsigned_sample() {
    let jpeg = lossless_single_pixel_jpeg_16bit_gray_0x1234();
    let layout = layout_with_precision(1, 1, 16, 16, PixelSignedness::Unsigned, 2.0, -4.0);

    let decoded = decode_jpeg_fragment(&jpeg, layout).expect("infallible: validated precondition");

    assert_eq!(decoded, vec![9316.0]);
}

#[test]
fn jpeg_lossless_l16_accepts_dicom_even_length_padding() {
    let mut jpeg = lossless_single_pixel_jpeg_16bit_gray_0x1234();
    assert!(!jpeg.len().is_multiple_of(2));
    jpeg.push(0);
    let layout = layout_with_precision(1, 1, 16, 16, PixelSignedness::Unsigned, 1.0, 0.0);

    let decoded = decode_jpeg_fragment(&jpeg, layout).expect("DICOM zero padding is permitted");

    assert_eq!(decoded, vec![4660.0]);
}
