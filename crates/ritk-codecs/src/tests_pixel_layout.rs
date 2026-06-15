use super::*;

#[test]
fn native_signed_16_decode_applies_linear_modality_lut() {
    let bytes = [-2i16, 0, 10]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let out = decode_native_pixel_bytes_checked(
        &bytes,
        PixelLayout {
            rows: 1,
            cols: 3,
            samples_per_pixel: 1,
            bits_allocated: 16,
            pixel_representation: PixelSignedness::Signed,
            rescale_slope: 2.0,
            rescale_intercept: 5.0,
        },
    )
    .unwrap();
    assert_eq!(out, vec![1.0, 5.0, 25.0]);
}

#[test]
fn native_signed_8_decode_applies_linear_modality_lut() {
    let bytes = [-2i8, 0, 10].iter().map(|v| *v as u8).collect::<Vec<_>>();

    let out = decode_native_pixel_bytes_checked(
        &bytes,
        PixelLayout {
            rows: 1,
            cols: 3,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: PixelSignedness::Signed,
            rescale_slope: 2.0,
            rescale_intercept: 5.0,
        },
    )
    .unwrap();

    assert_eq!(out, vec![1.0, 5.0, 25.0]);
}

#[test]
fn checked_native_decode_rejects_trailing_bytes() {
    let err = decode_native_pixel_bytes_checked(
        &[1, 0, 2],
        PixelLayout {
            rows: 1,
            cols: 1,
            samples_per_pixel: 1,
            bits_allocated: 16,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        },
    )
    .unwrap_err();

    assert!(
        err.to_string().contains("byte length"),
        "expected byte-length validation error, got {err:#}"
    );
}

#[test]
fn checked_native_decode_handles_unsigned_32bit_samples() {
    let bytes = [1u32, 65_535, 1_000_000]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();

    let out = decode_native_pixel_bytes_checked(
        &bytes,
        PixelLayout {
            rows: 1,
            cols: 3,
            samples_per_pixel: 1,
            bits_allocated: 32,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 0.5,
            rescale_intercept: -1.0,
        },
    )
    .unwrap();

    assert_eq!(out, vec![-0.5, 32766.5, 499999.0]);
}

#[test]
fn checked_native_decode_handles_signed_24bit_samples() {
    let bytes = [-2i32, 0, 10]
        .iter()
        .flat_map(|v| {
            let le = v.to_le_bytes();
            [le[0], le[1], le[2]]
        })
        .collect::<Vec<_>>();

    let out = decode_native_pixel_bytes_checked(
        &bytes,
        PixelLayout {
            rows: 1,
            cols: 3,
            samples_per_pixel: 1,
            bits_allocated: 24,
            pixel_representation: PixelSignedness::Signed,
            rescale_slope: 2.0,
            rescale_intercept: 5.0,
        },
    )
    .unwrap();

    assert_eq!(out, vec![1.0, 5.0, 25.0]);
}

#[test]
fn checked_native_decode_handles_signed_32bit_samples() {
    let bytes = [-2i32, 0, 10]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();

    let out = decode_native_pixel_bytes_checked(
        &bytes,
        PixelLayout {
            rows: 1,
            cols: 3,
            samples_per_pixel: 1,
            bits_allocated: 32,
            pixel_representation: PixelSignedness::Signed,
            rescale_slope: 2.0,
            rescale_intercept: 5.0,
        },
    )
    .unwrap();

    assert_eq!(out, vec![1.0, 5.0, 25.0]);
}

#[test]
fn checked_native_decode_rejects_invalid_pixel_representation_from_u16() {
    let err = PixelSignedness::try_from(2u16).unwrap_err();
    assert!(
        err.to_string().contains("pixel_representation"),
        "expected pixel representation conversion error, got {err:#}"
    );
}

#[test]
fn checked_native_decode_rejects_nonfinite_rescale_slope() {
    let err = decode_native_pixel_bytes_checked(
        &[1],
        PixelLayout {
            rows: 1,
            cols: 1,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: f32::NAN,
            rescale_intercept: 0.0,
        },
    )
    .unwrap_err();

    assert!(
        err.to_string().contains("rescale_slope"),
        "expected rescale_slope validation error, got {err:#}"
    );
}

#[test]
fn checked_native_decode_rejects_nonfinite_rescale_intercept() {
    let err = decode_native_pixel_bytes_checked(
        &[1],
        PixelLayout {
            rows: 1,
            cols: 1,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: PixelSignedness::Unsigned,
            rescale_slope: 1.0,
            rescale_intercept: f32::INFINITY,
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("rescale_intercept"),
        "expected rescale_intercept validation error, got {err:#}"
    );
}

/// GAP-R08g regression: signed i16 stored values with RescaleIntercept=-1024
/// must produce correct Hounsfield units.
///
/// DICOM PS3.3 C.7.6.3.1.4: output = stored_integer x slope + intercept.
/// For CT with PixelRepresentation=1, BitsAllocated=16, Slope=1,
/// Intercept=-1024: stored value -1024 -> HU = -1024*1 + (-1024) = -2048.
/// This was the root cause of GAP-R08g where RITK produced min=-1024
/// instead of the correct -2048.
#[test]
fn ct_signed_i16_rescale_intercept_minus_1024_produces_correct_hu() {
    // Stored values: -1024 (air), 0 (water), 1000 (bone)
    let stored: [i16; 3] = [-1024, 0, 1000];
    let bytes: Vec<u8> = stored.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = decode_native_pixel_bytes_checked(
        &bytes,
        PixelLayout {
            rows: 1,
            cols: 3,
            samples_per_pixel: 1,
            bits_allocated: 16,
            pixel_representation: PixelSignedness::Signed,
            rescale_slope: 1.0,
            rescale_intercept: -1024.0,
        },
    )
    .unwrap();
    // HU = stored * 1 + (-1024)
    assert_eq!(out[0], -2048.0, "air: -1024 * 1 + (-1024) = -2048 HU");
    assert_eq!(out[1], -1024.0, "water: 0 * 1 + (-1024) = -1024 HU");
    assert_eq!(out[2], -24.0, "bone: 1000 * 1 + (-1024) = -24 HU");
}

/// Verify that identity rescale (slope=1, intercept=0) passes stored values
/// through unchanged, which is the correct behavior when rescale has already
/// been applied upstream (e.g., by dicom-pixeldata in decode_via_dicom_rs).
#[test]
fn identity_rescale_preserves_signed_i16_stored_values() {
    let stored: [i16; 4] = [-1024, -512, 0, 3071];
    let bytes: Vec<u8> = stored.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = decode_native_pixel_bytes_checked(
        &bytes,
        PixelLayout {
            rows: 1,
            cols: 4,
            samples_per_pixel: 1,
            bits_allocated: 16,
            pixel_representation: PixelSignedness::Signed,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        },
    )
    .unwrap();
    assert_eq!(out, vec![-1024.0, -512.0, 0.0, 3071.0]);
}

#[test]
fn pixel_signedness_try_from_rejects_invalid_u16() {
    assert!(PixelSignedness::try_from(0u16).is_ok());
    assert!(PixelSignedness::try_from(1u16).is_ok());
    assert!(PixelSignedness::try_from(2u16).is_err());
    assert!(PixelSignedness::try_from(255u16).is_err());
}

#[test]
fn pixel_signedness_from_u16_round_trips() {
    assert_eq!(u16::from(PixelSignedness::Unsigned), 0);
    assert_eq!(u16::from(PixelSignedness::Signed), 1);
}

#[test]
fn pixel_signedness_default_is_unsigned() {
    assert_eq!(PixelSignedness::default(), PixelSignedness::Unsigned);
}
