use super::*;
use std::f64::consts::PI;

// ── ByteOrder::from_nrrd ──────────────────────────────────────────────

#[test]
fn from_nrrd_big_is_most_significant_byte_first() {
    assert_eq!(
        ByteOrder::from_nrrd("big"),
        ByteOrder::MostSignificantByteFirst
    );
}

#[test]
fn from_nrrd_little_is_least_significant_byte_first() {
    assert_eq!(
        ByteOrder::from_nrrd("little"),
        ByteOrder::LeastSignificantByteFirst
    );
}

#[test]
fn from_nrrd_is_case_insensitive() {
    assert_eq!(
        ByteOrder::from_nrrd("BIG"),
        ByteOrder::MostSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_nrrd("Little"),
        ByteOrder::LeastSignificantByteFirst
    );
}

#[test]
fn from_nrrd_trims_whitespace() {
    assert_eq!(
        ByteOrder::from_nrrd("  big  "),
        ByteOrder::MostSignificantByteFirst
    );
}

#[test]
fn from_nrrd_unknown_defaults_to_little_endian() {
    assert_eq!(
        ByteOrder::from_nrrd("msbfirst"),
        ByteOrder::LeastSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_nrrd("mostsignificantbytefirst"),
        ByteOrder::LeastSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_nrrd(""),
        ByteOrder::LeastSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_nrrd("middle"),
        ByteOrder::LeastSignificantByteFirst
    );
}

#[test]
fn from_nrrd_embedded_whitespace_defaults_to_little_endian() {
    assert_eq!(
        ByteOrder::from_nrrd("bi g"),
        ByteOrder::LeastSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_nrrd("lit tle"),
        ByteOrder::LeastSignificantByteFirst
    );
}

#[test]
fn from_nrrd_partial_match_defaults_to_little_endian() {
    assert_eq!(
        ByteOrder::from_nrrd("bigg"),
        ByteOrder::LeastSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_nrrd("littl"),
        ByteOrder::LeastSignificantByteFirst
    );
}

// ── ByteOrder::from_metaimage_msb ─────────────────────────────────────

#[test]
fn from_metaimage_msb_true_is_most_significant_byte_first() {
    assert_eq!(
        ByteOrder::from_metaimage_msb("True"),
        ByteOrder::MostSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_metaimage_msb("TRUE"),
        ByteOrder::MostSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_metaimage_msb("true"),
        ByteOrder::MostSignificantByteFirst
    );
}

#[test]
fn from_metaimage_msb_false_or_unknown_is_little_endian() {
    assert_eq!(
        ByteOrder::from_metaimage_msb("False"),
        ByteOrder::LeastSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_metaimage_msb("FALSE"),
        ByteOrder::LeastSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_metaimage_msb(""),
        ByteOrder::LeastSignificantByteFirst
    );
}

#[test]
fn from_metaimage_msb_mixed_case() {
    assert_eq!(
        ByteOrder::from_metaimage_msb("tRuE"),
        ByteOrder::MostSignificantByteFirst
    );
    assert_eq!(
        ByteOrder::from_metaimage_msb("fAlSe"),
        ByteOrder::LeastSignificantByteFirst
    );
}

#[test]
fn from_metaimage_msb_non_bool_string_defaults_to_little_endian() {
    for v in ["yes", "1", "on", "true ", " true", "True.", "big"] {
        assert_eq!(
            ByteOrder::from_metaimage_msb(v),
            ByteOrder::LeastSignificantByteFirst,
            "expected LSB for non-bool input {v:?}"
        );
    }
}

// ── parse_floats round-trip ──────────────────────────────────────────

#[test]
fn parse_floats_f64_round_trip() {
    let s = format!("1.0 2.5 {} 4.0", -PI);
    let v: Vec<f64> = parse_floats(&s, "test_field", 4).expect("infallible: validated precondition");
    assert_eq!(v, vec![1.0, 2.5, -PI, 4.0]);
}

#[test]
fn parse_floats_usize_round_trip() {
    let s = "10 20 30";
    let v: Vec<usize> = parse_floats(s, "sizes", 3).expect("infallible: validated precondition");
    assert_eq!(v, vec![10, 20, 30]);
}

#[test]
fn parse_floats_length_mismatch_returns_error() {
    let s = "1.0 2.0";
    let err = parse_floats::<f64>(s, "field", 3).unwrap_err();
    assert!(
        err.to_string().contains("must have 3 components, got 2"),
        "expected length-mismatch error, got: {err:#}"
    );
}

#[test]
fn parse_floats_bad_token_returns_error() {
    let s = "1.0 not_a_number 3.0";
    let err = parse_floats::<f64>(s, "field", 3).unwrap_err();
    assert!(
        err.to_string().contains("not_a_number"),
        "expected error to mention the offending token, got: {err:#}"
    );
}

// ── require_bytes ────────────────────────────────────────────────────

#[test]
fn require_bytes_ok_when_buffer_is_large_enough() {
    assert!(require_bytes(100, 10, 4, "f32").is_ok());
}

#[test]
fn require_bytes_err_when_buffer_is_too_short() {
    let err = require_bytes(10, 10, 4, "f32").unwrap_err();
    assert!(
        err.to_string().contains("need 40 bytes, got 10"),
        "expected explicit byte-count error, got: {err:#}"
    );
}

// ── decode_bytes_to_f32 round-trip (one combo per branch) ─────────────

#[test]
fn decode_u8_le_round_trips_to_f32() {
    let bytes = vec![0u8, 1, 127, 255];
    let out = decode_bytes_to_f32(
        &bytes,
        1,
        false,
        false,
        ByteOrder::LeastSignificantByteFirst,
        4,
        "uint8",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![0.0, 1.0, 127.0, 255.0]);
}

#[test]
fn decode_i16_be_round_trips_to_f32() {
    let raw: Vec<u8> = [1_i16, -1, 32767, -32768]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        2,
        true,
        false,
        ByteOrder::MostSignificantByteFirst,
        4,
        "int16",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![1.0, -1.0, 32767.0, -32768.0]);
}

#[test]
fn decode_f32_le_round_trips() {
    let raw: Vec<u8> = [0.0_f32, 1.0, -1.0, std::f32::consts::PI]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        4,
        false,
        true,
        ByteOrder::LeastSignificantByteFirst,
        4,
        "float",
    )
    .expect("infallible: validated precondition");
    for (i, expected) in [0.0_f32, 1.0, -1.0, std::f32::consts::PI]
        .iter()
        .enumerate()
    {
        assert_eq!(out[i].to_bits(), expected.to_bits());
    }
}

#[test]
fn decode_f64_be_narrows_to_f32() {
    let raw: Vec<u8> = [1.5_f64, -2.5]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        8,
        false,
        true,
        ByteOrder::MostSignificantByteFirst,
        2,
        "double",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out.len(), 2);
    assert!((out[0] - 1.5_f32).abs() < 1e-6);
    assert!((out[1] - -2.5_f32).abs() < 1e-6);
}

#[test]
fn decode_unsupported_element_size_returns_error() {
    let bytes = vec![0u8; 12];
    let err = decode_bytes_to_f32(
        &bytes,
        3,
        false,
        false,
        ByteOrder::LeastSignificantByteFirst,
        4,
        "weird",
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("Unsupported integer element size 3"),
        "expected explicit error, got: {err:#}"
    );
}
