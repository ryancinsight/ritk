use super::*;
use std::num::NonZeroUsize;

#[test]
fn le_round_trips() {
    let val: f32 = std::f32::consts::PI;
    let raw = val.to_le_bytes().to_vec();
    let dtype = Datatype::Float {
        bits: NonZeroUsize::new(32).expect("infallible: validated precondition"),
        byte_order: consus_core::ByteOrder::LittleEndian,
    };
    let result = decode_raw_bytes(&raw, &dtype).expect("infallible: validated precondition");
    assert_eq!(result.len(), 1);
    assert!((result[0] - std::f32::consts::PI).abs() < 1e-5);
}

#[test]
fn be_round_trips() {
    let val: f32 = 2.71;
    let raw = val.to_be_bytes().to_vec();
    let dtype = Datatype::Float {
        bits: NonZeroUsize::new(32).expect("infallible: validated precondition"),
        byte_order: consus_core::ByteOrder::BigEndian,
    };
    let result = decode_raw_bytes(&raw, &dtype).expect("infallible: validated precondition");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 2.71).abs() < 1e-5);
}

#[test]
fn double_precision_le_narrows_to_single() {
    let val: f64 = 1.23456789;
    let raw = val.to_le_bytes().to_vec();
    let dtype = Datatype::Float {
        bits: NonZeroUsize::new(64).expect("infallible: validated precondition"),
        byte_order: consus_core::ByteOrder::LittleEndian,
    };
    let result = decode_raw_bytes(&raw, &dtype).expect("infallible: validated precondition");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.234_567_9_f32).abs() < 1e-5);
}

#[test]
fn u8_identity_is_preserved() {
    let raw = vec![0u8, 128, 255];
    let dtype = Datatype::Integer {
        bits: NonZeroUsize::new(8).expect("infallible: validated precondition"),
        byte_order: consus_core::ByteOrder::LittleEndian,
        signed: false,
    };
    let result = decode_raw_bytes(&raw, &dtype).expect("infallible: validated precondition");
    assert_eq!(result, vec![0.0, 128.0, 255.0]);
}

#[test]
fn i8_signed_range_is_preserved() {
    let raw = vec![0u8, 127, 0x80]; // 0, 127, -128
    let dtype = Datatype::Integer {
        bits: NonZeroUsize::new(8).expect("infallible: validated precondition"),
        byte_order: consus_core::ByteOrder::LittleEndian,
        signed: true,
    };
    let result = decode_raw_bytes(&raw, &dtype).expect("infallible: validated precondition");
    assert_eq!(result, vec![0.0, 127.0, -128.0]);
}

#[test]
fn signed_short_le_boundary_values() {
    let raw: Vec<u8> = vec![
        0x00, 0x00, // 0
        0xFF, 0x7F, // 32767
        0x00, 0x80, // -32768
    ];
    let dtype = Datatype::Integer {
        bits: NonZeroUsize::new(16).expect("infallible: validated precondition"),
        byte_order: consus_core::ByteOrder::LittleEndian,
        signed: true,
    };
    let result = decode_raw_bytes(&raw, &dtype).expect("infallible: validated precondition");
    assert_eq!(result, vec![0.0, 32767.0, -32768.0]);
}

#[test]
fn convert_boolean_maps_nonzero_to_one() {
    let raw = vec![0u8, 1, 0, 255];
    let result =
        decode_raw_bytes(&raw, &Datatype::Boolean).expect("infallible: validated precondition");
    assert_eq!(result, vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn convert_unsupported_type_errors() {
    let dtype = Datatype::VariableString {
        encoding: consus_core::StringEncoding::Utf8,
    };
    assert!(decode_raw_bytes(&[], &dtype).is_err());
}

#[test]
fn multiple_le_float_values() {
    let values = [0.0f32, -1.0, 1.0, f32::MAX, f32::MIN_POSITIVE];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let dtype = Datatype::Float {
        bits: NonZeroUsize::new(32).expect("infallible: validated precondition"),
        byte_order: consus_core::ByteOrder::LittleEndian,
    };
    let result = decode_raw_bytes(&raw, &dtype).expect("infallible: validated precondition");
    assert_eq!(result.len(), 5);
    for (i, &expected) in values.iter().enumerate() {
        assert_eq!(
            result[i].to_bits(),
            expected.to_bits(),
            "mismatch at index {}",
            i
        );
    }
}
