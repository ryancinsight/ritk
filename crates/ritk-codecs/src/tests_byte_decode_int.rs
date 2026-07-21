use super::*;

// ── decode_bytes_to_f32 full matrix (8 signed + 8 unsigned + 2 float) ──

// ── 1-byte signed: i8 ─────────────────────────────────────────────────

#[test]
fn decode_i8_le_round_trips_to_f32() {
    let bytes = vec![0u8, 1, 127, 128, 255];
    let out = decode_bytes_to_f32(
        &bytes,
        1,
        true,
        false,
        ByteOrder::LeastSignificantByteFirst,
        5,
        "int8",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![0.0, 1.0, 127.0, -128.0, -1.0]);
}

#[test]
fn decode_i8_be_round_trips_to_f32() {
    let bytes = vec![0u8, 1, 127, 128, 255];
    let out = decode_bytes_to_f32(
        &bytes,
        1,
        true,
        false,
        ByteOrder::MostSignificantByteFirst,
        5,
        "int8",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![0.0, 1.0, 127.0, -128.0, -1.0]);
}

// ── 2-byte signed: i16 ────────────────────────────────────────────────

#[test]
fn decode_i16_le_round_trips_to_f32() {
    let raw: Vec<u8> = [1_i16, -1, 32767, -32768]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        2,
        true,
        false,
        ByteOrder::LeastSignificantByteFirst,
        4,
        "int16",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![1.0, -1.0, 32767.0, -32768.0]);
}

// ── 4-byte signed: i32 ────────────────────────────────────────────────

#[test]
fn decode_i32_le_round_trips_to_f32() {
    let raw: Vec<u8> = [1_i32, -1, i32::MAX, i32::MIN]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        4,
        true,
        false,
        ByteOrder::LeastSignificantByteFirst,
        4,
        "int32",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![1.0, -1.0, i32::MAX as f32, i32::MIN as f32]);
}

#[test]
fn decode_i32_be_round_trips_to_f32() {
    let raw: Vec<u8> = [1_i32, -1, i32::MAX, i32::MIN]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        4,
        true,
        false,
        ByteOrder::MostSignificantByteFirst,
        4,
        "int32",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![1.0, -1.0, i32::MAX as f32, i32::MIN as f32]);
}

// ── 8-byte signed: i64 ────────────────────────────────────────────────

#[test]
fn decode_i64_le_round_trips_to_f32() {
    let raw: Vec<u8> = [1_i64, -1, 123_456_789_012_345_i64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        8,
        true,
        false,
        ByteOrder::LeastSignificantByteFirst,
        3,
        "int64",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out[0], 1.0);
    assert_eq!(out[1], -1.0);
    assert!((out[2] - 123_456_789_012_345.0_f32).abs() < 1.0);
}

#[test]
fn decode_i64_be_round_trips_to_f32() {
    let raw: Vec<u8> = [1_i64, -1, 123_456_789_012_345_i64]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        8,
        true,
        false,
        ByteOrder::MostSignificantByteFirst,
        3,
        "int64",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out[0], 1.0);
    assert_eq!(out[1], -1.0);
    assert!((out[2] - 123_456_789_012_345.0_f32).abs() < 1.0);
}

// ── 1-byte unsigned: u8 ──────────────────────────────────────────────

#[test]
fn decode_u8_be_round_trips_to_f32() {
    let bytes = vec![0u8, 1, 127, 255];
    let out = decode_bytes_to_f32(
        &bytes,
        1,
        false,
        false,
        ByteOrder::MostSignificantByteFirst,
        4,
        "uint8",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![0.0, 1.0, 127.0, 255.0]);
}

// ── 2-byte unsigned: u16 ──────────────────────────────────────────────

#[test]
fn decode_u16_le_round_trips_to_f32() {
    let raw: Vec<u8> = [1_u16, 1000, 65535]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        2,
        false,
        false,
        ByteOrder::LeastSignificantByteFirst,
        3,
        "uint16",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![1.0, 1000.0, 65535.0]);
}

#[test]
fn decode_u16_be_round_trips_to_f32() {
    let raw: Vec<u8> = [1_u16, 1000, 65535]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        2,
        false,
        false,
        ByteOrder::MostSignificantByteFirst,
        3,
        "uint16",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![1.0, 1000.0, 65535.0]);
}

// ── 4-byte unsigned: u32 ──────────────────────────────────────────────

#[test]
fn decode_u32_le_round_trips_to_f32() {
    let raw: Vec<u8> = [1_u32, 1_000_000, u32::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        4,
        false,
        false,
        ByteOrder::LeastSignificantByteFirst,
        3,
        "uint32",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![1.0, 1_000_000.0, u32::MAX as f32]);
}

#[test]
fn decode_u32_be_round_trips_to_f32() {
    let raw: Vec<u8> = [1_u32, 1_000_000, u32::MAX]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        4,
        false,
        false,
        ByteOrder::MostSignificantByteFirst,
        3,
        "uint32",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![1.0, 1_000_000.0, u32::MAX as f32]);
}

// ── 8-byte unsigned: u64 ──────────────────────────────────────────────

#[test]
fn decode_u64_le_round_trips_to_f32() {
    let raw: Vec<u8> = [1_u64, 123_456_789_012_345_u64, u64::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        8,
        false,
        false,
        ByteOrder::LeastSignificantByteFirst,
        3,
        "uint64",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out[0], 1.0);
    assert!((out[1] - 123_456_789_012_345.0_f32).abs() < 1.0);
    assert!((out[2] - u64::MAX as f32).abs() / (u64::MAX as f32) < 1e-5);
}

#[test]
fn decode_u64_be_round_trips_to_f32() {
    let raw: Vec<u8> = [1_u64, 123_456_789_012_345_u64, u64::MAX]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let out = decode_bytes_to_f32(
        &raw,
        8,
        false,
        false,
        ByteOrder::MostSignificantByteFirst,
        3,
        "uint64",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out[0], 1.0);
    assert!((out[1] - 123_456_789_012_345.0_f32).abs() < 1.0);
    assert!((out[2] - u64::MAX as f32).abs() / (u64::MAX as f32) < 1e-5);
}

// ── Edge cases & error paths ──────────────────────────────────────────

#[test]
fn decode_count_zero_returns_empty() {
    let bytes = vec![0u8; 4];
    let out = decode_bytes_to_f32(
        &bytes,
        4,
        false,
        true,
        ByteOrder::LeastSignificantByteFirst,
        0,
        "float",
    )
    .expect("infallible: validated precondition");
    assert!(out.is_empty());
}

#[test]
fn decode_type_name_passes_through_to_error() {
    let bytes: Vec<u8> = vec![0u8; 4];
    let out = decode_bytes_to_f32(
        &bytes,
        4,
        false,
        true,
        ByteOrder::LeastSignificantByteFirst,
        1,
        "MET_CARDIAC_3D",
    )
    .expect("infallible: validated precondition");
    assert_eq!(out, vec![0.0]);
    let err = decode_bytes_to_f32(
        &bytes,
        4,
        false,
        true,
        ByteOrder::LeastSignificantByteFirst,
        10,
        "MET_CARDIAC_3D",
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("MET_CARDIAC_3D"),
        "expected error to mention the type name, got: {err:#}"
    );
}

#[test]
fn decode_unsupported_float_element_size_returns_error() {
    let bytes = vec![0u8; 24];
    let err = decode_bytes_to_f32(
        &bytes,
        12,
        false,
        true,
        ByteOrder::LeastSignificantByteFirst,
        2,
        "long_double",
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("Unsupported float element size 12"),
        "expected explicit error, got: {err:#}"
    );
}
