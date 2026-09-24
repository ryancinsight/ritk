//! `Data` payloads: each encoding, byte order, and indexing order, and the
//! ways each can disagree with its declared shape.

use std::io::Write as _;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use flate2::Compression;
use flate2::write::ZlibEncoder;

use super::{error_of, one_array, read};
use crate::{ArrayData, GiftiError};

const VALUES: [f32; 4] = [1.5, -2.25, 0.0, 3.0e-3];

fn float_attributes(encoding: &str, endian: &str, count: usize) -> String {
    format!(
        r#"Intent="NIFTI_INTENT_SHAPE" DataType="NIFTI_TYPE_FLOAT32" ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" Dim0="{count}" Encoding="{encoding}" Endian="{endian}""#
    )
}

fn little_endian(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn zlib(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes).expect("in-memory write");
    encoder.finish().expect("in-memory write")
}

fn values_of(document: &str) -> ArrayData {
    read(document).expect("valid document").arrays()[0]
        .data()
        .clone()
}

#[test]
fn base64_decodes_in_either_byte_order() {
    let little = STANDARD.encode(little_endian(&VALUES));
    let big: Vec<u8> = VALUES
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let big = STANDARD.encode(big);
    for (endian, payload) in [("LittleEndian", little), ("BigEndian", big)] {
        let document = one_array(&float_attributes("Base64Binary", endian, 4), &payload);
        assert_eq!(
            values_of(&document),
            ArrayData::Float32(VALUES.into()),
            "{endian}"
        );
    }
}

/// Section 5.0: the payload is a zlib stream, and base64 text may be wrapped.
#[test]
fn zlib_base64_decodes_and_tolerates_wrapped_text() {
    let encoded = STANDARD.encode(zlib(&little_endian(&VALUES)));
    let (head, tail) = encoded.split_at(encoded.len() / 2);
    let document = one_array(
        &float_attributes("GZipBase64Binary", "LittleEndian", 4),
        &format!("\n  {head}\n  {tail}\n"),
    );
    assert_eq!(values_of(&document), ArrayData::Float32(VALUES.into()));
}

#[test]
fn int32_binary_decodes_negative_values() {
    let values = [-7_i32, 0, i32::MAX, i32::MIN];
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let document = one_array(
        r#"Intent="NIFTI_INTENT_LABEL" DataType="NIFTI_TYPE_INT32" ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" Dim0="4" Encoding="Base64Binary" Endian="BigEndian""#,
        &STANDARD.encode(bytes),
    );
    assert_eq!(values_of(&document), ArrayData::Int32(values.into()));
}

/// Section 2.3.4.1: column-major `1 4 2 5 3 6` is the 2×3 matrix whose rows
/// are `1 2 3` and `4 5 6`.
#[test]
fn a_column_major_array_transposes_to_row_major() {
    let document = one_array(
        r#"Intent="NIFTI_INTENT_NONE" DataType="NIFTI_TYPE_INT32" ArrayIndexingOrder="ColumnMajorOrder" Dimensionality="2" Dim0="2" Dim1="3" Encoding="ASCII""#,
        "1 4 2 5 3 6",
    );
    let image = read(&document).expect("valid document");
    assert_eq!(
        image.arrays()[0].row_major(),
        ArrayData::Int32(vec![1, 2, 3, 4, 5, 6].into())
    );
}

/// A three-dimensional column-major array, checked element by element
/// against the index formula rather than a hand-transposed list.
#[test]
fn a_rank_three_column_major_array_transposes_by_the_index_formula() {
    let dims = [2_usize, 3, 4];
    // Store value 100·i + 10·j + k at column-major offset i + 2j + 6k.
    let mut stored = [0_i32; 24];
    for (i, j, k) in (0..2).flat_map(|i| (0..3).flat_map(move |j| (0..4).map(move |k| (i, j, k)))) {
        stored[i + 2 * j + 6 * k] = i32::try_from(100 * i + 10 * j + k).expect("small");
    }
    let text: Vec<String> = stored.iter().map(ToString::to_string).collect();
    let document = one_array(
        &format!(
            r#"Intent="NIFTI_INTENT_NONE" DataType="NIFTI_TYPE_INT32" ArrayIndexingOrder="ColumnMajorOrder" Dimensionality="3" Dim0="{}" Dim1="{}" Dim2="{}" Encoding="ASCII""#,
            dims[0], dims[1], dims[2]
        ),
        &text.join(" "),
    );
    let ArrayData::Int32(row_major) = read(&document).expect("valid").arrays()[0].row_major()
    else {
        panic!("INT32 data");
    };
    let expected: Vec<i32> = (0..2)
        .flat_map(|i| (0..3).flat_map(move |j| (0..4).map(move |k| 100 * i + 10 * j + k)))
        .collect();
    assert_eq!(row_major.to_vec(), expected);
}

fn data_error(document: &str) -> String {
    match error_of(document) {
        GiftiError::Data { array: 0, reason } => reason,
        other => panic!("expected a data error, got {other}"),
    }
}

#[test]
fn an_ascii_count_other_than_the_shape_is_rejected() {
    let attributes = float_attributes("ASCII", "LittleEndian", 3);
    assert!(data_error(&one_array(&attributes, "1 2")).contains("2 values"));
    assert!(data_error(&one_array(&attributes, "1 2 3 4")).contains("more than"));
    assert!(data_error(&one_array(&attributes, "1 x 3")).contains("\"x\""));
}

#[test]
fn a_binary_length_other_than_the_shape_is_rejected() {
    let payload = STANDARD.encode(little_endian(&VALUES));
    let reason = data_error(&one_array(
        &float_attributes("Base64Binary", "LittleEndian", 5),
        &payload,
    ));
    assert!(
        reason.contains("16 bytes decoded, shape needs 20"),
        "{reason}"
    );
}

/// A stream that inflates past its declared shape is caught after one extra
/// byte rather than inflated to its full size.
#[test]
fn a_zlib_stream_longer_than_the_shape_is_rejected() {
    let payload = STANDARD.encode(zlib(&vec![0_u8; 1 << 20]));
    let reason = data_error(&one_array(
        &float_attributes("GZipBase64Binary", "LittleEndian", 4),
        &payload,
    ));
    assert!(
        reason.contains("17 bytes decoded, shape needs 16"),
        "{reason}"
    );
}

#[test]
fn malformed_base64_and_zlib_are_rejected() {
    let bad_base64 = one_array(&float_attributes("Base64Binary", "LittleEndian", 4), "@@@@");
    assert!(data_error(&bad_base64).contains("base64"));
    let not_zlib = STANDARD.encode(little_endian(&VALUES));
    let bad_zlib = one_array(
        &float_attributes("GZipBase64Binary", "LittleEndian", 4),
        &not_zlib,
    );
    assert!(data_error(&bad_zlib).contains("zlib"));
}

#[test]
fn binary_data_without_a_byte_order_is_rejected() {
    let document = one_array(
        r#"Intent="NIFTI_INTENT_SHAPE" DataType="NIFTI_TYPE_FLOAT32" ArrayIndexingOrder="RowMajorOrder" Dimensionality="1" Dim0="4" Encoding="Base64Binary""#,
        &STANDARD.encode(little_endian(&VALUES)),
    );
    assert!(data_error(&document).contains("Endian"));
}
