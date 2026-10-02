//! Tests for the `.mif` header parser.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use ritk_core::rejection::assert_rejects;
use std::io::Cursor;

use super::*;

#[test]
fn parse_minimal_single_volume_header() {
    let input = "\
mrtrix image: version 3.0
dim: 128 128 60
vox: 1.0 1.0 2.0
layout: +0,+1,+2
datatype: Float32LE
file: . 256
END
";
    let mut reader = Cursor::new(input.as_bytes());
    let header = parse_mif_header(&mut reader).unwrap();

    assert_eq!(header.entries.get("dim").unwrap().as_line(), "128 128 60");
    assert_eq!(
        header.entries.get("datatype").unwrap().as_line(),
        "Float32LE"
    );
    assert_eq!(header.entries.get("layout").unwrap().as_line(), "+0,+1,+2");
    assert_eq!(header.entries.get("vox").unwrap().as_line(), "1.0 1.0 2.0");
    assert!(header.entries.contains_key("mrtrix image"));
}

#[test]
fn parse_transform_block() {
    let input = "\
mrtrix image: version 3.0
dim: 128 128 60
datatype: Float32LE
transform: 1.0 0.0 0.0 -64.0
 0.0 1.0 0.0 -64.0
 0.0 0.0 1.0 -30.0
 0.0 0.0 0.0 1.0
file: . 256
END
";
    let mut reader = Cursor::new(input.as_bytes());
    let header = parse_mif_header(&mut reader).unwrap();

    let transform = header.entries.get("transform").unwrap();
    assert!(transform.is_block(), "transform should be a block");
    let rows = transform.as_block();
    assert_eq!(rows.len(), 4, "transform should have 4 rows");
    assert!(rows[0].contains("1.0 0.0 0.0 -64.0"));
}

#[test]
fn parse_multiframe_header() {
    let input = "\
mrtrix image: version 3.0
dim: 128 128 60 33
vox: 1.7 1.7 2.2
layout: +0,+1,+2,+3
datatype: Float32LE
DW_scheme: 2,4
0,0,0,0
1,0,0,1000
file: . 256
END
";
    let mut reader = Cursor::new(input.as_bytes());
    let header = parse_mif_header(&mut reader).unwrap();

    assert_eq!(
        header.entries.get("dim").unwrap().as_line(),
        "128 128 60 33"
    );
    assert_eq!(
        header.entries.get("layout").unwrap().as_line(),
        "+0,+1,+2,+3"
    );
    assert!(header.entries.contains_key("dw_scheme"));
}

#[test]
fn backslash_continuation_lines() {
    let input = "\
mrtrix image: version 3.0
dim: 128 128 60 33
datatype: Float32LE
comments: this is a very long comment line that is \
 continued across multiple\
 physical lines in the header
file: . 256
END
";
    let mut reader = Cursor::new(input.as_bytes());
    let header = parse_mif_header(&mut reader).unwrap();

    let comments = header.entries.get("comments").unwrap().as_line();
    assert!(comments.contains("very long comment"));
    assert!(comments.contains("continued across multiple"));
    assert!(comments.contains("physical lines"));
}

#[test]
fn eof_before_end_is_error() {
    let input = "mrtrix image: version 3.0\ndim: 10 10 10\n";
    let mut reader = Cursor::new(input.as_bytes());
    let result = parse_mif_header(&mut reader);
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("EOF before END marker"));
}

#[test]
fn parse_dim() {
    assert_eq!(
        super::parse_dim("128 128 60", 3).unwrap(),
        vec![128, 128, 60]
    );
    assert_eq!(
        super::parse_dim("128 128 60 33", 3).unwrap(),
        vec![128, 128, 60, 33]
    );
}

#[test]
fn parse_dim_too_few_components() {
    let result = super::parse_dim("128 128", 3);
    assert_rejects(
        result,
        "dim: expected at least 3 values, got 2 ([128, 128])",
    );
}

#[test]
fn parse_vox_reads_the_spatial_sizes() {
    assert_eq!(parse_vox("1.0 1.5 2.0").unwrap(), vec![1.0, 1.5, 2.0]);
    assert_eq!(parse_vox("1 1 1 2.5").unwrap(), vec![1.0, 1.0, 1.0, 2.5]);
}

#[test]
fn parse_vox_rejects_fewer_than_three_sizes() {
    assert_rejects(
        parse_vox("1.0 2.0"),
        ".mif 'vox' expected at least 3 spatial sizes, got 2",
    );
}

#[test]
fn parse_layout_contiguous() {
    assert_eq!(
        super::parse_layout("+0,+1,+2,+3").unwrap(),
        vec![0, 1, 2, 3]
    );
}

/// The MRtrix spelling of every datatype this crate stores, with the type and
/// byte order it names.
const DATATYPE_SPELLINGS: [(&str, SampleType, ByteOrder); 18] = [
    ("Int8", SampleType::I8, ByteOrder::LittleEndian),
    ("UInt8", SampleType::U8, ByteOrder::LittleEndian),
    ("Int16LE", SampleType::I16, ByteOrder::LittleEndian),
    ("Int16BE", SampleType::I16, ByteOrder::BigEndian),
    ("UInt16LE", SampleType::U16, ByteOrder::LittleEndian),
    ("UInt16BE", SampleType::U16, ByteOrder::BigEndian),
    ("Int32LE", SampleType::I32, ByteOrder::LittleEndian),
    ("Int32BE", SampleType::I32, ByteOrder::BigEndian),
    ("UInt32LE", SampleType::U32, ByteOrder::LittleEndian),
    ("UInt32BE", SampleType::U32, ByteOrder::BigEndian),
    ("Int64LE", SampleType::I64, ByteOrder::LittleEndian),
    ("Int64BE", SampleType::I64, ByteOrder::BigEndian),
    ("UInt64LE", SampleType::U64, ByteOrder::LittleEndian),
    ("UInt64BE", SampleType::U64, ByteOrder::BigEndian),
    ("Float32LE", SampleType::F32, ByteOrder::LittleEndian),
    ("Float32BE", SampleType::F32, ByteOrder::BigEndian),
    ("Float64LE", SampleType::F64, ByteOrder::LittleEndian),
    ("Float64BE", SampleType::F64, ByteOrder::BigEndian),
];

#[test]
fn parse_datatype_names_the_stored_type_and_byte_order() {
    for (spelling, sample_type, order) in DATATYPE_SPELLINGS {
        assert_eq!(
            parse_datatype(spelling).unwrap(),
            (sample_type, order),
            "{spelling}"
        );
        assert_eq!(
            parse_datatype(&spelling.to_ascii_lowercase()).unwrap(),
            (sample_type, order),
            "{spelling} is case-insensitive"
        );
    }
}

#[test]
fn a_multi_byte_datatype_without_a_suffix_takes_the_native_byte_order() {
    let native = if cfg!(target_endian = "big") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    };
    assert_eq!(parse_datatype("Int32").unwrap(), (SampleType::I32, native));
    assert_eq!(
        parse_datatype("Float64").unwrap(),
        (SampleType::F64, native)
    );
}

#[test]
fn datatype_name_inverts_parse_datatype_for_every_type_and_order() {
    for sample_type in SampleType::ALL {
        for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
            let name = datatype_name(sample_type, order);
            let (parsed, parsed_order) = parse_datatype(&name).unwrap();
            assert_eq!(parsed, sample_type, "{name}");
            if sample_type.byte_width() > 1 {
                assert_eq!(parsed_order, order, "{name}");
            }
        }
    }
}

#[test]
fn bit_and_complex_datatypes_are_rejected_with_the_reason() {
    for name in ["Bit", "CFloat32LE", "CFloat64BE", "cfloat32"] {
        assert_rejects(parse_datatype(name), "Bit and complex voxels are not one");
    }
}

#[test]
fn a_byte_order_on_a_one_byte_datatype_is_rejected() {
    assert_rejects(
        parse_datatype("UInt8LE"),
        "a one-byte type has no byte order",
    );
    assert_rejects(
        parse_datatype("Int8BE"),
        "a one-byte type has no byte order",
    );
}

#[test]
fn an_unknown_datatype_is_rejected_by_name() {
    assert_rejects(
        parse_datatype("Float16LE"),
        "Unknown .mif datatype 'Float16LE'",
    );
}

#[test]
fn parse_transform_identity() {
    let rows = vec![
        "1 0 0 -64".to_string(),
        "0 1 0 -64".to_string(),
        "0 0 1 -30".to_string(),
        "0 0 0 1".to_string(),
    ];
    let matrix = super::parse_transform(&rows).unwrap();
    assert_eq!(matrix[0], [1.0, 0.0, 0.0, -64.0]);
    assert_eq!(matrix[3], [0.0, 0.0, 0.0, 1.0]);
}
