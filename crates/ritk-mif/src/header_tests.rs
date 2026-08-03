//! Tests for the `.mif` header parser.

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
file: . 0
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
file: . 0
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
file: . 0
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
file: . 0
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
    assert!(result.is_err());
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
    assert!(result.is_err());
}

#[test]
fn parse_f64_vec_basic() {
    assert_eq!(
        super::parse_f64_vec("1.0 1.5 2.0").unwrap(),
        vec![1.0, 1.5, 2.0]
    );
}

#[test]
fn parse_layout_contiguous() {
    assert_eq!(
        super::parse_layout("+0,+1,+2,+3").unwrap(),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn parse_datatype_float32le() {
    let (size, signed, float) = super::parse_datatype("Float32LE").unwrap();
    assert_eq!(size, 4);
    assert!(signed);
    assert!(float);
}

#[test]
fn parse_datatype_int16be() {
    let (size, signed, float) = super::parse_datatype("Int16BE").unwrap();
    assert_eq!(size, 2);
    assert!(signed);
    assert!(!float);
}

#[test]
fn parse_datatype_uint8() {
    let (size, signed, float) = super::parse_datatype("UInt8").unwrap();
    assert_eq!(size, 1);
    assert!(!signed);
    assert!(!float);
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
