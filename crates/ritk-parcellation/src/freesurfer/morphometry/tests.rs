use super::*;

/// Laid out field by field from the format in the module docs.
fn encode(values: &[f32], face_count: i32, per_vertex: i32) -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xFF, 0xFF];
    let vertex_count = i32::try_from(values.len()).expect("small fixture");
    for field in [vertex_count, face_count, per_vertex] {
        bytes.extend_from_slice(&field.to_be_bytes());
    }
    for value in values {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    bytes
}

const THICKNESS: [f32; 4] = [2.5, -0.125, 3.75, 0.0];

#[test]
fn a_file_reads_to_exactly_its_values() {
    let data = Morphometry::read(encode(&THICKNESS, 6, 1).as_slice()).expect("valid");
    assert_eq!(data.values(), &THICKNESS);
    assert_eq!(data.face_count(), 6);
}

#[test]
fn the_writer_emits_the_specified_bytes() {
    let data = Morphometry::new(THICKNESS.to_vec().into_boxed_slice(), 6);
    let mut written = Vec::new();
    data.write(&mut written).expect("writes");
    assert_eq!(written, encode(&THICKNESS, 6, 1));
    assert_eq!(Morphometry::read(written.as_slice()).expect("reads"), data);
}

/// A NaN marks a vertex FreeSurfer could not measure; it is data, not damage.
#[test]
fn a_nan_value_is_kept_as_stored() {
    let data = Morphometry::read(encode(&[f32::NAN, 1.0], 0, 1).as_slice()).expect("valid");
    assert!(data.values()[0].is_nan());
    assert_eq!(data.values()[1], 1.0);
}

/// An old-format file starts with its three-byte vertex count, which must be
/// reported as a wrong magic rather than parsed as something else.
#[test]
fn an_old_format_file_is_rejected_by_its_magic() {
    let mut bytes = vec![0x00, 0x00, 0x03, 0x00, 0x00, 0x01];
    bytes.extend_from_slice(&[0; 6]);
    let error = Morphometry::read(bytes.as_slice()).expect_err("invalid input must be rejected");
    assert!(
        matches!(
            error,
            FreeSurferError::InvalidMagic {
                expected: 0x00FF_FFFF,
                got: 3,
                ..
            }
        ),
        "got {error}"
    );
}

#[test]
fn more_than_one_value_per_vertex_is_unsupported() {
    let error = Morphometry::read(encode(&THICKNESS, 6, 2).as_slice())
        .expect_err("invalid input must be rejected");
    assert!(
        matches!(
            error,
            FreeSurferError::Unsupported {
                field: "values per vertex",
                got: 2,
                ..
            }
        ),
        "got {error}"
    );
}

#[test]
fn a_negative_count_is_rejected() {
    let mut bytes = encode(&THICKNESS, 6, 1);
    bytes[3..7].copy_from_slice(&(-5_i32).to_be_bytes());
    let error = Morphometry::read(bytes.as_slice()).expect_err("invalid input must be rejected");
    assert!(
        matches!(
            error,
            FreeSurferError::InvalidCount {
                field: "vertex count",
                count: -5,
                ..
            }
        ),
        "got {error}"
    );
}

/// A vertex count larger than the data present fails on the data, having
/// reserved only a bounded amount of memory for it.
#[test]
fn a_count_exceeding_the_data_fails_as_truncation() {
    let mut bytes = encode(&THICKNESS, 6, 1);
    bytes[3..7].copy_from_slice(&9_999_999_i32.to_be_bytes());
    let error = Morphometry::read(bytes.as_slice()).expect_err("invalid input must be rejected");
    assert!(
        matches!(&error, FreeSurferError::Io(io) if io.kind() == std::io::ErrorKind::UnexpectedEof),
        "got {error}"
    );
}

#[test]
fn every_truncation_of_a_valid_file_is_rejected() {
    let bytes = encode(&THICKNESS, 6, 1);
    for length in 0..bytes.len() {
        assert!(
            Morphometry::read(&bytes[..length]).is_err(),
            "prefix of {length} bytes was accepted"
        );
    }
}
