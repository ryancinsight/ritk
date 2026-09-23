use super::*;

// Fixtures are laid out field by field from the format in the module docs,
// never produced by the writer under test.

const UNKNOWN: (i32, &str, [i32; 4]) = (0, "unknown", [25, 5, 25, 0]);
const PRECENTRAL: (i32, &str, [i32; 4]) = (1, "precentral", [60, 20, 220, 0]);
const BANKSSTS: (i32, &str, [i32; 4]) = (3, "bankssts", [25, 100, 40, 0]);

fn packed(rgbt: [i32; 4]) -> i32 {
    rgbt[0] + rgbt[1] * 256 + rgbt[2] * 65_536
}

fn push(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_string(bytes: &mut Vec<u8>, text: &str) {
    push(bytes, i32::try_from(text.len() + 1).expect("short"));
    bytes.extend_from_slice(text.as_bytes());
    bytes.push(0);
}

/// Four vertices, records deliberately out of vertex order.
fn vertex_records(bytes: &mut Vec<u8>) {
    push(bytes, 4);
    for (vertex, value) in [
        (2, packed(BANKSSTS.2)),
        (0, packed(PRECENTRAL.2)),
        (1, 0),
        (3, packed(UNKNOWN.2)),
    ] {
        push(bytes, vertex);
        push(bytes, value);
    }
    push(bytes, 1); // colour-table tag
}

fn version_2_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    vertex_records(&mut bytes);
    push(&mut bytes, -2);
    push(&mut bytes, 4); // max index
    push_string(&mut bytes, "NOFILE");
    push(&mut bytes, 3);
    for (index, name, rgbt) in [UNKNOWN, PRECENTRAL, BANKSSTS] {
        push(&mut bytes, index);
        push_string(&mut bytes, name);
        rgbt.into_iter().for_each(|value| push(&mut bytes, value));
    }
    bytes
}

fn old_format_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    vertex_records(&mut bytes);
    push(&mut bytes, 4); // entry count; index is position
    push_string(&mut bytes, "/usr/local/freesurfer/FreeSurferColorLUT.txt");
    let unused = (2, "unused", [1, 2, 3, 0]);
    for (_, name, rgbt) in [UNKNOWN, PRECENTRAL, unused, BANKSSTS] {
        push_string(&mut bytes, name);
        rgbt.into_iter().for_each(|value| push(&mut bytes, value));
    }
    bytes
}

fn expected_names(annotation: &SurfaceAnnotation) -> Vec<(u32, String)> {
    annotation.color_table().region_names()
}

#[test]
fn a_version_2_file_reads_to_its_structure_indices() {
    let annotation = SurfaceAnnotation::read(version_2_bytes().as_slice()).expect("valid");

    assert_eq!(annotation.vertex_labels(), &[1, BACKGROUND, 3, 0]);
    assert_eq!(
        expected_names(&annotation),
        vec![
            (0, "unknown".to_owned()),
            (1, "precentral".to_owned()),
            (3, "bankssts".to_owned()),
        ]
    );
    let precentral = annotation.color_table().get(1).expect("present");
    assert_eq!(
        precentral.color(),
        LutColor {
            red: 60,
            green: 20,
            blue: 220,
            transparency: 0
        }
    );
}

#[test]
fn an_old_format_file_indexes_entries_by_position() {
    let annotation = SurfaceAnnotation::read(old_format_bytes().as_slice()).expect("valid");

    assert_eq!(annotation.vertex_labels(), &[1, BACKGROUND, 3, 0]);
    assert_eq!(
        annotation.color_table().get(2).map(LutEntry::name),
        Some("unused")
    );
}

/// The writer emits version 2 exactly as laid out in the fixture, apart from
/// vertex records, which it writes in vertex order.
#[test]
fn the_writer_emits_version_2_and_round_trips() {
    let annotation = SurfaceAnnotation::read(version_2_bytes().as_slice()).expect("valid");
    let mut written = Vec::new();
    annotation.write(&mut written).expect("writes");

    let mut expected = Vec::new();
    push(&mut expected, 4);
    for (vertex, value) in [
        (0, packed(PRECENTRAL.2)),
        (1, 0),
        (2, packed(BANKSSTS.2)),
        (3, 0),
    ] {
        push(&mut expected, vertex);
        push(&mut expected, value);
    }
    push(&mut expected, 1);
    expected.extend_from_slice(&version_2_bytes()[4 + 4 * 8 + 4..]);
    assert_eq!(written, expected);

    assert_eq!(
        SurfaceAnnotation::read(written.as_slice()).expect("reads"),
        annotation
    );
}

fn error_of(bytes: &[u8]) -> FreeSurferError {
    SurfaceAnnotation::read(bytes).expect_err("invalid input must be rejected")
}

#[test]
fn a_negative_vertex_count_is_rejected() {
    let mut bytes = Vec::new();
    push(&mut bytes, -1);
    assert!(matches!(
        error_of(&bytes),
        FreeSurferError::InvalidCount {
            field: "vertex count",
            count: -1,
            ..
        }
    ));
}

/// A count far beyond the data must fail on the missing data, not on an
/// allocation the header demanded.
#[test]
fn a_count_exceeding_the_data_fails_as_truncation() {
    let mut bytes = Vec::new();
    push(&mut bytes, 9_000_000);
    push(&mut bytes, 0);
    let error = error_of(&bytes);
    assert!(
        matches!(&error, FreeSurferError::Io(io) if io.kind() == std::io::ErrorKind::UnexpectedEof),
        "got {error}"
    );
}

#[test]
fn every_truncation_of_a_valid_file_is_rejected() {
    let bytes = version_2_bytes();
    for length in 0..bytes.len() {
        assert!(
            SurfaceAnnotation::read(&bytes[..length]).is_err(),
            "prefix of {length} bytes was accepted"
        );
    }
}

#[test]
fn a_vertex_outside_the_surface_is_rejected() {
    let mut bytes = version_2_bytes();
    bytes[4..8].copy_from_slice(&4_i32.to_be_bytes()); // first record's vertex
    assert!(matches!(
        error_of(&bytes),
        FreeSurferError::Malformed {
            field: "vertex record",
            index: 0,
            ..
        }
    ));
}

#[test]
fn a_value_no_entry_has_is_rejected() {
    let mut bytes = version_2_bytes();
    bytes[8..12].copy_from_slice(&0x0012_3456_i32.to_be_bytes()); // first record's value
    assert!(matches!(
        error_of(&bytes),
        FreeSurferError::Malformed {
            field: "vertex record",
            index: 0,
            ..
        }
    ));
}

#[test]
fn a_missing_colour_table_is_unsupported() {
    let mut bytes = version_2_bytes();
    bytes[36..40].copy_from_slice(&0_i32.to_be_bytes()); // the tag
    assert!(matches!(
        error_of(&bytes),
        FreeSurferError::Unsupported {
            field: "colour-table tag",
            got: 0,
            ..
        }
    ));
}

#[test]
fn a_colour_table_version_other_than_2_is_unsupported() {
    let mut bytes = version_2_bytes();
    bytes[40..44].copy_from_slice(&(-3_i32).to_be_bytes());
    assert!(matches!(
        error_of(&bytes),
        FreeSurferError::Unsupported {
            field: "colour-table version",
            got: 3,
            ..
        }
    ));
}

#[test]
fn an_oversized_string_length_is_rejected() {
    let mut bytes = version_2_bytes();
    bytes[48..52].copy_from_slice(&5000_i32.to_be_bytes()); // path length
    assert!(matches!(
        error_of(&bytes),
        FreeSurferError::InvalidCount {
            field: "string length",
            count: 5000,
            ..
        }
    ));
}

#[test]
fn an_entry_index_beyond_max_index_is_rejected() {
    let mut bytes = Vec::new();
    vertex_records(&mut bytes);
    push(&mut bytes, -2);
    push(&mut bytes, 2); // max index below bankssts' 3
    push_string(&mut bytes, "NOFILE");
    push(&mut bytes, 1);
    push(&mut bytes, 3);
    push_string(&mut bytes, "bankssts");
    [25, 100, 40, 0]
        .into_iter()
        .for_each(|value| push(&mut bytes, value));
    assert!(matches!(
        error_of(&bytes),
        FreeSurferError::Malformed {
            field: "colour-table entry",
            index: 0,
            ..
        }
    ));
}

#[test]
fn a_colour_component_outside_a_byte_is_rejected() {
    let mut bytes = Vec::new();
    push(&mut bytes, 0);
    push(&mut bytes, 1);
    push(&mut bytes, -2);
    push(&mut bytes, 1);
    push_string(&mut bytes, "NOFILE");
    push(&mut bytes, 1);
    push(&mut bytes, 0);
    push_string(&mut bytes, "unknown");
    [300, 0, 0, 0]
        .into_iter()
        .for_each(|value| push(&mut bytes, value));
    assert!(matches!(
        error_of(&bytes),
        FreeSurferError::Malformed {
            field: "colour-table entry",
            index: 0,
            ..
        }
    ));
}

#[test]
fn entries_sharing_a_colour_are_rejected() {
    let table = ColorLut::new([
        LutEntry::new(1, "a".to_owned(), LutColor::default()).expect("valid"),
        LutEntry::new(2, "b".to_owned(), LutColor::default()).expect("valid"),
    ])
    .expect("unique labels");
    let error = SurfaceAnnotation::new(vec![1, 2].into_boxed_slice(), table)
        .expect_err("invalid input must be rejected");
    assert!(matches!(
        error,
        FreeSurferError::Malformed {
            field: "colour-table entry",
            index: 2,
            ..
        }
    ));
}

#[test]
fn a_label_missing_from_the_table_is_rejected() {
    let table = ColorLut::new([LutEntry::new(
        1,
        "a".to_owned(),
        LutColor {
            red: 1,
            ..LutColor::default()
        },
    )
    .expect("valid")])
    .expect("unique");
    let error = SurfaceAnnotation::new(vec![1, 0, 7].into_boxed_slice(), table)
        .expect_err("invalid input must be rejected");
    assert!(matches!(
        error,
        FreeSurferError::Malformed {
            field: "vertex",
            index: 2,
            ..
        }
    ));
}
