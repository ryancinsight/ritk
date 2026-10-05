//! NRRD header parsing tests.

use std::io::Cursor;

use super::{parse_nrrd_header_from_reader, NrrdHeaderError, MAX_HEADER_BYTES};

#[test]
fn header_fields_and_custom_pairs_keep_separate_namespaces() {
    let input = b"NRRD0005\ntype: unsigned char\nsizes: 2 1 1\ntype:=signed char\nsizes:=1 1 1\n\n";
    let mut reader = Cursor::new(input);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("valid header");

    assert_eq!(
        header.fields.get("type").map(String::as_str),
        Some("unsigned char")
    );
    assert_eq!(
        header.fields.get("sizes").map(String::as_str),
        Some("2 1 1")
    );
    assert_eq!(
        header.key_values.get("type").map(String::as_str),
        Some("signed char")
    );
    assert_eq!(
        header.key_values.get("sizes").map(String::as_str),
        Some("1 1 1")
    );
}

#[test]
fn header_key_values_preserve_case_unescape_and_last_value() {
    let input = b"NRRD0005\nDWMRI_gradient_0000:=one\\ntwo\\\\three\nDWMRI_gradient_0000:=last\n\n";
    let mut reader = Cursor::new(input);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("valid header");

    assert_eq!(
        header
            .key_values
            .get("DWMRI_gradient_0000")
            .map(String::as_str),
        Some("last")
    );
    let input = b"NRRD0005\nCustomCase:=one\\ntwo\\\\three\n\n";
    let mut reader = Cursor::new(input);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("valid escapes");
    assert_eq!(
        header.key_values.get("CustomCase").map(String::as_str),
        Some("one\ntwo\\three")
    );
}

#[test]
fn header_rejects_duplicate_fields_and_missing_separator() {
    let input = b"NRRD0005\nsizes: 1\nSizes: 2\n\n";
    let mut reader = Cursor::new(input);
    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::DuplicateField { field }) if field == "sizes"
    ));

    let input = b"NRRD0005\nsizes: 1\n";
    let mut reader = Cursor::new(input);
    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::MissingSeparator)
    ));
}

#[test]
fn standard_field_aliases_share_one_canonical_key_and_reject_conflicts() {
    for (canonical, alias, value) in [
        ("byte skip", "byteskip", "12"),
        ("line skip", "lineskip", "3"),
        ("data file", "datafile", "volume.raw"),
        ("axis mins", "axismins", "0 0 0"),
        ("axis maxs", "axismaxs", "1 1 1"),
        ("centers", "centerings", "\"node\" \"node\" \"node\""),
    ] {
        let input = format!("NRRD0005\n{alias}: {value}\n\n");
        let mut reader = Cursor::new(input);
        let header = parse_nrrd_header_from_reader(&mut reader).expect("valid alias");
        assert_eq!(
            header.fields.get(canonical).map(String::as_str),
            Some(value)
        );

        for (first, second) in [(canonical, alias), (alias, canonical)] {
            let input = format!("NRRD0005\n{first}: {value}\n{second}: other\n\n");
            let mut reader = Cursor::new(input);
            assert!(matches!(
                parse_nrrd_header_from_reader(&mut reader),
                Err(NrrdHeaderError::DuplicateField { field: duplicate })
                    if duplicate == canonical
            ));
        }
    }
}

#[test]
fn header_field_values_trim_separator_whitespace() {
    let input = b"NRRD0005\nsizes:  2 1 1  \n\n";
    let mut reader = Cursor::new(input);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("valid header");

    assert_eq!(
        header.fields.get("sizes").map(String::as_str),
        Some("2 1 1")
    );
}

#[test]
fn supported_magic_versions_are_exact_and_version_one_rejects_key_values() {
    for magic in [
        "NRRD0001",
        "NRRD00.01",
        "NRRD0002",
        "NRRD0003",
        "NRRD0004",
        "NRRD0005",
    ] {
        let input = format!("{magic}\ntype: unsigned char\n\n");
        let mut reader = Cursor::new(input);
        let header =
            parse_nrrd_header_from_reader(&mut reader).expect("documented NRRD magic is supported");
        assert_eq!(
            header.fields.get("type").map(String::as_str),
            Some("unsigned char")
        );
    }

    for magic in ["NRRDgarbage", "NRRD9999", "NRRD0006", "NARRD0005"] {
        let input = format!("{magic}\ntype: unsigned char\n\n");
        let mut reader = Cursor::new(input);
        assert!(matches!(
            parse_nrrd_header_from_reader(&mut reader),
            Err(NrrdHeaderError::InvalidMagic)
        ));
    }

    let input = b"NRRD0001\ntype: unsigned char\ncustom:=value\n\n";
    let mut reader = Cursor::new(input);
    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::KeyValueBeforeVersionTwo { line_number: 3 })
    ));
}

#[test]
fn standard_fields_are_rejected_before_their_introducing_version() {
    for (magic, field, minimum_version, actual_version) in [
        ("NRRD0002", "kinds: domain", 3, 2),
        ("NRRD0003", "space directions: (1,0,0)", 4, 3),
        ("NRRD0004", "measurement frame: (1,0,0)", 5, 4),
    ] {
        let input = format!("{magic}\n{field}\n\n");
        let mut reader = Cursor::new(input);
        assert!(matches!(
            parse_nrrd_header_from_reader(&mut reader),
            Err(NrrdHeaderError::FieldRequiresVersion {
                minimum_version: minimum,
                actual_version: actual,
                ..
            }) if minimum == minimum_version && actual == actual_version
        ));
    }
}

#[test]
fn detached_header_may_end_at_eof_after_its_data_file_field() {
    let input = b"NRRD0004\ntype: unsigned char\ndata file: volume.raw";
    let mut reader = Cursor::new(input);
    let header = parse_nrrd_header_from_reader(&mut reader)
        .expect("detached header may terminate after data file");

    assert_eq!(
        header.fields.get("data file").map(String::as_str),
        Some("volume.raw")
    );
    assert_eq!(reader.position(), input.len() as u64);
}

#[test]
fn detached_header_alias_may_end_at_eof() {
    let input = b"NRRD0004\ntype: unsigned char\ndatafile: volume.raw";
    let mut reader = Cursor::new(input);
    let header = parse_nrrd_header_from_reader(&mut reader)
        .expect("canonicalized detached field may end the header");

    assert_eq!(
        header.fields.get("data file").map(String::as_str),
        Some("volume.raw")
    );
}

#[test]
fn detached_header_without_data_file_still_requires_separator() {
    let input = b"NRRD0004\ntype: unsigned char";
    let mut reader = Cursor::new(input);

    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::MissingSeparator)
    ));
}

#[test]
fn header_limit_bounds_a_single_unterminated_line() {
    let mut input = Vec::with_capacity(MAX_HEADER_BYTES + 1);
    input.extend_from_slice(b"NRRD0005\n");
    input.resize(MAX_HEADER_BYTES + 1, b'a');
    let mut reader = Cursor::new(input);

    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::HeaderTooLarge { maximum_bytes })
            if maximum_bytes == MAX_HEADER_BYTES
    ));
}
