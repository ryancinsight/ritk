//! NRRD header parsing tests.

use std::io::Cursor;

use super::{parse_nrrd_header_from_reader, NrrdHeaderError, MAX_HEADER_BYTES, MAX_HEADER_ENTRIES};

#[test]
fn non_ascii_header_lines_are_rejected() {
    for input in [
        b"NRRD0005\n# caf\xc3\xa9\n\n".as_slice(),
        b"NRRD0005\ncontent: caf\xc3\xa9\n\n",
        b"NRRD0005\ncustom:=caf\xc3\xa9\n\n",
    ] {
        let mut reader = Cursor::new(input);

        assert!(matches!(
            parse_nrrd_header_from_reader(&mut reader),
            Err(NrrdHeaderError::NonAsciiLine { line_number: 2 })
        ));
    }
}

#[test]
fn public_header_reader_retains_comments_and_repeated_custom_records() {
    let directory = tempfile::tempdir().expect("create header test directory");
    let path = directory.path().join("metadata.nrrd");
    std::fs::write(
        &path,
        b"NRRD0005\n# first comment\ntype: unsigned char\ncustom:=first\n# second comment\ncustom:=second\n\n",
    )
    .expect("write header fixture");

    let header = crate::read_nrrd_header(&path).expect("read public header");

    assert_eq!(header.format_version(), 5);
    assert_eq!(
        header.fields().get("type").map(String::as_str),
        Some("unsigned char")
    );
    assert_eq!(
        header.key_values().get("custom").map(String::as_str),
        Some("second")
    );
    let records = header.key_value_records();
    assert_eq!(records.len(), 2);
    let [first, second] = records else {
        panic!("two custom records were parsed");
    };
    assert_eq!(first.key(), "custom");
    assert_eq!(first.value(), "first");
    assert_eq!(second.key(), "custom");
    assert_eq!(second.value(), "second");
    assert_eq!(
        header.comments(),
        [
            String::from("# first comment"),
            String::from("# second comment")
        ]
    );
}

#[test]
fn repeated_custom_records_count_toward_the_header_entry_limit() {
    let mut accepted = Vec::with_capacity(MAX_HEADER_ENTRIES * 10);
    accepted.extend_from_slice(b"NRRD0005\n");
    for _ in 0..MAX_HEADER_ENTRIES {
        accepted.extend_from_slice(b"custom:=x\n");
    }
    accepted.push(b'\n');
    let mut reader = Cursor::new(accepted);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("limit permits exact count");
    assert_eq!(header.key_value_records.len(), MAX_HEADER_ENTRIES);

    let mut rejected = Vec::with_capacity((MAX_HEADER_ENTRIES + 1) * 10);
    rejected.extend_from_slice(b"NRRD0005\n");
    for _ in 0..=MAX_HEADER_ENTRIES {
        rejected.extend_from_slice(b"custom:=x\n");
    }
    rejected.push(b'\n');
    let mut reader = Cursor::new(rejected);

    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::TooManyEntries { maximum_entries })
            if maximum_entries == MAX_HEADER_ENTRIES
    ));
}

#[test]
fn comments_count_toward_the_header_entry_limit() {
    let mut accepted = Vec::with_capacity(MAX_HEADER_ENTRIES * 4 + 10);
    accepted.extend_from_slice(b"NRRD0005\n");
    for _ in 0..MAX_HEADER_ENTRIES {
        accepted.extend_from_slice(b"# x\n");
    }
    accepted.push(b'\n');
    let mut reader = Cursor::new(accepted);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("limit permits exact count");
    assert_eq!(header.comments.len(), MAX_HEADER_ENTRIES);

    let mut rejected = Vec::with_capacity((MAX_HEADER_ENTRIES + 1) * 4 + 10);
    rejected.extend_from_slice(b"NRRD0005\n");
    for _ in 0..=MAX_HEADER_ENTRIES {
        rejected.extend_from_slice(b"# x\n");
    }
    rejected.push(b'\n');
    let mut reader = Cursor::new(rejected);

    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::TooManyEntries { maximum_entries })
            if maximum_entries == MAX_HEADER_ENTRIES
    ));
}

#[test]
fn entry_capacity_rejects_before_malformed_record_decoding() {
    let mut input = Vec::with_capacity(MAX_HEADER_ENTRIES * 4 + 32);
    input.extend_from_slice(b"NRRD0005\n");
    for _ in 0..MAX_HEADER_ENTRIES {
        input.extend_from_slice(b"# x\n");
    }
    input.extend_from_slice(b"custom:=\\q\n\n");
    let mut reader = Cursor::new(input);

    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::TooManyEntries { maximum_entries })
            if maximum_entries == MAX_HEADER_ENTRIES
    ));
}

#[test]
fn ignored_comments_count_toward_the_header_byte_limit() {
    let prefix = b"NRRD0005\n";
    let suffix = b"\n\n";
    let comment_bytes = MAX_HEADER_BYTES - prefix.len() - suffix.len();
    let mut input = Vec::with_capacity(MAX_HEADER_BYTES + 1);
    input.extend_from_slice(prefix);
    input.resize(input.len() + comment_bytes, b'#');
    input.extend_from_slice(suffix);
    let mut reader = Cursor::new(&input);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("exact byte limit is valid");
    assert!(header.comments.is_empty());

    input.insert(input.len() - suffix.len(), b'#');
    let mut reader = Cursor::new(input);
    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::HeaderTooLarge { maximum_bytes })
            if maximum_bytes == MAX_HEADER_BYTES
    ));
}

#[test]
fn empty_comment_strings_are_ignored_and_do_not_consume_entry_capacity() {
    let mut input = Vec::with_capacity((MAX_HEADER_ENTRIES + 1) * 2 + 32);
    input.extend_from_slice(b"NRRD0005\ndimension: 3\n");
    for _ in 0..=MAX_HEADER_ENTRIES {
        input.extend_from_slice(b"# \n");
    }
    input.push(b'\n');
    let mut reader = Cursor::new(input);

    let header = parse_nrrd_header_from_reader(&mut reader)
        .expect("empty comment strings are not retained entries");
    assert_eq!(
        header.fields.get("dimension").map(String::as_str),
        Some("3")
    );
    assert!(header.comments.is_empty());
}

#[test]
fn fields_comments_and_records_share_the_entry_limit() {
    let record_count = MAX_HEADER_ENTRIES - 2;
    let mut accepted = Vec::with_capacity(record_count * 10 + 32);
    accepted.extend_from_slice(b"NRRD0005\ndimension: 3\n# retained\n");
    for _ in 0..record_count {
        accepted.extend_from_slice(b"custom:=x\n");
    }
    accepted.push(b'\n');
    let mut reader = Cursor::new(accepted);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("exact combined limit is valid");
    assert_eq!(header.fields.len(), 1);
    assert_eq!(header.comments.len(), 1);
    assert_eq!(header.key_value_records.len(), record_count);

    let mut rejected = Vec::with_capacity((record_count + 1) * 10 + 32);
    rejected.extend_from_slice(b"NRRD0005\ndimension: 3\n# retained\n");
    for _ in 0..=record_count {
        rejected.extend_from_slice(b"custom:=x\n");
    }
    rejected.push(b'\n');
    let mut reader = Cursor::new(rejected);
    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::TooManyEntries { maximum_entries })
            if maximum_entries == MAX_HEADER_ENTRIES
    ));
}

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
fn custom_key_value_delimiter_takes_precedence_inside_the_key() {
    let input = b"NRRD0005
custom: name:=value

";
    let mut reader = Cursor::new(input);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("valid custom record");

    assert_eq!(
        header.key_values.get("custom: name").map(String::as_str),
        Some("value")
    );
    assert_eq!(header.key_value_records.len(), 1);
    assert_eq!(header.key_value_records[0].key(), "custom: name");
    assert_eq!(header.key_value_records[0].value(), "value");
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
fn escaped_record_keys_and_values_preserve_case_and_spaces() {
    let input = b"NRRD0005\n Key\\nPart\\\\ := Value\\nTail\\\\ \n\n";
    let mut reader = Cursor::new(input);
    let header = parse_nrrd_header_from_reader(&mut reader).expect("valid escaped record");
    let record = header.key_value_records.first().expect("retained record");

    assert_eq!(
        (record.key(), record.value()),
        (" Key\nPart\\ ", " Value\nTail\\ ")
    );
    assert_eq!(
        header.key_values.get(record.key()).map(String::as_str),
        Some(record.value())
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
        ("block size", "blocksize", "4"),
        ("old min", "oldmin", "0"),
        ("old max", "oldmax", "255"),
        ("sample units", "sampleunits", "HU"),
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
fn compact_alias_reports_canonical_version_error() {
    let mut reader = Cursor::new(b"NRRD0003\nSAMPLEUNITS: HU\n\n");

    assert!(matches!(
        parse_nrrd_header_from_reader(&mut reader),
        Err(NrrdHeaderError::FieldRequiresVersion {
            field,
            minimum_version: 4,
            actual_version: 3,
        }) if field == "sample units"
    ));
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
