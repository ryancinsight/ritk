use super::*;

#[test]
fn document_read_and_write_retain_format_metadata_and_exact_samples() -> Result<()> {
    let directory = tempdir()?;
    let input = directory.path().join("source.nrrd");
    let output = directory.path().join("replacement.nrrd");
    let mut source = std::fs::File::create(&input)?;
    writeln!(source, "NRRD0005")?;
    writeln!(source, "#source comment")?;
    writeln!(source, "type: short")?;
    writeln!(source, "dimension: 3")?;
    writeln!(source, "sizes: 2 1 1")?;
    writeln!(source, "space: left-posterior-superior")?;
    writeln!(source, "space directions: (1,0,0) (0,1,0) (0,0,1)")?;
    writeln!(source, "sample units: \"HU\"")?;
    writeln!(source, "thicknesses: 0.5 1 2")?;
    writeln!(source, "measurement frame: (1,0,0) (0,1,0) (0,0,1)")?;
    writeln!(source, "endian: big")?;
    writeln!(source, "encoding: raw")?;
    writeln!(source, "repeat:=first\\nline")?;
    writeln!(source, "repeat:=second\\\\tail")?;
    writeln!(source)?;
    source.write_all(&[0x12, 0x34, 0xab, 0xcd])?;

    let document = read_nrrd_document(&input, ImageReadBudget::DEFAULT)?;
    assert_eq!(document.header().format_version(), 5);
    assert_eq!(document.header().comments(), ["#source comment"]);
    assert_eq!(document.sizes(), [2, 1, 1]);
    assert_eq!(document.sample_type(), SampleType::I16);
    assert_eq!(document.byte_order(), ByteOrder::MostSignificantByteFirst);
    assert_eq!(document.sample_count(), 2);
    assert_eq!(document.sample_bytes(), [0x12, 0x34, 0xab, 0xcd]);
    assert_eq!(
        document.header().key_value_records(),
        [
            ("repeat".to_owned(), "first\nline".to_owned()),
            ("repeat".to_owned(), "second\\tail".to_owned())
        ]
    );
    assert_eq!(
        document
            .header()
            .key_values()
            .get("repeat")
            .map(String::as_str),
        Some("second\\tail")
    );

    let stored_error = read_nrrd_stored(&input)
        .expect_err("the volume projection cannot represent NRRD sample units");
    assert!(matches!(
        stored_error,
        NrrdStoredReadError::UnsupportedSampleUnits { .. }
    ));

    std::fs::write(&output, b"old destination")?;
    write_nrrd_document(&output, &document)?;
    let written = std::fs::read(&output)?;
    let separator = written
        .windows(2)
        .position(|window| window == b"\n\n")
        .expect("writer emits the required blank header separator");
    let serialized_header = std::str::from_utf8(&written[..separator])?;
    let field_position = |field: &str| {
        serialized_header
            .find(field)
            .unwrap_or_else(|| panic!("serialized header is missing {field:?}"))
    };
    let dimension_position = field_position("dimension: 3");
    let space_position = field_position("space: left-posterior-superior");
    assert!(dimension_position < field_position("sizes: 2 1 1"));
    assert!(dimension_position < field_position("thicknesses: 0.5 1 2"));
    assert!(dimension_position < field_position("space directions:"));
    assert!(space_position < field_position("measurement frame:"));
    assert!(space_position < field_position("space directions:"));
    let round_trip = read_nrrd_document(&output, ImageReadBudget::DEFAULT)?;
    assert_eq!(round_trip.header().format_version(), 5);
    assert_eq!(round_trip.header().comments(), ["#source comment"]);
    assert_eq!(
        round_trip.header().fields().get("sample units"),
        Some(&"\"HU\"".to_owned())
    );
    assert_eq!(
        round_trip.header().fields().get("thicknesses"),
        Some(&"0.5 1 2".to_owned())
    );
    assert_eq!(
        round_trip.header().fields().get("measurement frame"),
        Some(&"(1,0,0) (0,1,0) (0,0,1)".to_owned())
    );
    assert_eq!(
        round_trip.header().key_value_records(),
        document.header().key_value_records()
    );
    assert_eq!(round_trip.sample_type(), document.sample_type());
    assert_eq!(round_trip.byte_order(), document.byte_order());
    assert_eq!(round_trip.sample_bytes(), document.sample_bytes());
    assert!(!round_trip.header().fields().contains_key("data file"));
    assert!(!round_trip.header().fields().contains_key("byte skip"));
    assert!(!round_trip.header().fields().contains_key("line skip"));
    Ok(())
}

#[test]
fn document_round_trip_preserves_all_scalar_bits_in_both_byte_orders() -> Result<()> {
    let directory = tempdir()?;
    for (sample_type, element_type, samples) in sample_cases() {
        for byte_order in [
            ByteOrder::LeastSignificantByteFirst,
            ByteOrder::MostSignificantByteFirst,
        ] {
            let suffix = match byte_order {
                ByteOrder::LeastSignificantByteFirst => "little",
                ByteOrder::MostSignificantByteFirst => "big",
            };
            let input = directory
                .path()
                .join(format!("{element_type}-{suffix}.nrrd"));
            let output = directory
                .path()
                .join(format!("{element_type}-{suffix}-written.nrrd"));
            let type_field = format!("type: {element_type}");
            let endian_field = format!("endian: {suffix}");
            let fields = [
                type_field.as_str(),
                "dimension: 3",
                "sizes: 2 1 1",
                endian_field.as_str(),
            ];
            let expected = samples.encode(byte_order)?;
            write_header(&input, &fields, &expected)?;

            let document = read_nrrd_document(&input, ImageReadBudget::DEFAULT)?;
            assert_eq!(document.sample_type(), sample_type);
            assert_eq!(document.byte_order(), byte_order);
            assert_eq!(document.sample_bytes(), expected);

            write_nrrd_document(&output, &document)?;
            let round_trip = read_nrrd_document(&output, ImageReadBudget::DEFAULT)?;
            assert_eq!(round_trip.sample_type(), sample_type);
            assert_eq!(round_trip.byte_order(), byte_order);
            assert_eq!(round_trip.sample_bytes(), expected);
        }
    }
    Ok(())
}

#[test]
fn document_read_normalizes_ascii_and_gzip_payloads_to_inline_raw() -> Result<()> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let directory = tempdir()?;
    let ascii = directory.path().join("ascii.nrrd");
    let ascii_output = directory.path().join("ascii-written.nrrd");
    write_header(
        &ascii,
        &[
            "type: short",
            "dimension: 3",
            "sizes: 2 1 1",
            "encoding: ascii",
        ],
        b"-2 4660",
    )?;
    let ascii_document = read_nrrd_document(&ascii, ImageReadBudget::DEFAULT)?;
    assert_eq!(
        ascii_document.byte_order(),
        ByteOrder::LeastSignificantByteFirst
    );
    assert_eq!(ascii_document.sample_bytes(), [0xfe, 0xff, 0x34, 0x12]);
    write_nrrd_document(&ascii_output, &ascii_document)?;
    let ascii_round_trip = read_nrrd_document(&ascii_output, ImageReadBudget::DEFAULT)?;
    assert_eq!(
        ascii_round_trip.sample_bytes(),
        ascii_document.sample_bytes()
    );
    assert_eq!(
        ascii_round_trip
            .header()
            .fields()
            .get("encoding")
            .map(String::as_str),
        Some("raw")
    );
    assert_eq!(
        ascii_round_trip
            .header()
            .fields()
            .get("endian")
            .map(String::as_str),
        Some("little")
    );

    let gzip = directory.path().join("gzip.nrrd");
    let gzip_output = directory.path().join("gzip-written.nrrd");
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&[0x12, 0x34, 0xab, 0xcd])?;
    let compressed = encoder.finish()?;
    write_header(
        &gzip,
        &[
            "type: short",
            "dimension: 3",
            "sizes: 2 1 1",
            "endian: big",
            "encoding: gzip",
        ],
        &compressed,
    )?;
    let gzip_document = read_nrrd_document(&gzip, ImageReadBudget::DEFAULT)?;
    assert_eq!(
        gzip_document.byte_order(),
        ByteOrder::MostSignificantByteFirst
    );
    assert_eq!(gzip_document.sample_bytes(), [0x12, 0x34, 0xab, 0xcd]);
    write_nrrd_document(&gzip_output, &gzip_document)?;
    let gzip_round_trip = read_nrrd_document(&gzip_output, ImageReadBudget::DEFAULT)?;
    assert_eq!(gzip_round_trip.sample_bytes(), gzip_document.sample_bytes());
    assert_eq!(
        gzip_round_trip
            .header()
            .fields()
            .get("encoding")
            .map(String::as_str),
        Some("raw")
    );
    Ok(())
}

#[test]
fn document_read_inlines_detached_data_and_removes_skip_fields() -> Result<()> {
    let directory = tempdir()?;
    let input = directory.path().join("source.nhdr");
    let detached = directory.path().join("payload.raw");
    let output = directory.path().join("inline.nrrd");
    std::fs::write(&detached, b"preamble\r\n?\x2aextra")?;
    write_header(
        &input,
        &[
            "type: unsigned char",
            "dimension: 3",
            "sizes: 1 1 1",
            "line skip: 1",
            "byte skip: 1",
            "data file: payload.raw",
        ],
        &[],
    )?;

    let document = read_nrrd_document(&input, ImageReadBudget::DEFAULT)?;
    assert_eq!(document.sample_bytes(), [0x2a]);
    write_nrrd_document(&output, &document)?;
    let round_trip = read_nrrd_document(&output, ImageReadBudget::DEFAULT)?;
    assert_eq!(round_trip.sample_bytes(), [0x2a]);
    assert!(!round_trip.header().fields().contains_key("data file"));
    assert!(!round_trip.header().fields().contains_key("line skip"));
    assert!(!round_trip.header().fields().contains_key("byte skip"));
    assert_eq!(
        round_trip
            .header()
            .fields()
            .get("encoding")
            .map(String::as_str),
        Some("raw")
    );
    Ok(())
}

#[test]
fn document_writer_stays_within_the_reader_record_limit_for_byte_samples() -> Result<()> {
    let directory = tempdir()?;
    let input = directory.path().join("maximum-records.nrrd");
    let output = directory.path().join("maximum-records-written.nrrd");
    let mut source = std::fs::File::create(&input)?;
    writeln!(source, "NRRD0004")?;
    writeln!(source, "type: unsigned char")?;
    writeln!(source, "dimension: 3")?;
    writeln!(source, "sizes: 1 1 1")?;
    writeln!(source, "encoding: raw")?;
    for _ in 0..(crate::reader::MAX_HEADER_ENTRIES - 4) {
        writeln!(source, "x:=")?;
    }
    writeln!(source)?;
    source.write_all(&[0x5a])?;

    let document = read_nrrd_document(&input, ImageReadBudget::DEFAULT)?;
    write_nrrd_document(&output, &document)?;
    let round_trip = read_nrrd_document(&output, ImageReadBudget::DEFAULT)?;
    assert_eq!(round_trip.sample_bytes(), [0x5a]);
    assert_eq!(
        round_trip.header().key_value_records().len(),
        crate::reader::MAX_HEADER_ENTRIES - 4
    );
    assert!(!round_trip.header().fields().contains_key("endian"));
    Ok(())
}

#[test]
fn document_writer_rejects_added_endian_before_replacing_output() -> Result<()> {
    let directory = tempdir()?;
    let input = directory.path().join("maximum-ascii-records.nrrd");
    let output = directory.path().join("existing-output.nrrd");
    let mut source = std::fs::File::create(&input)?;
    writeln!(source, "NRRD0004")?;
    writeln!(source, "type: short")?;
    writeln!(source, "dimension: 3")?;
    writeln!(source, "sizes: 1 1 1")?;
    writeln!(source, "encoding: ascii")?;
    for _ in 0..(crate::reader::MAX_HEADER_ENTRIES - 4) {
        writeln!(source, "x:=")?;
    }
    writeln!(source)?;
    source.write_all(b"7")?;

    let document = read_nrrd_document(&input, ImageReadBudget::DEFAULT)?;
    std::fs::write(&output, b"existing destination")?;
    let error = write_nrrd_document(&output, &document)
        .expect_err("the normalized raw output needs one more header record");
    assert!(matches!(
        error,
        crate::NrrdDocumentWriteError::HeaderTooManyEntries {
            entries,
            maximum_entries
        } if entries == maximum_entries + 1
    ));
    assert_eq!(std::fs::read(&output)?, b"existing destination");
    Ok(())
}

#[test]
fn volume_overflow_keeps_its_spatial_diagnostic() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("voxel-overflow.nrrd");
    let overflowing_axis = usize::MAX / 2 + 1;
    let sizes_field = format!("sizes: {overflowing_axis} 2 1");
    write_header(
        &path,
        &["type: unsigned char", "dimension: 3", sizes_field.as_str()],
        &[],
    )?;
    assert!(matches!(
        read_nrrd_stored(&path),
        Err(NrrdStoredReadError::VoxelCountOverflow { .. })
    ));
    Ok(())
}
