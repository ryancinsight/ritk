//! Stored NRRD payload tests.

use super::*;

#[test]
fn stored_reader_uses_typed_format_errors_and_ignores_trailing_payload() -> Result<()> {
    let directory = tempdir()?;
    let oversized = directory.path().join("trailing-payload.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space: LPS",
        "encoding: raw",
    ];
    write_header(&oversized, &fields, &[7, 8])?;
    let decoded = read_nrrd_stored(&oversized)?;
    assert_eq!(
        decoded
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [7]
    );

    let truncated = directory.path().join("truncated-payload.nrrd");
    let truncated_fields = [
        "type: unsigned short",
        "dimension: 3",
        "sizes: 1 1 1",
        "endian: little",
        "encoding: raw",
    ];
    write_header(&truncated, &truncated_fields, &[7])?;
    let error = read_nrrd_stored(&truncated).expect_err("partial payload is rejected");
    assert!(matches!(
        error,
        NrrdStoredReadError::TruncatedPayload {
            expected_bytes: 2,
            actual_bytes: 1,
        }
    ));

    let unsupported = directory.path().join("unsupported-encoding.nrrd");
    let unsupported_fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "encoding: hex",
    ];
    write_header(&unsupported, &unsupported_fields, &[7])?;
    let error = read_nrrd_stored(&unsupported).expect_err("unsupported encoding is typed");
    assert!(matches!(
        error,
        NrrdStoredReadError::UnsupportedEncoding { encoding } if encoding == "hex"
    ));

    let invalid_sizes = directory.path().join("invalid-sizes.nrrd");
    let invalid_size_fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 0 1",
        "encoding: raw",
    ];
    write_header(&invalid_sizes, &invalid_size_fields, &[])?;
    let error = read_nrrd_stored(&invalid_sizes).expect_err("zero-length axis is typed");
    assert!(matches!(error, NrrdStoredReadError::EmptyAxis { axis: 1 }));
    Ok(())
}

#[test]
fn public_stored_reader_enforces_the_callers_decoded_byte_budget() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("decoded-budget.nrrd");
    let fields = [
        "type: unsigned short",
        "dimension: 3",
        "sizes: 2 1 1",
        "endian: little",
        "encoding: raw",
    ];
    write_header(&path, &fields, &[7, 0, 8, 0])?;
    let budget =
        ritk_image_io::ImageReadBudget::new(16, 3, 1).expect("all configured limits are nonzero");

    let error = crate::read_nrrd_stored(&path, budget)
        .expect_err("declared decoded bytes exceed the caller's budget");
    assert!(matches!(
        error,
        crate::NrrdStoredReadError::ReadBudget {
            source: ritk_image_io::ImageReadBudgetError::Exceeded {
                resource: ritk_image_io::ImageReadResource::DecodedBytes,
                actual: 4,
                maximum: 3,
            }
        }
    ));
    Ok(())
}

#[test]
fn ascii_encoding_decodes_every_fixed_width_integer_without_endian_metadata() -> Result<()> {
    let directory = tempdir()?;
    let values = [
        "0 255",
        "-128 127",
        "0 65535",
        "-32768 32767",
        "0 4294967295",
        "-2147483648 2147483647",
        "0 18446744073709551615",
        "-9223372036854775808 9223372036854775807",
    ];
    for ((sample_type, element_type, expected), text) in
        sample_cases().into_iter().take(values.len()).zip(values)
    {
        let path = directory.path().join(format!("{element_type}.nrrd"));
        let fields = [
            format!("type: {element_type}"),
            "dimension: 3".to_owned(),
            "sizes: 2 1 1".to_owned(),
            "encoding: ascii".to_owned(),
        ];
        let fields = fields.iter().map(String::as_str).collect::<Vec<_>>();
        write_header(&path, &fields, text.as_bytes())?;

        let actual = read_nrrd_stored(&path)?;
        assert_eq!(actual.samples().sample_type(), sample_type);
        assert_eq!(
            actual
                .samples()
                .encode(ByteOrder::LeastSignificantByteFirst)?,
            expected.encode(ByteOrder::LeastSignificantByteFirst)?
        );
    }
    Ok(())
}

#[test]
fn ascii_aliases_whitespace_special_floats_and_trailing_values_follow_nrrd_rules() -> Result<()> {
    let directory = tempdir()?;
    for encoding in ["text", "txt"] {
        let path = directory.path().join(format!("{encoding}.nrrd"));
        let fields = [
            "type: unsigned char".to_owned(),
            "dimension: 3".to_owned(),
            "sizes: 1 1 1".to_owned(),
            format!("encoding: {encoding}"),
        ];
        let fields = fields.iter().map(String::as_str).collect::<Vec<_>>();
        write_header(&path, &fields, b"7 99")?;
        let actual = read_nrrd_stored(&path)?;
        assert_eq!(
            actual
                .samples()
                .encode(ByteOrder::LeastSignificantByteFirst)?,
            [7]
        );
    }

    let whitespace = directory.path().join("c-whitespace.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 4 1 1",
        "encoding: ascii",
    ];
    write_header(&whitespace, &fields, b"7\t8\x0b9\x0c10\r999")?;
    let decoded = read_nrrd_stored(&whitespace)?;
    assert_eq!(
        decoded
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [7, 8, 9, 10]
    );

    let floating = directory.path().join("special-floats.nrrd");
    let fields = [
        "type: float",
        "dimension: 3",
        "sizes: 3 1 1",
        "encoding: ascii",
    ];
    write_header(&floating, &fields, b"NaN -Inf +INF")?;
    let bytes = read_nrrd_stored(&floating)?
        .samples()
        .encode(ByteOrder::LeastSignificantByteFirst)?;
    assert!(f32::from_le_bytes(bytes[0..4].try_into()?).is_nan());
    assert_eq!(
        f32::from_le_bytes(bytes[4..8].try_into()?),
        f32::NEG_INFINITY
    );
    assert_eq!(f32::from_le_bytes(bytes[8..12].try_into()?), f32::INFINITY);

    let decimals = directory.path().join("decimal-floats.nrrd");
    let fields = [
        "type: float",
        "dimension: 3",
        "sizes: 3 1 1",
        "encoding: ascii",
    ];
    write_header(&decimals, &fields, b"-0.0 -3.125 1.25e+3")?;
    let bytes = read_nrrd_stored(&decimals)?
        .samples()
        .encode(ByteOrder::LeastSignificantByteFirst)?;
    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into()?), 0x8000_0000);
    assert_eq!(f32::from_le_bytes(bytes[4..8].try_into()?), -3.125);
    assert_eq!(f32::from_le_bytes(bytes[8..12].try_into()?), 1250.0);
    Ok(())
}

#[test]
fn raw_line_and_byte_skips_apply_in_order_and_byte_skip_minus_one_uses_file_end() -> Result<()> {
    let directory = tempdir()?;
    let skipped = directory.path().join("line-and-byte-skip.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "lineskip: 1",
        "byteskip: 1",
        "encoding: raw",
    ];
    write_header(&skipped, &fields, b"preamble\r\n?\x2aextra")?;
    let decoded = read_nrrd_stored(&skipped)?;
    assert_eq!(
        decoded
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [0x2a]
    );

    let from_end = directory.path().join("negative-byte-skip.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "line skip: 99",
        "byte skip: -1",
        "encoding: raw",
    ];
    write_header(&from_end, &fields, b"variable prefix\x2a")?;
    let decoded = read_nrrd_stored(&from_end)?;
    assert_eq!(
        decoded
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [0x2a]
    );
    Ok(())
}

#[test]
fn compressed_byte_skip_applies_to_decompressed_payload() -> Result<()> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let directory = tempdir()?;
    let path = directory.path().join("gzip-byte-skip.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "line skip: 1",
        "byte skip: 2",
        "encoding: gzip",
    ];
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(b"##\x2a")?;
    let compressed = encoder.finish()?;
    let mut payload = b"ignored line\n".to_vec();
    payload.extend_from_slice(&compressed);
    write_header(&path, &fields, &payload)?;
    let decoded = read_nrrd_stored(&path)?;
    assert_eq!(
        decoded
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [0x2a]
    );
    Ok(())
}

#[test]
fn concatenated_gzip_members_form_one_nrrd_payload() -> Result<()> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let directory = tempdir()?;
    let path = directory.path().join("multi-member-gzip.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 2 1 1",
        "encoding: gzip",
    ];
    let mut payload = Vec::new();
    for sample in [7_u8, 8] {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&[sample])?;
        payload.extend_from_slice(&encoder.finish()?);
    }
    write_header(&path, &fields, &payload)?;

    let decoded = read_nrrd_stored(&path)?;
    assert_eq!(
        decoded
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [7, 8]
    );
    Ok(())
}

#[test]
fn stored_reader_requires_encoding_and_rejects_invalid_skips_before_payload_use() -> Result<()> {
    let directory = tempdir()?;
    let missing = directory.path().join("missing-encoding.nrrd");
    write_header_with_version(
        &missing,
        &["type: unsigned char", "dimension: 3", "sizes: 1 1 1"],
        &[7],
        "NRRD0004",
        None,
    )?;
    assert!(matches!(
        read_nrrd_stored(&missing),
        Err(NrrdStoredReadError::MissingHeaderField { field: "encoding" })
    ));

    for (name, field, expected) in [
        ("negative-line-skip", "line skip: -1", "line"),
        ("too-negative-byte-skip", "byte skip: -2", "byte"),
    ] {
        let path = directory.path().join(format!("{name}.nrrd"));
        let fields = [
            "type: unsigned char",
            "dimension: 3",
            "sizes: 1 1 1",
            field,
            "encoding: raw",
        ];
        write_header(&path, &fields, &[])?;
        let error = read_nrrd_stored(&path).expect_err("invalid skip is rejected");
        match (expected, error) {
            ("line", NrrdStoredReadError::NegativeLineSkip { value: -1 }) => {}
            ("byte", NrrdStoredReadError::InvalidByteSkip { value: -2, .. }) => {}
            (_, other) => panic!("unexpected NRRD skip error: {other:?}"),
        }
    }

    let gzip_end_skip = directory.path().join("gzip-end-skip.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "byte skip: -1",
        "encoding: gzip",
    ];
    write_header(&gzip_end_skip, &fields, &[])?;
    assert!(matches!(
        read_nrrd_stored(&gzip_end_skip),
        Err(NrrdStoredReadError::InvalidByteSkip { value: -1, .. })
    ));

    let invalid_ascii = directory.path().join("invalid-ascii.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "encoding: ascii",
    ];
    write_header(&invalid_ascii, &fields, b"not-a-number")?;
    assert!(matches!(
        read_nrrd_stored(&invalid_ascii),
        Err(NrrdStoredReadError::InvalidAsciiSample {
            sample_index: 0,
            ..
        })
    ));

    let overflow_ascii = directory.path().join("overflow-ascii.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "encoding: ascii",
    ];
    write_header(&overflow_ascii, &fields, b"256")?;
    assert!(matches!(
        read_nrrd_stored(&overflow_ascii),
        Err(NrrdStoredReadError::InvalidAsciiSample {
            sample_index: 0,
            ..
        })
    ));

    let long_ascii = directory.path().join("long-ascii-token.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "encoding: ascii",
    ];
    let token = "1".repeat(129);
    write_header(&long_ascii, &fields, token.as_bytes())?;
    assert!(matches!(
        read_nrrd_stored(&long_ascii),
        Err(NrrdStoredReadError::AsciiTokenTooLong {
            sample_index: 0,
            maximum_bytes: 128,
        })
    ));
    Ok(())
}

#[test]
fn raw_declared_length_is_checked_before_allocating_payload_storage() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("oversized-declaration.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1000000000 1 1",
        "encoding: raw",
    ];
    write_header(&path, &fields, &[])?;
    assert!(matches!(
        read_nrrd_stored(&path),
        Err(NrrdStoredReadError::TruncatedPayload {
            expected_bytes: 1_000_000_000,
            actual_bytes: 0,
        })
    ));
    Ok(())
}
