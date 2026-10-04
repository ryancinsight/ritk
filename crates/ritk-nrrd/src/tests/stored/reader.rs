//! Stored NRRD reader tests.

use super::*;

#[test]
fn stored_reader_preserves_all_types_endianness_geometry_and_float_bits() -> Result<()> {
    let directory = tempdir()?;
    let fields = [
        "dimension: 3",
        "sizes: 2 1 1",
        "space: left-posterior-superior",
        "space directions: (0.5,0,0) (0,1.5,0) (0,0,2)",
        "space origin: (-11,7.5,3.25)",
    ];
    for (sample_type, element_type, samples) in sample_cases() {
        for byte_order in [
            ByteOrder::LeastSignificantByteFirst,
            ByteOrder::MostSignificantByteFirst,
        ] {
            let path = directory
                .path()
                .join(format!("{element_type}-{byte_order:?}.nrrd"));
            let bytes = samples.encode(byte_order)?;
            let endian = match byte_order {
                ByteOrder::LeastSignificantByteFirst => "little",
                ByteOrder::MostSignificantByteFirst => "big",
            };
            let mut header = vec![format!("type: {element_type}")];
            header.extend(fields.iter().map(ToString::to_string));
            header.push(format!("endian: {endian}"));
            header.push("encoding: raw".to_string());
            let header_fields = header.iter().map(String::as_str).collect::<Vec<_>>();
            write_header(&path, &header_fields, &bytes)?;

            let volume = read_nrrd_stored(&path)?;
            assert_eq!(volume.shape(), [1, 1, 2]);
            assert_eq!(volume.samples().sample_type(), sample_type);
            assert_eq!(volume.samples().len(), 2);
            assert_eq!(volume.samples().encode(byte_order)?, bytes);
            assert_eq!(volume.metadata().origin().to_array(), [-11.0, 7.5, 3.25]);
            assert_eq!(volume.metadata().spacing().to_array(), [2.0, 1.5, 0.5]);
        }
    }
    Ok(())
}

#[test]
fn custom_key_value_names_cannot_replace_structural_fields() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("field-collision.nrrd");
    write_header(
        &path,
        &[
            "type: unsigned char",
            "dimension: 3",
            "sizes: 2 1 1",
            "type:=signed char",
            "sizes:=1 1 1",
        ],
        &[7, 8],
    )?;

    let map_error = crate::read_nrrd_header_map(&path)
        .expect_err("a flat map cannot represent colliding namespaces");
    assert!(matches!(
        map_error.downcast_ref::<crate::NrrdHeaderError>(),
        Some(crate::NrrdHeaderError::FieldKeyValueCollision { key })
            if key == "sizes" || key == "type"
    ));

    let volume = read_nrrd_stored(&path)?;

    assert_eq!(volume.shape(), [1, 1, 2]);
    assert_eq!(volume.samples().sample_type(), SampleType::U8);
    assert_eq!(
        volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)?,
        [7, 8]
    );
    Ok(())
}
