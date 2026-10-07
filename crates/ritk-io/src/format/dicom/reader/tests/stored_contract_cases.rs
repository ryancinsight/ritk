use super::{
    load_dicom_stored_series, scan_dicom_path, write_slice, write_slice_with_encoding,
    CalibrationFixture, PixelEncoding, StoredDicomError,
};
use ritk_codecs::ByteOrder;
use ritk_image_io::IntensityCalibration;

#[test]
fn stored_reader_accepts_unsigned_lut_count_bits_in_signed_descriptor() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    let packed_entries = vec![0x0201; 16_384];
    write_slice_with_encoding(
        &directory.path().join("wide-signed-descriptor.dcm"),
        1,
        "0",
        1,
        1,
        "MONOCHROME2",
        CalibrationFixture::ModalityLookup {
            entry_count: 32_768,
            first_mapped_value: 0,
            output_bits: 8,
            entries: &packed_entries,
            signed_descriptor: true,
            unit: Some("HU"),
        },
        vec![0, 0, 1, 0],
        PixelEncoding::DEFAULT,
    );

    let scanned = scan_dicom_path(directory.path()).expect("scan signed descriptor LUT");
    let (series, _) = load_dicom_stored_series(scanned).expect("load unsigned count bits");
    let volume = &series.volumes()[0];
    assert_eq!(
        volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)
            .expect("stored sample bytes"),
        [0, 0, 1, 0]
    );
    let IntensityCalibration::ModalityLookup(table) = volume.calibration() else {
        panic!("the descriptor must create a modality LUT");
    };
    assert_eq!(table.entries().len(), 32_768);
    assert_eq!(table.map(0), 1);
    assert_eq!(table.map(32_767), 2);
}

#[test]
fn stored_reader_checks_raw_direction_cosines_and_single_slice_spacing() {
    for orientation in ["2\\0\\0\\0\\1\\0", "1\\0\\0\\0.70710678\\0.70710678\\0"] {
        let directory = tempfile::tempdir().expect("temporary series directory");
        write_slice_with_encoding(
            &directory.path().join("invalid-orientation.dcm"),
            1,
            "0",
            0,
            1,
            "MONOCHROME2",
            CalibrationFixture::Linear {
                slope: "1",
                intercept: "0",
                rescale_type: Some("HU"),
            },
            vec![0, 0, 0, 0],
            PixelEncoding {
                orientation: Some(orientation),
                ..PixelEncoding::DEFAULT
            },
        );
        let scanned = scan_dicom_path(directory.path()).expect("scan orientation fixture");
        assert!(matches!(
            load_dicom_stored_series(scanned),
            Err(StoredDicomError::InvalidGeometry {
                field: "ImageOrientationPatient (0020,0037) is not orthonormal"
            })
        ));
    }

    for (spacing_between_slices, slice_thickness, expected) in
        [(Some("3.25"), Some("2.5"), 3.25), (None, Some("2.5"), 2.5)]
    {
        let directory = tempfile::tempdir().expect("temporary series directory");
        write_slice_with_encoding(
            &directory.path().join("single-slice.dcm"),
            1,
            "0",
            0,
            1,
            "MONOCHROME2",
            CalibrationFixture::Linear {
                slope: "1",
                intercept: "0",
                rescale_type: Some("HU"),
            },
            vec![0, 0, 0, 0],
            PixelEncoding {
                slice_thickness,
                spacing_between_slices,
                ..PixelEncoding::DEFAULT
            },
        );
        let mut scanned = scan_dicom_path(directory.path()).expect("scan single slice");
        scanned.metadata.spacing[0] = 999.0;
        let (series, _) = load_dicom_stored_series(scanned).expect("load raw single-slice scale");
        assert_eq!(series.volumes()[0].metadata().spacing()[0], expected);
    }

    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice_with_encoding(
        &directory.path().join("missing-depth-spacing.dcm"),
        1,
        "0",
        0,
        1,
        "MONOCHROME2",
        CalibrationFixture::Linear {
            slope: "1",
            intercept: "0",
            rescale_type: Some("HU"),
        },
        vec![0, 0, 0, 0],
        PixelEncoding {
            slice_thickness: None,
            spacing_between_slices: None,
            ..PixelEncoding::DEFAULT
        },
    );
    let scanned = scan_dicom_path(directory.path()).expect("scan missing raw depth scale");
    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::InvalidGeometry {
            field: "single-slice series requires raw SpacingBetweenSlices or SliceThickness"
        })
    ));
}

#[test]
fn stored_reader_rejects_invalid_signed_lut_descriptors_and_missing_type() {
    for (name, entry_count, output_bits, unit) in [
        ("zero-count-data-length", 0, 16, Some("HU")),
        ("negative-precision", 3, -1, Some("HU")),
        ("missing-type", 3, 16, None),
    ] {
        let directory = tempfile::tempdir().expect("temporary series directory");
        write_slice(
            &directory.path().join(format!("{name}.dcm")),
            1,
            "0",
            1,
            1,
            "MONOCHROME2",
            CalibrationFixture::ModalityLookup {
                entry_count,
                first_mapped_value: -2,
                output_bits,
                entries: &[10, 20, 30],
                signed_descriptor: true,
                unit,
            },
            vec![0, 0, 0, 0],
        );
        let scanned = scan_dicom_path(directory.path()).expect("scan signed LUT descriptor");
        assert!(matches!(
            load_dicom_stored_series(scanned),
            Err(StoredDicomError::InvalidModalityLookupTable { .. })
        ));
    }
}
