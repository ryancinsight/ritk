#![expect(clippy::unwrap_used, reason = "DICOM Part 10 fixture setup")]

use super::super::super::{
    load_dicom_stored_series, load_dicom_stored_series_with_budget, scan_dicom_path,
    DicomReadBudget, StoredDicomError,
};
use super::{
    write_signed_series, write_slice, write_slice_with_encoding, CalibrationFixture, PixelEncoding,
};
use ritk_codecs::{ByteOrder, SampleType};
use ritk_dicom::ParseBudget;
use ritk_image_io::{IntensityCalibration, LutOutputBits};
#[test]
fn stored_reader_preserves_unsigned_eight_bit_samples_and_valid_pad() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice_with_encoding(
        &directory.path().join("unsigned-byte.dcm"),
        1,
        "0",
        0,
        1,
        "MONOCHROME2",
        CalibrationFixture::Linear {
            slope: "1",
            intercept: "0",
            rescale_type: Some("US"),
        },
        vec![0x80, 0],
        PixelEncoding {
            rows: 1,
            columns: 1,
            bits_allocated: 8,
            bits_stored: 8,
            high_bit: 7,
            ..PixelEncoding::DEFAULT
        },
    );
    let scanned = scan_dicom_path(directory.path()).expect("scan unsigned byte series");
    let (series, _) = load_dicom_stored_series(scanned).expect("load unsigned byte series");
    let samples = &series.volumes()[0].samples();
    assert_eq!(samples.sample_type(), SampleType::U8);
    assert_eq!(
        samples
            .encode(ByteOrder::LeastSignificantByteFirst)
            .expect("encode stored byte"),
        [0x80]
    );
}

#[test]
fn stored_reader_maps_allocated_width_to_exact_signed_and_unsigned_samples() {
    struct Case {
        representation: u16,
        bits: u16,
        input: &'static [u8],
        sample_type: SampleType,
        expected: &'static [u8],
    }

    let cases = [
        Case {
            representation: 1,
            bits: 8,
            input: &[0xfe, 0],
            sample_type: SampleType::I8,
            expected: &[0xfe],
        },
        Case {
            representation: 0,
            bits: 16,
            input: &[0x34, 0x12],
            sample_type: SampleType::U16,
            expected: &[0x34, 0x12],
        },
        Case {
            representation: 1,
            bits: 24,
            input: &[0xfe, 0xff, 0xff, 0],
            sample_type: SampleType::I32,
            expected: &[0xfe, 0xff, 0xff, 0xff],
        },
        Case {
            representation: 0,
            bits: 24,
            input: &[0x56, 0x34, 0x12, 0],
            sample_type: SampleType::U32,
            expected: &[0x56, 0x34, 0x12, 0],
        },
        Case {
            representation: 0,
            bits: 32,
            input: &[0x78, 0x56, 0x34, 0x12],
            sample_type: SampleType::U32,
            expected: &[0x78, 0x56, 0x34, 0x12],
        },
        Case {
            representation: 1,
            bits: 32,
            input: &[0xfe, 0xff, 0xff, 0xff],
            sample_type: SampleType::I32,
            expected: &[0xfe, 0xff, 0xff, 0xff],
        },
    ];

    for (index, case) in cases.into_iter().enumerate() {
        let directory = tempfile::tempdir().expect("temporary series directory");
        write_slice_with_encoding(
            &directory.path().join("integer-width.dcm"),
            u16::try_from(
                index
                    .checked_add(1)
                    .expect("fixture index increment fits usize"),
            )
            .expect("fixture index fits instance number"),
            "0",
            case.representation,
            1,
            "MONOCHROME2",
            CalibrationFixture::Linear {
                slope: "1",
                intercept: "0",
                rescale_type: Some("US"),
            },
            case.input.to_vec(),
            PixelEncoding {
                rows: 1,
                columns: 1,
                bits_allocated: case.bits,
                bits_stored: case.bits,
                high_bit: case.bits.checked_sub(1).expect("fixture bits are nonzero"),
                ..PixelEncoding::DEFAULT
            },
        );
        let scanned = scan_dicom_path(directory.path()).expect("scan allocated-width series");
        let (series, _) = load_dicom_stored_series(scanned).expect("load allocated-width series");
        let samples = &series.volumes()[0].samples();
        assert_eq!(samples.sample_type(), case.sample_type);
        assert_eq!(
            samples
                .encode(ByteOrder::LeastSignificantByteFirst)
                .expect("encode fixed-width stored sample"),
            case.expected
        );
    }
}

#[test]
fn stored_reader_rejects_invalid_high_bit_and_missing_rescale_type() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice_with_encoding(
        &directory.path().join("invalid-high-bit.dcm"),
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
            high_bit: 10,
            ..PixelEncoding::DEFAULT
        },
    );
    let scanned = scan_dicom_path(directory.path()).expect("scan invalid high-bit series");
    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::InvalidTag {
            tag: "HighBit (0028,0102)"
        })
    ));

    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice(
        &directory.path().join("missing-rescale-type.dcm"),
        1,
        "0",
        0,
        1,
        "MONOCHROME2",
        CalibrationFixture::Linear {
            slope: "1",
            intercept: "0",
            rescale_type: None,
        },
        vec![0, 0, 0, 0],
    );
    let scanned = scan_dicom_path(directory.path()).expect("scan missing rescale type");
    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::MissingTag {
            tag: "RescaleType (0028,1054)"
        })
    ));
}

#[test]
fn stored_reader_rejects_truncated_pixels_and_nonzero_padding() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice(
        &directory.path().join("truncated.dcm"),
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
        vec![0, 0],
    );
    let scanned = scan_dicom_path(directory.path()).expect("scan truncated pixel data");
    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::PixelDataLength {
            actual: 2,
            expected: 4
        })
    ));

    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice_with_encoding(
        &directory.path().join("nonzero-padding.dcm"),
        1,
        "0",
        0,
        1,
        "MONOCHROME2",
        CalibrationFixture::Linear {
            slope: "1",
            intercept: "0",
            rescale_type: Some("US"),
        },
        vec![0x80, 1],
        PixelEncoding {
            rows: 1,
            columns: 1,
            bits_allocated: 8,
            bits_stored: 8,
            high_bit: 7,
        },
    );
    let scanned = scan_dicom_path(directory.path()).expect("scan nonzero pixel padding");
    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::PixelDataLength {
            actual: 2,
            expected: 1
        })
    ));
}

#[test]
fn stored_reader_requires_retained_bytes_and_enforces_decoded_budget() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_signed_series(directory.path());
    let mut scanned = scan_dicom_path(directory.path()).expect("scan signed series");
    scanned.metadata.slices[0].part10_bytes = None;
    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::MissingRetainedBytes)
    ));

    let scanned = scan_dicom_path(directory.path()).expect("scan signed series again");
    let budget = DicomReadBudget::try_new(ParseBudget::DEFAULT, 1024 * 1024, 1)
        .expect("nonzero workflow budget");
    assert!(matches!(
        load_dicom_stored_series_with_budget(scanned, &budget),
        Err(StoredDicomError::Budget(_))
    ));
}

#[test]
fn stored_reader_budgets_metadata_for_many_small_slices_before_reserving() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    for index in 0..3_u16 {
        write_slice(
            &directory.path().join(format!("slice-{index}.dcm")),
            index.checked_add(1).expect("fixture instance number fits"),
            &index.to_string(),
            1,
            1,
            "MONOCHROME2",
            CalibrationFixture::Linear {
                slope: "1",
                intercept: "0",
                rescale_type: Some("HU"),
            },
            vec![0, 0, 0, 0],
        );
    }
    let scanned = scan_dicom_path(directory.path()).expect("scan small-slice series");
    let largest_instance = std::fs::read_dir(directory.path())
        .expect("read fixture directory")
        .map(|entry| {
            entry
                .expect("fixture directory entry")
                .metadata()
                .expect("fixture metadata")
                .len()
        })
        .max()
        .expect("at least one fixture");
    let largest_instance = usize::try_from(largest_instance).expect("fixture size fits usize");
    let decoded_samples = 3_usize * 2 * std::mem::size_of::<i16>();
    let frame_bytes = 2_usize * std::mem::size_of::<i16>();
    let prior_workspace_bound = largest_instance + decoded_samples + frame_bytes * 2;
    let budget = DicomReadBudget::try_new(ParseBudget::DEFAULT, 1024 * 1024, prior_workspace_bound)
        .expect("positive retained and decoded budgets");

    assert!(matches!(
        load_dicom_stored_series_with_budget(scanned, &budget),
        Err(StoredDicomError::Budget(_))
    ));
}

#[test]
fn stored_reader_retains_signed_modality_lut_and_exact_pixels() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice(
        &directory.path().join("signed-lut.dcm"),
        1,
        "0",
        1,
        1,
        "MONOCHROME2",
        CalibrationFixture::ModalityLookup {
            entry_count: 3,
            first_mapped_value: -2,
            output_bits: 16,
            entries: &[10, 20, 30],
            signed_descriptor: true,
            unit: Some("HU"),
        },
        vec![0xfe, 0xff, 0xff, 0xff],
    );
    let scanned = scan_dicom_path(directory.path()).expect("scan signed LUT series");
    let (series, _) = load_dicom_stored_series(scanned).expect("load signed LUT series");
    let volume = &series.volumes()[0];
    assert_eq!(
        volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)
            .expect("stored samples remain signed words"),
        [0xfe, 0xff, 0xff, 0xff]
    );
    let IntensityCalibration::ModalityLookup(table) = volume.calibration() else {
        panic!("Modality LUT must remain typed calibration");
    };
    assert_eq!(
        volume.intensity_unit().map(|unit| unit.as_str()),
        Some("HU")
    );
    assert_eq!(table.first_mapped_value(), -2);
    assert_eq!(table.entries(), [10, 20, 30]);
    assert_eq!(table.output_bits(), LutOutputBits::Sixteen);
    assert_eq!(table.map(-2), 10);
    assert_eq!(table.map(-1), 20);
    assert_eq!(table.map(1), 30);
}

#[test]
fn stored_reader_unpacks_eight_bit_modality_lut_words() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice(
        &directory.path().join("eight-bit-lut.dcm"),
        1,
        "0",
        0,
        1,
        "MONOCHROME2",
        CalibrationFixture::ModalityLookup {
            entry_count: 3,
            first_mapped_value: 0,
            output_bits: 8,
            entries: &[0x0201, 0x0080],
            signed_descriptor: false,
            unit: Some("US"),
        },
        vec![0, 0, 1, 0],
    );
    let scanned = scan_dicom_path(directory.path()).expect("scan eight-bit LUT series");
    let (series, _) = load_dicom_stored_series(scanned).expect("load eight-bit LUT series");
    let IntensityCalibration::ModalityLookup(table) = series.volumes()[0].calibration() else {
        panic!("Modality LUT must remain typed calibration");
    };
    assert_eq!(table.entries(), [1, 2, 128]);
    assert_eq!(table.output_bits(), LutOutputBits::Eight);
    assert_eq!(
        series.volumes()[0]
            .intensity_unit()
            .map(|unit| unit.as_str()),
        Some("US")
    );
}

#[test]
fn stored_reader_uses_retained_patient_geometry_instead_of_synthesized_tilt() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    for (instance, z) in [(1_u16, "0"), (2, "2")] {
        write_slice_with_encoding(
            &directory.path().join(format!("slice-{instance}.dcm")),
            instance,
            z,
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
                rows: 2,
                columns: 1,
                orientation: Some("1\\0\\0\\0\\1\\0"),
                gantry_tilt: Some("20"),
                ..PixelEncoding::DEFAULT
            },
        );
    }
    let scanned = scan_dicom_path(directory.path()).expect("scanner handles gantry tilt");
    assert_ne!(
        scanned.metadata.slices[0].image_orientation_patient,
        Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    );
    let (series, _) = load_dicom_stored_series(scanned).expect("load retained stored geometry");
    assert_eq!(
        series.volumes()[0].metadata().spacing().to_array(),
        [2.0, 0.5, 0.25]
    );
    let CoordinateMap::SliceSeries(coordinates) = series.volumes()[0].coordinate_map() else {
        panic!("stored DICOM keeps per-slice patient transforms");
    };
    assert_eq!(
        coordinates.world_from_index(0.0, 1.0, 0.0),
        [10.0, 20.5, 0.0]
    );
}

#[test]
fn stored_reader_rejects_missing_raw_orientation_even_if_scan_synthesizes_it() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice_with_encoding(
        &directory.path().join("missing-orientation.dcm"),
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
            orientation: None,
            gantry_tilt: Some("20"),
            ..PixelEncoding::DEFAULT
        },
    );
    let scanned =
        scan_dicom_path(directory.path()).expect("scanner synthesizes a scan orientation");
    assert!(scanned.metadata.slices[0]
        .image_orientation_patient
        .is_some());

    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::InvalidGeometry {
            field: "ImageOrientationPatient (0020,0037)"
        })
    ));
}
