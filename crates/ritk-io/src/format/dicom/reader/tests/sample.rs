//! Series ambiguity regression with unequal acquisition populations.
#![expect(clippy::unwrap_used, reason = "test fixture setup")]
use super::super::scan::scan_dicom_directory;
use super::support::*;
#[test]
fn test_scan_directory_rejects_ambiguous_series_regardless_of_population() {
    use dicom::core::smallvec::SmallVec;

    let dir = tempfile::tempdir().unwrap();

    // Write a minimal CT DICOM file with the given SeriesInstanceUID, instance
    // number, and z-position (used for IPP and sort ordering).
    let write_ct_slice = |path: &std::path::Path, series_uid: &str, instance: u32, z: f64| {
        let sop_instance_uid = format!("{}.{}", series_uid, instance);
        let mut obj = InMemDicomObject::new_empty();
        // SOP class: CT Image Storage
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from(sop_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from("CT"),
        ));
        // SeriesInstanceUID (0020,000E) — the tag under test.
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from(series_uid),
        ));
        // StudyInstanceUID
        obj.put(DataElement::new(
            Tag(0x0020, 0x000D),
            VR::UI,
            PrimitiveValue::from("1.2.3.4.5"),
        ));
        // InstanceNumber
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from(format!("{}", instance).as_str()),
        ));
        // ImagePositionPatient (0020,0032): 0.0\0.0\z
        obj.put(DataElement::new(
            Tag(0x0020, 0x0032),
            VR::DS,
            PrimitiveValue::from(format!("0.0\\0.0\\{:.1}", z).as_str()),
        ));
        // ImageOrientationPatient (0020,0037): axial [1,0,0,0,1,0]
        obj.put(DataElement::new(
            Tag(0x0020, 0x0037),
            VR::DS,
            PrimitiveValue::from("1.0\\0.0\\0.0\\0.0\\1.0\\0.0"),
        ));
        // Rows / Cols: 8×8
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(8_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(8_u16),
        ));
        // BitsAllocated / BitsStored / HighBit / PixelRepresentation
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(15_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        // SamplesPerPixel / PhotometricInterpretation
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from("MONOCHROME2"),
        ));
        // PixelSpacing: 1.0\1.0
        obj.put(DataElement::new(
            Tag(0x0028, 0x0030),
            VR::DS,
            PrimitiveValue::from("1.0\\1.0"),
        ));
        // PixelData: 8×8 × 2 bytes = 128 bytes of zeroes
        let pixel_bytes: Vec<u8> = vec![0u8; 8 * 8 * 2];
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                    .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .expect("meta build must not fail");
        file_obj.write_to_file(path).expect("write must not fail");
    };

    // Series A: 3 slices — the most-populated series.
    write_ct_slice(&dir.path().join("A1.dcm"), "2.25.71001", 1, 0.0);
    write_ct_slice(&dir.path().join("A2.dcm"), "2.25.71001", 2, 1.0);
    write_ct_slice(&dir.path().join("A3.dcm"), "2.25.71001", 3, 2.0);
    // A directory is not an acquisition identity, even with unequal counts.
    write_ct_slice(&dir.path().join("B1.dcm"), "2.25.71002", 1, 5.0);

    // Population is not a selection instruction (RITK-SNAP-OPEN-001).
    let error = scan_dicom_directory(dir.path()).expect_err("multiple UIDs must reject");
    assert!(error.to_string().contains("ambiguous DICOM input"));
    let selected = super::super::scan::scan_dicom_path(dir.path().join("B1.dcm"))
        .expect("selected minority acquisition");
    assert_eq!(
        selected.metadata.series_instance_uid.as_deref(),
        Some("2.25.71002")
    );
    assert_eq!(selected.num_slices, 1);
    assert_eq!(selected.metadata.origin, [0.0, 0.0, 5.0]);
}
