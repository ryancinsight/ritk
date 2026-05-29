#![allow(unused_imports)]

use super::super::geometry::{
    analyze_slice_spacing, dot_3d, normalize_3d, resample_frames_linear, slice_normal_from_iop,
};
use super::super::loader::{
    load_dicom_series, load_dicom_series_with_metadata, load_from_series, read_dicom_series,
    read_dicom_series_with_metadata,
};
use super::super::pixel::{decode_pixel_bytes, read_slice_pixels};
use super::super::scan::scan_dicom_directory;
use super::super::types::{
    DicomReadMetadata, DicomSeriesInfo, DicomSliceMetadata, PatientPosition,
};
use super::super::utils::is_likely_dicom_file;
use super::support::*;
use crate::format::dicom::{
    DicomObjectNode, DicomPreservationSet, DicomPreservedElement, DicomTag, DicomValue,
};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_dicom::TransferSyntaxKind;
#[test]
fn test_scan_empty_directory_errors() {
    let temp = tempfile::tempdir().unwrap();
    let err = scan_dicom_directory(temp.path()).unwrap_err();
    assert!(err.to_string().contains("no DICOM files"));
}

#[test]
fn test_scan_non_directory_errors() {
    let temp = tempfile::NamedTempFile::new().unwrap();
    let err = scan_dicom_directory(temp.path()).unwrap_err();
    assert!(err.to_string().contains("not a directory"));
}

#[test]
fn test_scan_all_non_image_sop_returns_error_with_rejected_uids() {
    let temp = tempfile::tempdir().unwrap();

    // RT Structure Set
    write_stub_dicom(
        &temp.path().join("rtstruct.dcm"),
        "1.2.840.10008.5.1.4.1.1.481.3",
        "2.25.10001",
    );
    // Basic Text SR
    write_stub_dicom(
        &temp.path().join("sr.dcm"),
        "1.2.840.10008.5.1.4.1.1.88.11",
        "2.25.10002",
    );
    // Encapsulated PDF
    write_stub_dicom(
        &temp.path().join("pdf.dcm"),
        "1.2.840.10008.5.1.4.1.1.104.1",
        "2.25.10003",
    );

    let result = scan_dicom_directory(temp.path());
    assert!(result.is_err(), "all-non-image directory must return Err");

    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("none are image-bearing SOP classes"),
        "error must state none are image-bearing; got: {msg}"
    );
    // At least one of the three non-image UIDs must appear in the message.
    assert!(
        msg.contains("1.2.840.10008.5.1.4.1.1.481.3")
            || msg.contains("1.2.840.10008.5.1.4.1.1.88.11")
            || msg.contains("1.2.840.10008.5.1.4.1.1.104.1"),
        "error must list at least one rejected SOP UID; got: {msg}"
    );
}

#[test]
fn test_scan_mixed_non_image_and_ct_retains_image_slice() {
    let temp = tempfile::tempdir().unwrap();

    // Non-image: RT Structure Set
    write_stub_dicom(
        &temp.path().join("rtstruct.dcm"),
        "1.2.840.10008.5.1.4.1.1.481.3",
        "2.25.20001",
    );

    // Image-bearing: CT Image Storage -- include Rows/Cols so metadata is populated
    {
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.20002"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(4_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(4_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from("1"),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                    .media_storage_sop_instance_uid("2.25.20002")
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .unwrap();
        file_obj.write_to_file(temp.path().join("ct.dcm")).unwrap();
    }

    let result = scan_dicom_directory(temp.path());
    assert!(
        result.is_ok(),
        "scan must succeed when at least one image-bearing SOP exists; err={:?}",
        result.err()
    );

    let info = result.unwrap();
    assert_eq!(
        info.num_slices, 1,
        "only the CT slice must survive filtering; got {}",
        info.num_slices
    );
    assert_eq!(
        info.metadata.slices[0].sop_class_uid.as_deref(),
        Some("1.2.840.10008.5.1.4.1.1.2"),
        "retained slice must carry CT Image Storage SOP UID"
    );
}

#[test]
fn test_scan_rt_plan_and_waveform_returns_error_with_two_uids() {
    let temp = tempfile::tempdir().unwrap();

    // RT Plan
    write_stub_dicom(
        &temp.path().join("rtplan.dcm"),
        "1.2.840.10008.5.1.4.1.1.481.5",
        "2.25.30001",
    );
    // 12-lead ECG Waveform
    write_stub_dicom(
        &temp.path().join("ecg.dcm"),
        "1.2.840.10008.5.1.4.1.1.9.1.1",
        "2.25.30002",
    );

    let result = scan_dicom_directory(temp.path());
    assert!(result.is_err(), "all-non-image directory must return Err");

    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("1.2.840.10008.5.1.4.1.1.481.5"),
        "error must list RT Plan UID; got: {msg}"
    );
    assert!(
        msg.contains("1.2.840.10008.5.1.4.1.1.9.1.1"),
        "error must list ECG Waveform UID; got: {msg}"
    );
}

#[test]
fn test_scan_private_sequence_is_preserved_in_object_model() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("private_sequence.dcm");

    let mut nested_item = InMemDicomObject::new_empty();
    nested_item.put(DataElement::new(
        Tag(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Test^Patient"),
    ));
    nested_item.put(DataElement::new(
        Tag(0x0009, 0x1001),
        VR::OB,
        PrimitiveValue::U8(vec![1, 2, 3, 4].into()),
    ));

    let seq = dicom::core::value::DataSetSequence::new(
        vec![nested_item],
        dicom::core::header::Length::UNDEFINED,
    );
    let seq_value = dicom::core::value::Value::from(seq);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.40001"),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(4_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(4_u16),
    ));
    obj.put(DataElement::new(
        Tag(0x0020, 0x0013),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    obj.put(DataElement::new(
        Tag(0x0009, 0x0010),
        VR::LO,
        PrimitiveValue::from("RITK"),
    ));
    obj.put(DataElement::new(Tag(0x0009, 0x1000), VR::SQ, seq_value));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                .media_storage_sop_instance_uid("2.25.40001")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .unwrap();
    file_obj.write_to_file(&path).unwrap();

    let info = scan_dicom_directory(temp.path()).unwrap();
    let preserved = info
        .metadata
        .slices
        .first()
        .and_then(|s| s.preservation.object.get(DicomTag::new(0x0009, 0x1000)))
        .expect("private SQ must be preserved");

    assert!(preserved.is_sequence(), "private SQ must remain a sequence");
    let items = match &preserved.value {
        DicomValue::Sequence(items) => items,
        _ => panic!("private SQ must decode as DicomValue::Sequence"),
    };
    assert_eq!(items.len(), 1);
    let first = &items[0];
    assert_eq!(
        first
            .get(DicomTag::new(0x0010, 0x0010))
            .and_then(|n| n.value.as_text())
            .map(str::trim),
        Some("Test^Patient")
    );
    let raw = first
        .get(DicomTag::new(0x0009, 0x1001))
        .expect("private OB must be preserved");
    assert!(raw.value.is_bytes(), "private OB must remain raw bytes");
    assert_eq!(
        match &raw.value {
            DicomValue::Bytes(bytes) => bytes.as_slice(),
            _ => panic!("private OB must decode as DicomValue::Bytes"),
        },
        &[1, 2, 3, 4]
    );
}
