//! Tests for `NativeBackend`.
//!
//! Extracted to keep the 500-line structural limit.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::super::*;
use crate::backend::{parse_bytes_with, parse_file_with, write_bytes_with, write_file_with};
use crate::dataset::SmallVec;
use crate::dataset::{DataElement, PrimitiveValue, VR};
use crate::dataset::{Dataset, FileMetaBuilder};

/// `NativeBackend::parse_bytes` must round-trip an in-memory DICOM object.
///
/// Analytical basis:
/// - `Dataset` with File Meta is written to a temp file.
/// - Raw bytes are read back and parsed via `parse_bytes_with::<NativeBackend>`.
/// - The parsed object must contain the same PatientName value that was written,
///   proving `parse_bytes` constructs a semantically equivalent object from
///   Part 10 bytes without file I/O.

#[test]
fn native_backend_parse_bytes_round_trips_in_memory_object() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("test_parse_bytes.dcm");

    let mut obj = Dataset::new();
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.2001"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Test^Patient"),
    ));

    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("2.25.2001")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("DICOM file must be written");

    let bytes = std::fs::read(&path).expect("temp file must be readable");
    let parsed = parse_bytes_with::<NativeBackend>(&bytes)
        .expect("parse_bytes must succeed on valid Part 10 bytes");

    let patient_name = parsed
        .element(DicomTag::new(0x0010, 0x0010))
        .expect("PatientName must be present")
        .value()
        .to_str()
        .expect("PatientName must be a string")
        .trim_end_matches(['\0', ' '])
        .to_owned();

    assert_eq!(patient_name, "Test^Patient");
}

/// `NativeBackend::parse_bytes` must reject garbage input.
///
/// Analytical basis: bytes without a valid DICOM Part 10 preamble or
/// DICM magic cannot be parsed. The function must return `Err`, not panic.
#[test]
fn native_backend_parse_bytes_rejects_garbage_input() {
    let result = parse_bytes_with::<NativeBackend>(&[0xDE, 0xAD, 0xBE, 0xEF]);
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.to_lowercase().contains("dicom")
            || msg.to_lowercase().contains("parse")
            || msg.to_lowercase().contains("failed"),
        "error must describe parse failure, got: {msg}"
    );
}

#[test]
fn native_backend_writes_bytes_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let path = dir.path().join("write_bytes_src.dcm");

    let mut obj = Dataset::new();
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.3001"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Encoder^Roundtrip"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(SmallVec::from_vec(vec![42_u16])),
    ));

    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("2.25.3001")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&path)
    .expect("source DICOM file must be written");

    let parsed = parse_file_with::<NativeBackend, _>(&path).expect("parse must succeed");
    let bytes = write_bytes_with::<NativeBackend>(&parsed).expect("write_bytes must succeed");
    let reparsed = parse_bytes_with::<NativeBackend>(&bytes).expect("re-parse must succeed");

    let patient_name = reparsed
        .element(DicomTag::new(0x0010, 0x0010))
        .expect("PatientName must be present")
        .value()
        .to_str()
        .expect("PatientName must be a string")
        .trim_end_matches(['\0', ' '])
        .to_owned();
    assert_eq!(patient_name, "Encoder^Roundtrip");
}

#[test]
fn native_backend_writes_file_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let src = dir.path().join("write_file_src.dcm");
    let dst = dir.path().join("write_file_dst.dcm");

    let mut obj = Dataset::new();
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.3002"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0010, 0x0010),
        VR::PN,
        PrimitiveValue::from("Encoder^File"),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::from(1_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::from(0_u16),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(SmallVec::from_vec(vec![7_u16])),
    ));

    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("2.25.3002")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("file meta must be valid")
    .write_to_file(&src)
    .expect("source DICOM file must be written");

    let parsed = parse_file_with::<NativeBackend, _>(&src).expect("parse must succeed");
    write_file_with::<NativeBackend, _>(&dst, &parsed).expect("write_file must succeed");

    let reparsed = parse_file_with::<NativeBackend, _>(&dst).expect("re-parse must succeed");
    let patient_name = reparsed
        .element(DicomTag::new(0x0010, 0x0010))
        .expect("PatientName must be present")
        .value()
        .to_str()
        .expect("PatientName must be a string")
        .trim_end_matches(['\0', ' '])
        .to_owned();
    assert_eq!(patient_name, "Encoder^File");
}
