//! General-purpose DICOM object writer.
//!
//! Converts a `DicomObjectModel` to a `dicom::object::InMemDicomObject`
//! and writes it as a valid DICOM Part 10 file.
//!
//! # Invariants
//! - Every node in the model appears exactly once in the output.
//! - Byte nodes use OB VR unconditionally.
//! - Sequence nodes produce SQ elements with undefined length.

use super::object_model::{DicomObjectModel, DicomTag};
use super::transfer_syntax::EXPLICIT_VR_LE;
use super::writer::elements::node_to_element;
use super::writer::output::write_file;
use super::writer::pixel_encoding::DICOM_SOP_CLASS_SECONDARY_CAPTURE;
use anyhow::Result;
use dicom::object::{meta::FileMetaTableBuilder, InMemDicomObject};
use std::path::Path;

/// Convert a `DicomObjectModel` to an `InMemDicomObject`.
pub fn model_to_in_mem(model: &DicomObjectModel) -> Result<InMemDicomObject> {
    let mut obj = InMemDicomObject::new_empty();
    for node in &model.nodes {
        obj.put(node_to_element(node));
    }
    Ok(obj)
}

/// Write a `DicomObjectModel` to a DICOM Part 10 file.
pub fn write_object(model: &DicomObjectModel, path: &Path) -> Result<()> {
    let obj = model_to_in_mem(model)?;
    let sop_class = model
        .get(DicomTag::new(0x0008, 0x0016))
        .and_then(|n| n.value.as_text())
        .unwrap_or(DICOM_SOP_CLASS_SECONDARY_CAPTURE);
    let sop_inst = model
        .get(DicomTag::new(0x0008, 0x0018))
        .and_then(|n| n.value.as_text())
        .unwrap_or("2.25.0");
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(sop_class)
                .media_storage_sop_instance_uid(sop_inst)
                .transfer_syntax(EXPLICIT_VR_LE),
        )
        .map_err(|e| anyhow::anyhow!("DICOM meta build failed: {e}"))?;
    write_file(path, &file_obj)
}

#[cfg(test)]
mod tests {
    use super::super::object_model::{
        DicomObjectModel, DicomObjectNode, DicomSequenceItem, DicomTag,
    };
    use super::*;
    use crate::format::dicom::writer::DicomWriteError;
    use dicom::core::Tag;
    use dicom::object::open_file;

    fn pixel_model(attributes: &[(u16, u16, u16)], payload: Vec<u8>) -> DicomObjectModel {
        let mut model = DicomObjectModel::new();
        for &(group, element, value) in attributes {
            model.insert(DicomObjectNode::with_value(
                DicomTag::new(group, element),
                "US",
                value,
            ));
        }
        model.insert(DicomObjectNode::bytes(
            DicomTag::new(0x7FE0, 0x0010),
            "OB",
            payload,
        ));
        model
    }

    #[test]
    fn test_write_object_empty_model_creates_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("empty.dcm");
        let model = DicomObjectModel::new();
        write_object(&model, &path).expect("write_object");
        assert!(path.exists(), "file must exist after write_object");
    }

    #[test]
    fn test_model_to_in_mem_text_node() {
        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x0008, 0x0060),
            "CS",
            "CT",
        ));
        let obj = model_to_in_mem(&model).expect("model_to_in_mem");
        assert_eq!(obj.iter().count(), 1);
    }

    #[test]
    fn test_model_to_in_mem_unsigned_node() {
        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::with_value(
            DicomTag::new(0x0028, 0x0100),
            "US",
            16u16,
        ));
        let obj = model_to_in_mem(&model).expect("model_to_in_mem");
        assert_eq!(obj.iter().count(), 1);
    }

    #[test]
    fn test_write_object_bytes_node_non_empty() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("bytes.dcm");
        let model = pixel_model(
            &[
                (0x0028, 0x0010, 2),
                (0x0028, 0x0011, 5),
                (0x0028, 0x0002, 1),
                (0x0028, 0x0100, 8),
                (0x0028, 0x0101, 8),
                (0x0028, 0x0102, 7),
                (0x0028, 0x0103, 0),
            ],
            vec![0; 10],
        );
        write_object(&model, &path).expect("write_object");
        let len = std::fs::metadata(&path).expect("metadata").len();
        assert!(len > 128, "file must exceed preamble size, got {len}");
    }

    #[test]
    fn test_write_object_rejects_malformed_pixel_description_before_replacing_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("malformed.dcm");
        std::fs::write(&path, b"sentinel").expect("sentinel");
        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::with_value(
            DicomTag::new(0x0028, 0x0100),
            "US",
            16u16,
        ));
        model.insert(DicomObjectNode::bytes(
            DicomTag::new(0x7FE0, 0x0010),
            "OW",
            vec![0u8; 2],
        ));
        let error = write_object(&model, &path).expect_err("malformed metadata must fail");
        assert_eq!(
            error.downcast_ref::<DicomWriteError>(),
            Some(&DicomWriteError::MissingPixelAttribute {
                attribute: "BitsStored"
            })
        );
        assert_eq!(std::fs::read(&path).expect("sentinel remains"), b"sentinel");
    }
    #[test]
    fn test_write_object_rejects_missing_rows_before_replacing_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("missing-rows.dcm");
        std::fs::write(&path, b"sentinel").expect("sentinel");
        let model = pixel_model(
            &[
                (0x0028, 0x0002, 1),
                (0x0028, 0x0011, 2),
                (0x0028, 0x0100, 8),
                (0x0028, 0x0101, 8),
                (0x0028, 0x0102, 7),
                (0x0028, 0x0103, 0),
            ],
            vec![0; 2],
        );
        let error = write_object(&model, &path).expect_err("missing Rows must fail");
        assert_eq!(
            error.downcast_ref::<DicomWriteError>(),
            Some(&DicomWriteError::MissingPixelAttribute { attribute: "Rows" })
        );
        assert_eq!(std::fs::read(&path).expect("sentinel remains"), b"sentinel");
    }

    #[test]
    fn test_write_object_counts_frames_and_samples_in_payload_length() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("payload-length.dcm");
        std::fs::write(&path, b"sentinel").expect("sentinel");
        let model = pixel_model(
            &[
                (0x0028, 0x0002, 2),
                (0x0028, 0x0008, 2),
                (0x0028, 0x0010, 2),
                (0x0028, 0x0011, 2),
                (0x0028, 0x0100, 8),
                (0x0028, 0x0101, 8),
                (0x0028, 0x0102, 7),
                (0x0028, 0x0103, 0),
            ],
            vec![0; 4],
        );
        let error = write_object(&model, &path).expect_err("short payload must fail");
        assert_eq!(
            error.downcast_ref::<DicomWriteError>(),
            Some(&DicomWriteError::PixelPayloadLengthMismatch {
                expected: 16,
                actual: 4,
            })
        );
        assert_eq!(std::fs::read(&path).expect("sentinel remains"), b"sentinel");
    }

    #[test]
    fn test_write_object_text_node_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("roundtrip.dcm");
        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x0008, 0x0060),
            "CS",
            "MR",
        ));
        write_object(&model, &path).expect("write_object");
        let obj = open_file(&path).expect("open_file");
        let val = obj
            .element(Tag(0x0008, 0x0060))
            .expect("element")
            .to_str()
            .expect("to_str");
        assert_eq!(val.trim(), "MR", "roundtrip value mismatch");
    }

    #[test]
    fn test_write_object_preserves_nested_sequence_structure() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("sequence.dcm");

        let mut item = DicomSequenceItem::new();
        item.insert(DicomObjectNode::text(
            DicomTag::new(0x0010, 0x0010),
            "PN",
            "Test^Patient",
        ));
        item.insert(DicomObjectNode::bytes(
            DicomTag::new(0x0009, 0x1001),
            "OB",
            vec![1, 2, 3, 4],
        ));

        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::sequence(
            DicomTag::new(0x0008, 0x1111),
            "SQ",
            vec![item],
        ));

        write_object(&model, &path).expect("write_object");
        let obj = open_file(&path).expect("open_file");

        let seq = obj.element(Tag(0x0008, 0x1111)).expect("sequence element");
        assert_eq!(seq.vr().to_string(), "SQ");

        let items = seq.value().items().expect("sequence items");
        assert_eq!(items.len(), 1);

        let first = &items[0];
        let pn = first
            .element(Tag(0x0010, 0x0010))
            .expect("sequence text element");
        assert_eq!(pn.to_str().expect("pn text").trim(), "Test^Patient");

        let private = first
            .element(Tag(0x0009, 0x1001))
            .expect("sequence private element");
        assert_eq!(private.vr().to_string(), "OB");
        assert_eq!(
            private.to_bytes().expect("private bytes").as_ref(),
            &[1, 2, 3, 4]
        );
    }

    #[test]
    fn test_model_to_in_mem_preserves_raw_bytes_and_sequence_node_count() {
        let mut child = DicomSequenceItem::new();
        child.insert(DicomObjectNode::text(
            DicomTag::new(0x0010, 0x0020),
            "LO",
            "PAT001",
        ));

        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::bytes(
            DicomTag::new(0x0019, 0x10AA),
            "OB",
            vec![9, 8, 7, 6],
        ));
        model.insert(DicomObjectNode::sequence(
            DicomTag::new(0x0008, 0x2222),
            "SQ",
            vec![child],
        ));

        let obj = model_to_in_mem(&model).expect("model_to_in_mem");
        assert_eq!(obj.iter().count(), 2);

        let preserved = obj.element(Tag(0x0019, 0x10AA)).expect("private bytes");
        assert_eq!(preserved.vr().to_string(), "OB");
        assert_eq!(preserved.to_bytes().expect("bytes").as_ref(), &[9, 8, 7, 6]);

        let seq = obj.element(Tag(0x0008, 0x2222)).expect("sequence");
        assert_eq!(seq.vr().to_string(), "SQ");
        assert_eq!(seq.value().items().expect("sequence items").len(), 1);
    }
}
