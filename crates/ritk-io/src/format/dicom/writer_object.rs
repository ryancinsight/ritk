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
use dicom::object::{InMemDicomObject, meta::FileMetaTableBuilder};
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

    fn pixel_model(
        rows: Option<u16>,
        columns: Option<u16>,
        samples: u16,
        frames: Option<u16>,
        payload: Vec<u8>,
    ) -> DicomObjectModel {
        let mut model = DicomObjectModel::new();
        for (element, value) in [
            (0x0002, samples),
            (0x0100, 8),
            (0x0101, 8),
            (0x0102, 7),
            (0x0103, 0),
        ] {
            model.insert(DicomObjectNode::with_value(
                DicomTag::new(0x0028, element),
                "US",
                value,
            ));
        }
        for (element, value) in [(0x0010, rows), (0x0011, columns), (0x0008, frames)] {
            if let Some(value) = value {
                model.insert(DicomObjectNode::with_value(
                    DicomTag::new(0x0028, element),
                    "US",
                    value,
                ));
            }
        }
        model.insert(DicomObjectNode::bytes(
            DicomTag::new(0x7FE0, 0x0010),
            "OB",
            payload,
        ));
        model
    }

    fn assert_rejected(model: DicomObjectModel, expected: DicomWriteError) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("rejected.dcm");
        std::fs::write(&path, b"sentinel").expect("sentinel");
        let error = write_object(&model, &path).expect_err("invalid pixel model must fail");
        assert_eq!(error.downcast_ref::<DicomWriteError>(), Some(&expected));
        assert_eq!(std::fs::read(&path).expect("sentinel remains"), b"sentinel");
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
        let model = pixel_model(Some(1), Some(1), 1, Some(1), vec![7]);
        write_object(&model, &path).expect("write_object");
        let obj = open_file(&path).expect("open_file");
        let value = |tag| {
            obj.element(tag)
                .expect("tag")
                .to_int::<u16>()
                .expect("value")
        };
        assert_eq!(value(Tag(0x0028, 0x0010)), 1);
        assert_eq!(
            obj.element(Tag(0x7FE0, 0x0010))
                .expect("PixelData")
                .to_bytes()
                .expect("pixel bytes"),
            vec![7, 0]
        );
    }

    #[test]
    fn test_write_object_rejects_malformed_pixel_description_before_replacing_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("malformed.dcm");
        std::fs::write(&path, b"sentinel").expect("sentinel");
        let model = pixel_model(Some(1), Some(1), 1, None, vec![0]);
        let error = write_object(&model, &path).expect_err("missing frames must fail");
        assert_eq!(
            error.downcast_ref::<DicomWriteError>(),
            Some(&DicomWriteError::MissingPixelAttribute {
                attribute: "NumberOfFrames"
            })
        );
        assert_eq!(std::fs::read(&path).expect("sentinel remains"), b"sentinel");
    }
    #[test]
    fn test_write_object_rejects_missing_rows_before_replacing_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("missing-rows.dcm");
        std::fs::write(&path, b"sentinel").expect("sentinel");
        let model = pixel_model(None, Some(2), 1, Some(1), vec![0; 2]);
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
        let model = pixel_model(Some(2), Some(2), 2, Some(2), vec![0; 4]);
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
    fn test_write_object_rejects_malformed_zero_and_overflow_attributes() {
        let mut malformed = pixel_model(Some(1), Some(1), 1, Some(1), vec![0]);
        malformed.insert(DicomObjectNode::text(
            DicomTag::new(0x0028, 0x0010),
            "US",
            "bad",
        ));
        assert_rejected(
            malformed,
            DicomWriteError::MalformedPixelAttribute { attribute: "Rows" },
        );
        assert_rejected(
            pixel_model(Some(0), Some(1), 1, Some(1), vec![]),
            DicomWriteError::ZeroPixelAttribute { attribute: "Rows" },
        );
        let mut overflow = pixel_model(
            Some(u16::MAX),
            Some(u16::MAX),
            u16::MAX,
            Some(u16::MAX),
            vec![],
        );
        for (tag, value) in [
            (0x0100, u16::MAX - 7),
            (0x0101, u16::MAX - 7),
            (0x0102, u16::MAX - 8),
        ] {
            overflow.insert(DicomObjectNode::with_value(
                DicomTag::new(0x0028, tag),
                "US",
                value,
            ));
        }
        assert_rejected(overflow, DicomWriteError::PixelCountOverflow);
    }

    #[test]
    fn test_write_object_rejects_invalid_pixel_vr_before_replacing_file() {
        let mut model = pixel_model(Some(1), Some(1), 1, Some(1), vec![7]);
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x7FE0, 0x0010),
            "UN",
            "7",
        ));
        assert_rejected(model, DicomWriteError::InvalidPixelDataVr);
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
