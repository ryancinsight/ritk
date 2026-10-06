//! General-purpose DICOM object writer.
//!
//! Converts a `DicomObjectModel` to a `dicom::object::InMemDicomObject`
//! and writes it as a valid DICOM Part 10 file.
//!
//! # Invariants
//! - Every node in the model appears exactly once in the output.
//! - Byte nodes retain their declared VR; native PixelData obeys OB/OW width rules.
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

    fn insert_u16(model: &mut DicomObjectModel, element: u16, value: u16) {
        model.insert(DicomObjectNode::with_value(DicomTag::new(0x0028, element), "US", value));
    }

    fn pixel_model(
        rows: Option<u16>, columns: Option<u16>, samples: u16, frames: Option<u16>,
        payload: Vec<u8>,
    ) -> DicomObjectModel {
        let mut model = DicomObjectModel::new();
        for (element, value) in [(0x0002, samples), (0x0100, 8), (0x0101, 8), (0x0102, 7), (0x0103, 0)] {
            insert_u16(&mut model, element, value);
        }
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x0028, 0x0004), "CS", if samples == 1 { "MONOCHROME2" } else { "RGB" },
        ));
        if samples > 1 { insert_u16(&mut model, 0x0006, 0); }
        for (element, value) in [(0x0010, rows), (0x0011, columns), (0x0008, frames)] {
            if let Some(value) = value { insert_u16(&mut model, element, value); }
        }
        model.insert(DicomObjectNode::bytes(DicomTag::new(0x7FE0, 0x0010), "OB", payload));
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

    fn insert_u16(model: &mut DicomObjectModel, element: u16, value: u16) {
        model.insert(DicomObjectNode::with_value(
            DicomTag::new(0x0028, element),
            "US",
            value,
        ));
    }

    fn set_pixel_format(model: &mut DicomObjectModel, bits: u16, vr: &str, payload: Vec<u8>) {
        for (element, value) in [(0x0100, bits), (0x0101, bits), (0x0102, bits - 1)] {
            insert_u16(model, element, value);
        }
        model.insert(DicomObjectNode::bytes(DicomTag::new(0x7FE0, 0x0010), vr, payload));
    }

    fn assert_pixels(model: &DicomObjectModel, path: &std::path::Path, expected: &[u8]) {
        write_object(model, path).expect("write_object");
        let object = open_file(path).expect("open_file");
        assert_eq!(
            object.element(Tag(0x7FE0, 0x0010)).expect("PixelData")
                .to_bytes().expect("pixel bytes"),
            expected
        );
    }

    #[test]
    fn test_write_object_bytes_node_roundtrip_and_single_frame_default() {
        for (frames, name) in [(Some(1), "bytes.dcm"), (None, "single-frame.dcm")] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let path = tmp.path().join(name);
            assert_pixels(&pixel_model(Some(1), Some(1), 1, frames, vec![7]), &path, &[7, 0]);
        }
    }

    #[test]
    fn test_write_object_round_trips_unsigned_and_signed_width_pixels() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("wide-pixels.dcm");
        let mut model = pixel_model(Some(1), Some(2), 1, Some(2), vec![]);
        let pixels = vec![1, 0, 255, 255, 0, 128, 0, 64];
        set_pixel_format(&mut model, 16, "OW", pixels.clone());
        assert_pixels(&model, &path, &pixels);
        insert_u16(&mut model, 0x0103, 1);
        assert_pixels(&model, &path, &pixels);
    }

    #[test]
    fn test_write_object_rejects_pixel_contracts_without_replacing_file() {
        let mut rgb = pixel_model(Some(1), Some(1), 3, Some(1), vec![1, 2, 3]);
        rgb.nodes.retain(|node| node.tag != DicomTag::new(0x0028, 0x0006));
        let mut wide = pixel_model(Some(1), Some(1), 1, Some(1), vec![7, 0]);
        set_pixel_format(&mut wide, 16, "OB", vec![7, 0]);
        let cases = [
            (pixel_model(Some(2), Some(2), 1, Some(1), vec![7]),
             DicomWriteError::PixelPayloadLengthMismatch { expected: 4, actual: 1 }),
            (wide, DicomWriteError::PixelDataVrMismatch { value: "OB".to_owned(), bits_allocated: 16 }),
            (rgb, DicomWriteError::MissingPixelAttribute { attribute: "PlanarConfiguration" }),
        ];
        for (model, expected) in cases { assert_rejected(model, expected); }
    }

    #[test]
    fn test_write_object_uses_ybr_full_422_encoded_length() {
        let mut model = pixel_model(Some(1), Some(3), 3, Some(1), vec![0; 8]);
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x0028, 0x0004), "CS", "YBR_FULL_422",
        ));
        model.nodes.retain(|node| node.tag != DicomTag::new(0x0028, 0x0006));
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("ybr422.dcm");
        assert_pixels(&model, &path, &[0; 8]);
        model.insert(DicomObjectNode::bytes(
            DicomTag::new(0x7FE0, 0x0010), "OB", vec![0; 6],
        ));
        assert_rejected(
            model,
            DicomWriteError::YbrFull422PayloadLengthMismatch { expected: 8, actual: 6 },
        );
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
