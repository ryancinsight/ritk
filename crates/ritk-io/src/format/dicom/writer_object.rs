//! General-purpose DICOM object writer.
//!
//! Converts a `DicomObjectModel` to a `dicom::object::InMemDicomObject`
//! and writes it as a valid DICOM Part 10 file.
//!
//! # Invariants
//! - Every node in the model appears exactly once in the output.
//! - Byte nodes use OB VR unconditionally.
//! - Sequence nodes produce SQ elements with undefined length.

use anyhow::{Context, Result};
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::core::header::Length;
use dicom::core::value::{DataSetSequence, Value as DicomCoreValue};
use dicom::object::{InMemDicomObject, meta::FileMetaTableBuilder};
use std::path::Path;
use super::object_model::{
    DicomObjectModel, DicomObjectNode, DicomSequenceItem, DicomTag, DicomValue,
};

fn str_to_vr(s: &str) -> VR {
    match s {
        "AE" => VR::AE, "AS" => VR::AS, "AT" => VR::AT,
        "CS" => VR::CS, "DA" => VR::DA, "DS" => VR::DS,
        "DT" => VR::DT, "FL" => VR::FL, "FD" => VR::FD,
        "IS" => VR::IS, "LO" => VR::LO, "LT" => VR::LT,
        "OB" => VR::OB, "OD" => VR::OD, "OF" => VR::OF,
        "OL" => VR::OL, "OW" => VR::OW, "PN" => VR::PN,
        "SH" => VR::SH, "SL" => VR::SL, "SQ" => VR::SQ,
        "SS" => VR::SS, "ST" => VR::ST, "TM" => VR::TM,
        "UC" => VR::UC, "UI" => VR::UI, "UL" => VR::UL,
        "UN" => VR::UN, "UR" => VR::UR, "US" => VR::US,
        "UT" => VR::UT,
        _ => VR::UN,
    }
}

fn sequence_item_to_dicom(item: &DicomSequenceItem) -> InMemDicomObject {
    let mut obj = InMemDicomObject::new_empty();
    for node in &item.elements {
        if let Ok(elem) = node_to_element(node) {
            obj.put(elem);
        }
    }
    obj
}

fn node_to_element(node: &DicomObjectNode) -> Result<DataElement<InMemDicomObject>> {
    let tag = Tag(node.tag.group, node.tag.element);
    let vr = node.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
    let elem = match &node.value {
        DicomValue::Text(s) => {
            DataElement::new(tag, vr, PrimitiveValue::from(s.as_str()))
        }
        DicomValue::Bytes(b) => {
            DataElement::new(tag, VR::OB, PrimitiveValue::U8(SmallVec::from_vec(b.clone())))
        }
        DicomValue::U16(v) => {
            DataElement::new(tag, vr, PrimitiveValue::from(*v))
        }
        DicomValue::I32(v) => {
            DataElement::new(tag, vr, PrimitiveValue::from(format!("{}", v).as_str()))
        }
        DicomValue::F64(v) => {
            DataElement::new(tag, vr, PrimitiveValue::from(format!("{:.6}", v).as_str()))
        }
        DicomValue::Sequence(items) => {
            let dicom_items: Vec<InMemDicomObject> =
                items.iter().map(sequence_item_to_dicom).collect();
            let seq = DataSetSequence::new(dicom_items, Length::UNDEFINED);
            let val: DicomCoreValue<InMemDicomObject> = DicomCoreValue::from(seq);
            DataElement::new(tag, VR::SQ, val)
        }
        DicomValue::Empty => {
            DataElement::new(tag, vr, PrimitiveValue::Empty)
        }
    };
    Ok(elem)
}

/// Convert a `DicomObjectModel` to an `InMemDicomObject`.
pub fn model_to_in_mem(model: &DicomObjectModel) -> Result<InMemDicomObject> {
    let mut obj = InMemDicomObject::new_empty();
    for node in &model.nodes {
        let elem = node_to_element(node)
            .with_context(|| format!("node_to_element failed for tag {:?}", node.tag))?;
        obj.put(elem);
    }
    Ok(obj)
}

/// Write a `DicomObjectModel` to a DICOM Part 10 file.
pub fn write_object(model: &DicomObjectModel, path: &Path) -> Result<()> {
    let obj = model_to_in_mem(model)?;
    let sop_class = model.get(DicomTag::new(0x0008, 0x0016))
        .and_then(|n| n.value.as_text())
        .unwrap_or("1.2.840.10008.5.1.4.1.1.7");
    let sop_inst = model.get(DicomTag::new(0x0008, 0x0018))
        .and_then(|n| n.value.as_text())
        .unwrap_or("2.25.0");
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(sop_class)
                .media_storage_sop_instance_uid(sop_inst)
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .map_err(|e| anyhow::anyhow!("DICOM meta build failed: {e}"))?;
    file_obj.write_to_file(path)
        .map_err(|e| anyhow::anyhow!("write_to_file {:?} failed: {e}", path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::object_model::{DicomObjectModel, DicomObjectNode, DicomTag};
    use dicom::object::open_file;

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
            DicomTag::new(0x0008, 0x0060), "CS", "CT",
        ));
        let obj = model_to_in_mem(&model).expect("model_to_in_mem");
        assert_eq!(obj.iter().count(), 1);
    }

    #[test]
    fn test_model_to_in_mem_u16_node() {
        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::u16(
            DicomTag::new(0x0028, 0x0100), "US", 16,
        ));
        let obj = model_to_in_mem(&model).expect("model_to_in_mem");
        assert_eq!(obj.iter().count(), 1);
    }

    #[test]
    fn test_write_object_bytes_node_non_empty() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("bytes.dcm");
        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::bytes(
            DicomTag::new(0x7FE0, 0x0010), "OB", vec![0u8; 20],
        ));
        write_object(&model, &path).expect("write_object");
        let len = std::fs::metadata(&path).expect("metadata").len();
        assert!(len > 128, "file must exceed preamble size, got {len}");
    }

    #[test]
    fn test_write_object_text_node_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("roundtrip.dcm");
        let mut model = DicomObjectModel::new();
        model.insert(DicomObjectNode::text(
            DicomTag::new(0x0008, 0x0060), "CS", "MR",
        ));
        write_object(&model, &path).expect("write_object");
        let obj = open_file(&path).expect("open_file");
        let val = obj.element(dicom::core::Tag(0x0008, 0x0060))
            .expect("element")
            .to_str()
            .expect("to_str");
        assert_eq!(val.trim(), "MR", "roundtrip value mismatch");
    }
}
