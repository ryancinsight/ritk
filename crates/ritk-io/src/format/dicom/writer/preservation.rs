use super::super::object_model::{DicomPreservationSet, DicomSequenceItem, DicomValue};
use super::utils::{str_to_vr, writer_tag_key};
use dicom::core::header::Length;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::{DataSetSequence, Value as DicomCoreValue};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::InMemDicomObject;
use std::collections::HashSet;

/// Recursively convert a DicomSequenceItem into an InMemDicomObject for writing.
pub(super) fn sequence_item_to_dicom(item: &DicomSequenceItem) -> InMemDicomObject {
    let mut obj = InMemDicomObject::new_empty();
    for node in &item.elements {
        let tag = Tag(node.tag.group, node.tag.element);
        let vr = node.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
        match &node.value {
            DicomValue::Text(s) => {
                obj.put(DataElement::new(tag, vr, PrimitiveValue::from(s.as_str())));
            }
            DicomValue::Bytes(b) => {
                obj.put(DataElement::new(
                    tag,
                    VR::OB,
                    PrimitiveValue::U8(SmallVec::from_vec(b.clone())),
                ));
            }
            DicomValue::U16(v) => {
                obj.put(DataElement::new(tag, vr, PrimitiveValue::from(*v)));
            }
            DicomValue::I32(v) => {
                obj.put(DataElement::new(
                    tag,
                    vr,
                    PrimitiveValue::from(format!("{}", v).as_str()),
                ));
            }
            DicomValue::F64(v) => {
                obj.put(DataElement::new(
                    tag,
                    vr,
                    PrimitiveValue::from(format!("{:.6}", v).as_str()),
                ));
            }
            DicomValue::Sequence(sub_items) => {
                let dicom_items: Vec<InMemDicomObject> =
                    sub_items.iter().map(sequence_item_to_dicom).collect();
                let seq = DataSetSequence::new(dicom_items, Length::UNDEFINED);
                let val: DicomCoreValue<InMemDicomObject> = DicomCoreValue::from(seq);
                obj.put(DataElement::new(tag, VR::SQ, val));
            }
            DicomValue::Empty => {}
        }
    }
    obj
}

/// Emit preserved nodes from a DicomPreservationSet into obj, skipping tags in exclusion.
///
/// Must be called BEFORE adding PixelData so the Image Pixel Module ordering invariant
/// (BitsAllocated, BitsStored, HighBit before PixelData) is preserved.
pub(super) fn emit_preservation_nodes(
    obj: &mut InMemDicomObject,
    preservation: &DicomPreservationSet,
    exclusion: &HashSet<u32>,
) {
    for node in &preservation.object.nodes {
        let key = writer_tag_key(node.tag.group, node.tag.element);
        if exclusion.contains(&key) {
            continue;
        }
        let tag = Tag(node.tag.group, node.tag.element);
        let vr = node.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
        match &node.value {
            DicomValue::Text(s) => {
                obj.put(DataElement::new(tag, vr, PrimitiveValue::from(s.as_str())));
            }
            DicomValue::Bytes(b) => {
                obj.put(DataElement::new(
                    tag,
                    VR::OB,
                    PrimitiveValue::U8(SmallVec::from_vec(b.clone())),
                ));
            }
            DicomValue::U16(v) => {
                obj.put(DataElement::new(tag, vr, PrimitiveValue::from(*v)));
            }
            DicomValue::I32(v) => {
                obj.put(DataElement::new(
                    tag,
                    vr,
                    PrimitiveValue::from(format!("{}", v).as_str()),
                ));
            }
            DicomValue::F64(v) => {
                obj.put(DataElement::new(
                    tag,
                    vr,
                    PrimitiveValue::from(format!("{:.6}", v).as_str()),
                ));
            }
            DicomValue::Sequence(items) => {
                let dicom_items: Vec<InMemDicomObject> =
                    items.iter().map(sequence_item_to_dicom).collect();
                let seq = DataSetSequence::new(dicom_items, Length::UNDEFINED);
                let val: DicomCoreValue<InMemDicomObject> = DicomCoreValue::from(seq);
                obj.put(DataElement::new(tag, VR::SQ, val));
            }
            DicomValue::Empty => {}
        }
    }
    for elem in &preservation.preserved {
        let key = writer_tag_key(elem.tag.group, elem.tag.element);
        if exclusion.contains(&key) {
            continue;
        }
        let tag = Tag(elem.tag.group, elem.tag.element);
        let vr = elem.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
        obj.put(DataElement::new(
            tag,
            vr,
            PrimitiveValue::U8(SmallVec::from_vec(elem.bytes.clone())),
        ));
    }
}
