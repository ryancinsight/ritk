use super::super::object_model::{DicomObjectNode, DicomSequenceItem, DicomValue};
use super::pixel_encoding::str_to_vr;
use dicom::core::header::Length;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::{DataSetSequence, Value as DicomCoreValue};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::InMemDicomObject;
use std::fmt::Display;

pub(crate) fn put_text(obj: &mut InMemDicomObject, tag: Tag, vr: VR, value: &str) {
    obj.put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
}

pub(crate) fn put_u16(obj: &mut InMemDicomObject, tag: Tag, value: u16) {
    obj.put(DataElement::new(tag, VR::US, PrimitiveValue::from(value)));
}

pub(crate) fn put_is(obj: &mut InMemDicomObject, tag: Tag, value: impl Display) {
    let value = value.to_string();
    put_text(obj, tag, VR::IS, value.as_str());
}

pub(crate) fn put_ds(obj: &mut InMemDicomObject, tag: Tag, value: impl Display) {
    let value = value.to_string();
    put_text(obj, tag, VR::DS, value.as_str());
}

pub(crate) fn put_bytes(obj: &mut InMemDicomObject, tag: Tag, vr: VR, bytes: Vec<u8>) {
    obj.put(DataElement::new(
        tag,
        vr,
        PrimitiveValue::U8(SmallVec::from_vec(bytes)),
    ));
}

pub(crate) fn put_sequence(obj: &mut InMemDicomObject, tag: Tag, items: Vec<InMemDicomObject>) {
    obj.put(DataElement::new(tag, VR::SQ, sequence_value(items)));
}

pub(crate) fn sequence_item_to_dicom(item: &DicomSequenceItem) -> InMemDicomObject {
    let mut obj = InMemDicomObject::new_empty();
    for node in &item.elements {
        obj.put(node_to_element(node));
    }
    obj
}

pub(crate) fn node_to_element(node: &DicomObjectNode) -> DataElement<InMemDicomObject> {
    let tag = Tag(node.tag.group, node.tag.element);
    let vr = node.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
    match &node.value {
        DicomValue::Text(s) => DataElement::new(tag, vr, PrimitiveValue::from(s.as_str())),
        DicomValue::Bytes(bytes) => DataElement::new(
            tag,
            VR::OB,
            PrimitiveValue::U8(SmallVec::from_vec(bytes.clone())),
        ),
        DicomValue::U16(value) => DataElement::new(tag, vr, PrimitiveValue::from(*value)),
        DicomValue::I32(value) => {
            let value = value.to_string();
            DataElement::new(tag, vr, PrimitiveValue::from(value.as_str()))
        }
        DicomValue::F64(value) => {
            let value = format!("{value:.6}");
            DataElement::new(tag, vr, PrimitiveValue::from(value.as_str()))
        }
        DicomValue::Sequence(items) => {
            let dicom_items = items.iter().map(sequence_item_to_dicom).collect();
            DataElement::new(tag, VR::SQ, sequence_value(dicom_items))
        }
        DicomValue::Empty => DataElement::new(tag, vr, PrimitiveValue::Empty),
    }
}

fn sequence_value(items: Vec<InMemDicomObject>) -> DicomCoreValue<InMemDicomObject> {
    DicomCoreValue::from(DataSetSequence::new(items, Length::UNDEFINED))
}
