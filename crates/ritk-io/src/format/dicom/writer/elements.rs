//! DICOM data elements from object-model nodes.
//!
//! Every writer that emits `DicomObjectNode`s converts them here, so a node
//! value maps to one element encoding whichever writer emits it. A node always
//! produces exactly one element: `DicomValue::Empty` is a present element with
//! a zero-length value (a DICOM Type 2 attribute), never an omitted one.

use crate::format::dicom::object_model::{DicomObjectNode, DicomSequenceItem, DicomValue};
use crate::format::dicom::writer::pixel_encoding::str_to_vr;
use dicom::core::header::Length;
use dicom::core::smallvec::SmallVec;
use dicom::core::value::{DataSetSequence, Value as DicomCoreValue};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::InMemDicomObject;

/// Convert one node into the element that encodes it.
///
/// A node without a VR is written as `UN`; byte payloads are always `OB` and
/// sequences always `SQ` with undefined length.
pub(crate) fn node_to_element(node: &DicomObjectNode) -> DataElement<InMemDicomObject> {
    let tag = Tag(node.tag.group, node.tag.element);
    let vr = node.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
    match &node.value {
        DicomValue::Text(s) => DataElement::new(tag, vr, PrimitiveValue::from(s.as_str())),
        DicomValue::Bytes(b) => DataElement::new(
            tag,
            VR::OB,
            PrimitiveValue::U8(SmallVec::from_vec(b.clone())),
        ),
        DicomValue::U16(v) => DataElement::new(tag, vr, PrimitiveValue::from(*v)),
        DicomValue::I32(v) => {
            DataElement::new(tag, vr, PrimitiveValue::from(format!("{v}").as_str()))
        }
        DicomValue::F64(v) => {
            DataElement::new(tag, vr, PrimitiveValue::from(format!("{v:.6}").as_str()))
        }
        DicomValue::Sequence(items) => {
            let dicom_items: Vec<InMemDicomObject> =
                items.iter().map(sequence_item_to_dicom).collect();
            let seq = DataSetSequence::new(dicom_items, Length::UNDEFINED);
            let val: DicomCoreValue<InMemDicomObject> = DicomCoreValue::from(seq);
            DataElement::new(tag, VR::SQ, val)
        }
        DicomValue::Empty => DataElement::new(tag, vr, PrimitiveValue::Empty),
    }
}

/// Convert a sequence item, recursively, into the object it encodes.
pub(crate) fn sequence_item_to_dicom(item: &DicomSequenceItem) -> InMemDicomObject {
    let mut obj = InMemDicomObject::new_empty();
    for node in &item.elements {
        obj.put(node_to_element(node));
    }
    obj
}
