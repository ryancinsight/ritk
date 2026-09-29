use super::super::object_model::DicomPreservationSet;
use super::elements::node_to_element;
use super::pixel_encoding::{str_to_vr, writer_tag_key};
use crate::format::dicom::writer::elements::PutValue;
use dicom::core::smallvec::SmallVec;
use dicom::core::{PrimitiveValue, Tag, VR};
use dicom::object::InMemDicomObject;
use std::collections::HashSet;

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
        obj.put(node_to_element(node));
    }
    for elem in &preservation.preserved {
        let key = writer_tag_key(elem.tag.group, elem.tag.element);
        if exclusion.contains(&key) {
            continue;
        }
        let tag = Tag(elem.tag.group, elem.tag.element);
        let vr = elem.vr.as_deref().map(str_to_vr).unwrap_or(VR::UN);
        obj.put_value(
            tag,
            vr,
            PrimitiveValue::U8(SmallVec::from_vec(elem.bytes.clone())),
        );
    }
}
