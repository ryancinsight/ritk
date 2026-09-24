use super::super::object_model::{DicomPreservationSet, DicomValue};
use super::elements::{node_to_element, put_bytes};
use super::pixel_encoding::writer_tag_key;
use dicom::core::Tag;
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
        match &node.value {
            DicomValue::Empty => {}
            _ => {
                obj.put(node_to_element(node));
            }
        }
    }
    for elem in &preservation.preserved {
        let key = writer_tag_key(elem.tag.group, elem.tag.element);
        if exclusion.contains(&key) {
            continue;
        }
        put_bytes(
            obj,
            Tag(elem.tag.group, elem.tag.element),
            elem.vr
                .as_deref()
                .map(super::pixel_encoding::str_to_vr)
                .unwrap_or(dicom::core::VR::UN),
            elem.bytes.clone(),
        );
    }
}
