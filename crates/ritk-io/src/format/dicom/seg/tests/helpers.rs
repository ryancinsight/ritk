use dicom::core::header::Length;
use dicom::core::value::{DataSetSequence, Value};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;

use super::super::types::SEG_SOP_CLASS_UID;

/// Build a minimal DICOM-SEG InMemDicomObject with given geometry and raw pixel bytes.
#[allow(clippy::too_many_arguments)] // 8-arg test helper; collecting into a struct would add ceremony
pub(super) fn build_seg_obj(
    rows: u16,
    cols: u16,
    n_frames: u32,
    bits_allocated: u16,
    segmentation_type: &str,
    seg_items: Vec<InMemDicomObject>,
    pf_items: Vec<InMemDicomObject>,
    pixel_bytes: Vec<u8>,
) -> InMemDicomObject {
    let mut obj = InMemDicomObject::new_empty();

    obj.put(DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(rows),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(cols),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from(n_frames.to_string().as_str()),
    ));
    obj.put(DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(bits_allocated),
    ));
    obj.put(DataElement::new(
        Tag(0x0062, 0x0001),
        VR::CS,
        PrimitiveValue::from(segmentation_type),
    ));

    if !seg_items.is_empty() {
        let seq = DataSetSequence::new(seg_items, Length::UNDEFINED);
        obj.put(DataElement::new(
            Tag(0x0062, 0x0002),
            VR::SQ,
            Value::from(seq),
        ));
    }

    if !pf_items.is_empty() {
        let seq = DataSetSequence::new(pf_items, Length::UNDEFINED);
        obj.put(DataElement::new(
            Tag(0x5200, 0x9230),
            VR::SQ,
            Value::from(seq),
        ));
    }

    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OB,
        PrimitiveValue::U8(dicom::core::smallvec::SmallVec::from_vec(pixel_bytes)),
    ));

    obj
}

/// Write an InMemDicomObject as a DICOM-SEG Part 10 file.
pub(super) fn write_seg_file(obj: InMemDicomObject, path: &std::path::Path) {
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(SEG_SOP_CLASS_UID)
            .media_storage_sop_instance_uid("2.25.1")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("meta build")
    .write_to_file(path)
    .expect("write DICOM-SEG file");
}

/// Build a segment sequence item for a single segment.
pub(super) fn make_segment_item(segment_number: u16, label: &str) -> InMemDicomObject {
    let mut item = InMemDicomObject::new_empty();
    item.put(DataElement::new(
        Tag(0x0062, 0x0004),
        VR::US,
        PrimitiveValue::from(segment_number),
    ));
    item.put(DataElement::new(
        Tag(0x0062, 0x0005),
        VR::LO,
        PrimitiveValue::from(label),
    ));
    item
}

/// Build a per-frame item containing SegmentIdentification and optional PlanePosition.
pub(super) fn make_per_frame_item(
    referenced_segment_number: u16,
    image_position: Option<&str>,
) -> InMemDicomObject {
    let mut seg_id_item = InMemDicomObject::new_empty();
    seg_id_item.put(DataElement::new(
        Tag(0x0062, 0x000B),
        VR::US,
        PrimitiveValue::from(referenced_segment_number),
    ));
    let seg_id_seq = DataSetSequence::new(vec![seg_id_item], Length::UNDEFINED);

    let mut pf = InMemDicomObject::new_empty();
    pf.put(DataElement::new(
        Tag(0x0062, 0x000A),
        VR::SQ,
        Value::from(seg_id_seq),
    ));

    if let Some(pos_str) = image_position {
        let mut pos_item = InMemDicomObject::new_empty();
        pos_item.put(DataElement::new(
            Tag(0x0020, 0x0032),
            VR::DS,
            PrimitiveValue::from(pos_str),
        ));
        let pos_seq = DataSetSequence::new(vec![pos_item], Length::UNDEFINED);
        pf.put(DataElement::new(
            Tag(0x0020, 0x9113),
            VR::SQ,
            Value::from(pos_seq),
        ));
    }

    pf
}
