use ritk_dicom::dataset::Dataset;
use ritk_dicom::dataset::FileMetaBuilder;
use ritk_dicom::dataset::Length;
use ritk_dicom::dataset::{DataElement, DicomTag, PrimitiveValue, VR};
use ritk_dicom::dataset::{Sequence, Value};

use super::super::types::SEG_SOP_CLASS_UID;

/// Build a minimal DICOM-SEG Dataset with given geometry and raw pixel bytes.
#[expect(clippy::too_many_arguments, reason = "ratchet RITK-LINT-1")] // 8-arg test helper; collecting into a struct would add ceremony
pub(super) fn build_seg_obj(
    rows: u16,
    cols: u16,
    n_frames: u32,
    bits_allocated: u16,
    segmentation_type: &str,
    seg_items: Vec<Dataset>,
    pf_items: Vec<Dataset>,
    pixel_bytes: Vec<u8>,
) -> Dataset {
    let mut obj = Dataset::new();

    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::from(rows),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::from(cols),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0008),
        VR::IS,
        PrimitiveValue::from(n_frames.to_string().as_str()),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::from(bits_allocated),
    ));
    obj.put(DataElement::new(
        DicomTag::new(0x0062, 0x0001),
        VR::CS,
        PrimitiveValue::from(segmentation_type),
    ));

    if !seg_items.is_empty() {
        let seq = Sequence::new(seg_items, Length::Undefined);
        obj.put(DataElement::new(
            DicomTag::new(0x0062, 0x0002),
            VR::SQ,
            Value::from(seq),
        ));
    }

    if !pf_items.is_empty() {
        let seq = Sequence::new(pf_items, Length::Undefined);
        obj.put(DataElement::new(
            DicomTag::new(0x5200, 0x9230),
            VR::SQ,
            Value::from(seq),
        ));
    }

    obj.put(DataElement::new(
        DicomTag::new(0x7FE0, 0x0010),
        VR::OB,
        PrimitiveValue::from(pixel_bytes),
    ));

    obj
}

/// Write an Dataset as a DICOM-SEG Part 10 file.
pub(super) fn write_seg_file(obj: Dataset, path: &std::path::Path) {
    obj.with_meta(
        FileMetaBuilder::new()
            .media_storage_sop_class_uid(SEG_SOP_CLASS_UID)
            .media_storage_sop_instance_uid("2.25.1")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("meta build")
    .write_to_file(path)
    .expect("write DICOM-SEG file");
}

/// Build a segment sequence item for a single segment.
pub(super) fn make_segment_item(segment_number: u16, label: &str) -> Dataset {
    let mut item = Dataset::new();
    item.put(DataElement::new(
        DicomTag::new(0x0062, 0x0004),
        VR::US,
        PrimitiveValue::from(segment_number),
    ));
    item.put(DataElement::new(
        DicomTag::new(0x0062, 0x0005),
        VR::LO,
        PrimitiveValue::from(label),
    ));
    item
}

/// Build a per-frame item containing SegmentIdentification and optional PlanePosition.
pub(super) fn make_per_frame_item(
    referenced_segment_number: u16,
    image_position: Option<&str>,
) -> Dataset {
    let mut seg_id_item = Dataset::new();
    seg_id_item.put(DataElement::new(
        DicomTag::new(0x0062, 0x000B),
        VR::US,
        PrimitiveValue::from(referenced_segment_number),
    ));
    let seg_id_seq = Sequence::new(vec![seg_id_item], Length::Undefined);

    let mut pf = Dataset::new();
    pf.put(DataElement::new(
        DicomTag::new(0x0062, 0x000A),
        VR::SQ,
        Value::from(seg_id_seq),
    ));

    if let Some(pos_str) = image_position {
        let mut pos_item = Dataset::new();
        pos_item.put(DataElement::new(
            DicomTag::new(0x0020, 0x0032),
            VR::DS,
            PrimitiveValue::from(pos_str),
        ));
        let pos_seq = Sequence::new(vec![pos_item], Length::Undefined);
        pf.put(DataElement::new(
            DicomTag::new(0x0020, 0x9113),
            VR::SQ,
            Value::from(pos_seq),
        ));
    }

    pf
}
