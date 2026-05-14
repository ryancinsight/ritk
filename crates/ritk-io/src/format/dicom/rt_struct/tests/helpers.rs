use super::*;

pub(super) fn write_rt_struct_file(obj: InMemDicomObject, path: &std::path::Path) {
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.481.3")
            .media_storage_sop_instance_uid("2.25.2")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("meta")
    .write_to_file(path)
    .expect("write");
}

pub(super) fn write_wrong_sop_file(sop: &str, path: &std::path::Path) {
    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(sop),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.99"),
    ));
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(sop)
            .media_storage_sop_instance_uid("2.25.99")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("meta")
    .write_to_file(path)
    .expect("write");
}

/// Build a minimal RT Structure Set with one ROI and one CLOSED_PLANAR contour.
///
/// - Structure Set Label = "TestPlan"
/// - ROI number=1, name="GTV", color="255\0\0"
/// - 1 CLOSED_PLANAR contour: [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
pub(super) fn build_single_roi_obj() -> InMemDicomObject {
    let contour_data = "0.0\\0.0\\0.0\\1.0\\0.0\\0.0\\1.0\\1.0\\0.0\\0.0\\1.0\\0.0";

    let mut contour_item = InMemDicomObject::new_empty();
    contour_item.put(DataElement::new(
        Tag(0x3006, 0x0042),
        VR::CS,
        PrimitiveValue::from("CLOSED_PLANAR"),
    ));
    contour_item.put(DataElement::new(
        Tag(0x3006, 0x0050),
        VR::DS,
        PrimitiveValue::from(contour_data),
    ));
    let contour_seq = DataSetSequence::new(vec![contour_item], Length::UNDEFINED);

    let mut roi_contour_item = InMemDicomObject::new_empty();
    roi_contour_item.put(DataElement::new(
        Tag(0x3006, 0x0084),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    roi_contour_item.put(DataElement::new(
        Tag(0x3006, 0x002A),
        VR::IS,
        PrimitiveValue::from("255\\0\\0"),
    ));
    roi_contour_item.put(DataElement::new(
        Tag(0x3006, 0x0040),
        VR::SQ,
        DicomValue::from(contour_seq),
    ));
    let roi_contour_seq = DataSetSequence::new(vec![roi_contour_item], Length::UNDEFINED);

    let mut roi_item = InMemDicomObject::new_empty();
    roi_item.put(DataElement::new(
        Tag(0x3006, 0x0022),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    roi_item.put(DataElement::new(
        Tag(0x3006, 0x0026),
        VR::LO,
        PrimitiveValue::from("GTV"),
    ));
    let roi_seq = DataSetSequence::new(vec![roi_item], Length::UNDEFINED);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x3006, 0x0002),
        VR::LO,
        PrimitiveValue::from("TestPlan"),
    ));
    obj.put(DataElement::new(
        Tag(0x3006, 0x0020),
        VR::SQ,
        DicomValue::from(roi_seq),
    ));
    obj.put(DataElement::new(
        Tag(0x3006, 0x0039),
        VR::SQ,
        DicomValue::from(roi_contour_seq),
    ));
    obj
}
