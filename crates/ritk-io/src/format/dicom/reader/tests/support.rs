pub(super) use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
pub(super) use dicom::object::meta::FileMetaTableBuilder;
pub(super) use dicom::object::InMemDicomObject;

/// Write a minimal DICOM Part-10 file carrying only the mandatory UID tags.
pub(super) fn write_stub_dicom(
    path: &std::path::Path,
    sop_class_uid: &str,
    sop_instance_uid: &str,
) {
    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from(sop_class_uid),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from(sop_instance_uid),
    ));
    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(sop_class_uid)
                .media_storage_sop_instance_uid(sop_instance_uid)
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build must not fail");
    file_obj.write_to_file(path).expect("write must not fail");
}
