#![allow(unused_imports)]

use super::super::geometry::{
    analyze_slice_spacing, dot, normalize, resample_frames_linear, slice_normal_from_iop,
};
use super::super::loader::{
    load_dicom_series_with_metadata, load_from_series, read_dicom_series_with_metadata,
};
use super::super::pixel::{decode_pixel_bytes, read_slice_pixels};
use super::super::scan::scan_dicom_directory;
use super::super::types::{
    DicomReadMetadata, DicomSeriesInfo, DicomSliceMetadata, PatientPosition,
};
use super::super::utils::is_likely_dicom_file;
use super::support::*;
use crate::format::dicom::{
    DicomObjectNode, DicomPreservationSet, DicomPreservedElement, DicomTag, DicomValue,
};
use ritk_core::image::Image;
use ritk_dicom::TransferSyntaxKind;
use ritk_spatial::{Direction, Point, Spacing};
#[test]
fn test_patient_position_parser_covers_known_codes() {
    assert_eq!(
        PatientPosition::from_code("HFS"),
        PatientPosition::HeadFirstSupine
    );
    assert_eq!(
        PatientPosition::from_code("HFP"),
        PatientPosition::HeadFirstProne
    );
    assert_eq!(
        PatientPosition::from_code("FFS"),
        PatientPosition::FeetFirstSupine
    );
    assert_eq!(
        PatientPosition::from_code("FFP"),
        PatientPosition::FeetFirstProne
    );
    assert_eq!(
        PatientPosition::from_code("HFDR"),
        PatientPosition::HeadFirstDecubitusRight
    );
    assert_eq!(
        PatientPosition::from_code("HFDL"),
        PatientPosition::HeadFirstDecubitusLeft
    );
    assert_eq!(
        PatientPosition::from_code("FFDR"),
        PatientPosition::FeetFirstDecubitusRight
    );
    assert_eq!(
        PatientPosition::from_code("FFDL"),
        PatientPosition::FeetFirstDecubitusLeft
    );
    assert_eq!(PatientPosition::from_code("hfs").to_string(), "HFS");

    match PatientPosition::from_code("mystery") {
        PatientPosition::Unknown(code) => assert_eq!(code.as_str(), "MYST"),
        other => panic!("unexpected patient position variant: {:?}", other),
    }
}

#[test]
fn test_patient_position_is_captured_from_dicom_tag() {
    use dicom::core::{Tag, VR};
    use dicom::object::{FileMetaTableBuilder, InMemDicomObject};
    use dicom_core::smallvec::SmallVec;
    use dicom_core::PrimitiveValue;

    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("position_series");
    std::fs::create_dir_all(&dir).unwrap();

    let slice_path = dir.join("slice_0000.dcm");
    let mut obj = InMemDicomObject::new_empty();
    obj.put(dicom::core::DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.999002"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0008, 0x0060),
        VR::CS,
        PrimitiveValue::from("CT"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0010),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[1u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0011),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[1u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0100),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[16u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[16u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0102),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[15u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0103),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[0u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0002),
        VR::US,
        PrimitiveValue::U16(SmallVec::from_slice(&[1u16])),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0004),
        VR::CS,
        PrimitiveValue::from("MONOCHROME2"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0020, 0x0037),
        VR::DS,
        PrimitiveValue::from("1.000000\\0.000000\\0.000000\\0.000000\\1.000000\\0.000000"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0020, 0x0032),
        VR::DS,
        PrimitiveValue::from("0.000000\\0.000000\\0.000000"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0028, 0x0030),
        VR::DS,
        PrimitiveValue::from("1.000000\\1.000000"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0018, 0x0050),
        VR::DS,
        PrimitiveValue::from("1.000000"),
    ));
    obj.put(dicom::core::DataElement::new(
        Tag(0x0018, 0x5100),
        VR::CS,
        PrimitiveValue::from("HFP"),
    ));
    let pixel_u16: Vec<u16> = vec![1234];
    obj.put(dicom::core::DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
    ));

    let file_obj = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                .media_storage_sop_instance_uid("2.25.999002")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("meta build failed");
    file_obj
        .write_to_file(&slice_path)
        .expect("write slice failed");

    let info = scan_dicom_directory(&dir).expect("scan must succeed");
    assert_eq!(info.num_slices, 1, "must find one slice");

    let slice = &info.metadata.slices[0];
    assert_eq!(
        slice.patient_position,
        Some(PatientPosition::HeadFirstProne)
    );
    assert_eq!(
        slice
            .patient_position
            .as_ref()
            .map(|value| value.to_string()),
        Some("HFP".to_string())
    );
}
