use super::helpers::make_image;
use super::super::write_dicom_series_with_metadata;
use crate::format::dicom::object_model::{
    DicomObjectNode, DicomPreservationSet, DicomPreservedElement, DicomSequenceItem, DicomTag,
    DicomValue,
};
use dicom::core::Tag;
use dicom::object::open_file;

#[test]
fn test_preservation_private_text_round_trip() {
    let mut preservation = DicomPreservationSet::new();
    preservation.object.insert(DicomObjectNode::text(
        DicomTag::new(0x0009, 0x0010),
        "LO",
        "PRIVATE_ROUND_TRIP",
    ));
    let mut meta = super::helpers::make_test_metadata();
    meta.preservation = preservation;

    let image = make_image(1, 4, 4, 10.0);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("priv_rt_series");
    write_dicom_series_with_metadata(&path, &image, Some(&meta)).expect("write must succeed");

    let dcm_path = path.join("slice_0000.dcm");
    let obj = open_file(&dcm_path).expect("must open written DICOM");
    let elem = obj
        .element(Tag(0x0009, 0x0010))
        .expect("private tag (0009,0010) must exist in written DICOM");
    assert_eq!(
        elem.to_str().unwrap().trim(),
        "PRIVATE_ROUND_TRIP",
        "private tag value must survive write"
    );
}

#[test]
fn test_preservation_sequence_round_trip() {
    let mut preservation = DicomPreservationSet::new();
    let mut seq_item = DicomSequenceItem::new();
    seq_item.insert(DicomObjectNode::text(
        DicomTag::new(0x0008, 0x0104),
        "LO",
        "TestCodeMeaning",
    ));
    preservation.object.insert(DicomObjectNode {
        tag: DicomTag::new(0x0008, 0x0096),
        vr: Some("SQ".to_string()),
        value: DicomValue::Sequence(vec![seq_item]),
        private: false,
        source: None,
    });

    let mut meta = super::helpers::make_test_metadata();
    meta.preservation = preservation;

    let image = make_image(1, 4, 4, 20.0);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("seq_rt_series");
    write_dicom_series_with_metadata(&path, &image, Some(&meta)).expect("write must succeed");

    let dcm_path = path.join("slice_0000.dcm");
    let obj = open_file(&dcm_path).expect("must open written DICOM");
    let seq_elem = obj
        .element(Tag(0x0008, 0x0096))
        .expect("sequence tag (0008,0096) must exist in written DICOM");
    assert_eq!(
        seq_elem.vr(),
        dicom::core::VR::SQ,
        "sequence element must have VR=SQ"
    );
    let items = seq_elem.value().items().expect("sequence must have items");
    assert_eq!(items.len(), 1, "sequence must have exactly one item");
    let code_meaning = items[0]
        .element(Tag(0x0008, 0x0104))
        .expect("(0008,0104) must exist inside sequence item");
    assert_eq!(
        code_meaning.to_str().unwrap().trim(),
        "TestCodeMeaning",
        "sequence item value must survive write"
    );
}

#[test]
fn test_preservation_raw_bytes_round_trip() {
    let mut preservation = DicomPreservationSet::new();
    preservation.preserve(DicomPreservedElement::new(
        DicomTag::new(0x0019, 0x1001),
        Some("OB".to_string()),
        vec![0xDE_u8, 0xAD, 0xBE, 0xEF],
    ));

    let mut meta = super::helpers::make_test_metadata();
    meta.preservation = preservation;

    let image = make_image(1, 4, 4, 30.0);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("raw_rt_series");
    write_dicom_series_with_metadata(&path, &image, Some(&meta)).expect("write must succeed");

    let dcm_path = path.join("slice_0000.dcm");
    let obj = open_file(&dcm_path).expect("must open written DICOM");
    let raw_elem = obj
        .element(Tag(0x0019, 0x1001))
        .expect("raw bytes tag (0019,1001) must exist in written DICOM");
    let bytes = raw_elem.to_bytes().expect("must get raw bytes");
    assert_eq!(
        bytes.as_ref(),
        &[0xDE_u8, 0xAD, 0xBE, 0xEF],
        "raw byte payload must survive write"
    );
}
