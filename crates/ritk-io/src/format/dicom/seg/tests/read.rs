use super::helpers::{build_seg_obj, make_per_frame_item, make_segment_item, write_seg_file};
use super::super::read_dicom_seg;
use dicom::core::header::Length;
use dicom::core::value::{DataSetSequence, Value};
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;

#[test]
fn test_read_seg_missing_file_returns_error() {
    let result = read_dicom_seg("/nonexistent/path/seg.dcm");
    assert!(result.is_err(), "expected Err for missing file");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("nonexistent") || msg.contains("open DICOM"),
        "error must mention file path or open action; got: {msg}"
    );
}

#[test]
fn test_read_seg_wrong_sop_class_returns_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("ct.dcm");

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x0008, 0x0016),
        VR::UI,
        PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2"),
    ));
    obj.put(DataElement::new(
        Tag(0x0008, 0x0018),
        VR::UI,
        PrimitiveValue::from("2.25.99"),
    ));
    obj.put(DataElement::new(
        Tag(0x7FE0, 0x0010),
        VR::OW,
        PrimitiveValue::U8(dicom::core::smallvec::SmallVec::new()),
    ));
    obj.with_meta(
        FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
            .media_storage_sop_instance_uid("2.25.99")
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )
    .expect("meta")
    .write_to_file(&path)
    .expect("write CT stub");

    let result = read_dicom_seg(&path);
    assert!(result.is_err(), "expected Err for wrong SOP class");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("SOP"),
        "error message must contain 'SOP'; got: {msg}"
    );
}

#[test]
fn test_read_seg_binary_4x4_single_frame() {
    // 4×4 = 16 pixels → 2 bytes, all bits set = [0xFF, 0xFF]
    let pixel_bytes: Vec<u8> = vec![0xFF, 0xFF];

    let seg_items = vec![make_segment_item(1, "TUMOR")];
    let pf_items = vec![make_per_frame_item(1, None)];

    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("seg_4x4.dcm");

    let obj = build_seg_obj(4, 4, 1, 1, "BINARY", seg_items, pf_items, pixel_bytes);
    write_seg_file(obj, &path);

    let seg = read_dicom_seg(&path).expect("read_dicom_seg 4x4 binary");

    assert_eq!(seg.rows, 4, "rows");
    assert_eq!(seg.cols, 4, "cols");
    assert_eq!(seg.n_frames, 1, "n_frames");
    assert_eq!(seg.segmentation_type, "BINARY", "segmentation_type");
    assert_eq!(seg.segments.len(), 1, "segment count");
    assert_eq!(seg.segments[0].segment_label, "TUMOR", "segment label");
    assert_eq!(seg.segments[0].segment_number, 1, "segment number");
    assert_eq!(seg.frame_segment_numbers.len(), 1, "frame_segment_numbers len");
    assert_eq!(seg.frame_segment_numbers[0], 1, "frame 0 references segment 1");
    assert_eq!(seg.pixel_data.len(), 1, "pixel_data frames");
    assert_eq!(seg.pixel_data[0].len(), 16, "pixels per frame");
    assert_eq!(
        seg.pixel_data[0],
        vec![1u8; 16],
        "all 16 pixels must be 1 for 0xFF 0xFF"
    );
}

#[test]
fn test_read_seg_two_frames_two_segments() {
    // Frame 0: all-ones (16 pixels, 2 bytes = [0xFF, 0xFF])
    // Frame 1: all-zeros (16 pixels, 2 bytes = [0x00, 0x00])
    let pixel_bytes: Vec<u8> = vec![0xFF, 0xFF, 0x00, 0x00];

    let seg_items = vec![make_segment_item(1, "GTV"), make_segment_item(2, "CTV")];
    let pf_items = vec![make_per_frame_item(1, None), make_per_frame_item(2, None)];

    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("seg_two_frames.dcm");

    let obj = build_seg_obj(4, 4, 2, 1, "BINARY", seg_items, pf_items, pixel_bytes);
    write_seg_file(obj, &path);

    let seg = read_dicom_seg(&path).expect("read_dicom_seg two frames");

    assert_eq!(seg.n_frames, 2, "n_frames");
    assert_eq!(seg.segments.len(), 2, "segment count");
    assert_eq!(seg.segments[0].segment_label, "GTV", "segment 0 label");
    assert_eq!(seg.segments[1].segment_label, "CTV", "segment 1 label");
    assert_eq!(
        seg.frame_segment_numbers,
        vec![1u16, 2u16],
        "frame_segment_numbers"
    );
    assert_eq!(seg.pixel_data[0], vec![1u8; 16], "frame 0: all ones");
    assert_eq!(seg.pixel_data[1], vec![0u8; 16], "frame 1: all zeros");
}

#[test]
fn test_read_seg_preserves_pixel_spacing() {
    let pixel_bytes: Vec<u8> = vec![0xFF, 0xFF];

    let seg_items = vec![make_segment_item(1, "ROI")];
    let pf_items = vec![make_per_frame_item(1, None)];

    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("seg_spacing.dcm");

    let mut px_meas_item = InMemDicomObject::new_empty();
    px_meas_item.put(DataElement::new(
        Tag(0x0028, 0x0030),
        VR::DS,
        PrimitiveValue::from("0.5\\0.5"),
    ));
    px_meas_item.put(DataElement::new(
        Tag(0x0018, 0x0050),
        VR::DS,
        PrimitiveValue::from("2.5"),
    ));
    let px_meas_seq = DataSetSequence::new(vec![px_meas_item], Length::UNDEFINED);

    let mut shared_item = InMemDicomObject::new_empty();
    shared_item.put(DataElement::new(
        Tag(0x0028, 0x9110),
        VR::SQ,
        Value::from(px_meas_seq),
    ));
    let shared_seq = DataSetSequence::new(vec![shared_item], Length::UNDEFINED);

    let mut obj = build_seg_obj(4, 4, 1, 1, "BINARY", seg_items, pf_items, pixel_bytes);
    obj.put(DataElement::new(
        Tag(0x5200, 0x9229),
        VR::SQ,
        Value::from(shared_seq),
    ));
    write_seg_file(obj, &path);

    let seg = read_dicom_seg(&path).expect("read_dicom_seg pixel spacing");

    assert_eq!(
        seg.pixel_spacing,
        Some([0.5, 0.5]),
        "pixel_spacing must be [0.5, 0.5]"
    );
    assert_eq!(
        seg.slice_thickness,
        Some(2.5),
        "slice_thickness must be 2.5"
    );
}

#[test]
fn test_read_seg_per_frame_image_position() {
    // 2-frame 4×4 BINARY SEG with explicit per-frame ImagePositionPatient.
    let pixel_bytes: Vec<u8> = vec![0xFF, 0xFF, 0x00, 0x00];

    let seg_items = vec![make_segment_item(1, "S1"), make_segment_item(2, "S2")];
    let pf_items = vec![
        make_per_frame_item(1, Some("0.0\\0.0\\0.0")),
        make_per_frame_item(2, Some("0.0\\0.0\\5.0")),
    ];

    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("seg_positions.dcm");

    let obj = build_seg_obj(4, 4, 2, 1, "BINARY", seg_items, pf_items, pixel_bytes);
    write_seg_file(obj, &path);

    let seg = read_dicom_seg(&path).expect("read_dicom_seg per-frame positions");

    assert_eq!(
        seg.image_position_per_frame.len(),
        2,
        "image_position_per_frame length"
    );
    assert_eq!(
        seg.image_position_per_frame[0],
        Some([0.0, 0.0, 0.0]),
        "frame 0 position"
    );
    assert_eq!(
        seg.image_position_per_frame[1],
        Some([0.0, 0.0, 5.0]),
        "frame 1 position"
    );
}
