use super::helpers::*;
use super::*;

/// Invariant: a nonexistent path must produce Err.
#[test]
fn test_read_rt_struct_missing_file_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("nonexistent.dcm");
    let result = read_rt_struct(&path);
    assert!(result.is_err(), "nonexistent path must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("nonexistent") || msg.contains("open") || msg.contains("No such"),
        "error must mention the open failure; got: {msg}"
    );
}

/// Invariant: a file whose SOP Class UID ≠ RT Structure Set must produce Err
/// containing the rejected UID in the message.
#[test]
fn test_read_rt_struct_wrong_sop_class_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("wrong_sop.dcm");
    write_wrong_sop_file("1.2.840.10008.5.1.4.1.1.2", &path);

    let result = read_rt_struct(&path);
    assert!(result.is_err(), "wrong SOP class must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("1.2.840.10008.5.1.4.1.1.2"),
        "error must contain the rejected SOP UID; got: {msg}"
    );
}

/// Invariant: a single-ROI RT Structure Set with one CLOSED_PLANAR contour
/// must parse the label, roi_name, display_color, and all four contour points.
#[test]
fn test_read_rt_struct_single_roi_closed_planar() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("single_roi.dcm");
    write_rt_struct_file(build_single_roi_obj(), &path);

    let ss = read_rt_struct(&path).expect("read_rt_struct must succeed");

    assert_eq!(ss.structure_set_label, "TestPlan", "structure_set_label");
    assert_eq!(ss.rois.len(), 1, "must contain exactly 1 ROI");

    let roi = &ss.rois[0];
    assert_eq!(roi.roi_name, "GTV", "roi_name must be 'GTV'");
    assert_eq!(roi.display_color, Some([255, 0, 0]), "display_color");
    assert_eq!(roi.contours.len(), 1, "must contain 1 contour");

    let contour = &roi.contours[0];
    assert_eq!(
        contour.geometric_type.as_dicom_str(),
        "CLOSED_PLANAR",
        "geometric_type"
    );
    assert_eq!(contour.points.len(), 4, "contour must contain 4 points");
    assert_eq!(contour.points[0], [0.0, 0.0, 0.0], "point[0]");
    assert_eq!(contour.points[1], [1.0, 0.0, 0.0], "point[1]");
    assert_eq!(contour.points[2], [1.0, 1.0, 0.0], "point[2]");
    assert_eq!(contour.points[3], [0.0, 1.0, 0.0], "point[3]");
}

/// Invariant: when two ROIs are present, both are returned sorted ascending
/// by roi_number regardless of insertion order in the DICOM sequence.
#[test]
fn test_read_rt_struct_two_rois() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("two_rois.dcm");

    let mut roi_item2 = InMemDicomObject::new_empty();
    roi_item2.put(DataElement::new(
        Tag(0x3006, 0x0022),
        VR::IS,
        PrimitiveValue::from("2"),
    ));
    roi_item2.put(DataElement::new(
        Tag(0x3006, 0x0026),
        VR::LO,
        PrimitiveValue::from("PTV"),
    ));

    let mut roi_item1 = InMemDicomObject::new_empty();
    roi_item1.put(DataElement::new(
        Tag(0x3006, 0x0022),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    roi_item1.put(DataElement::new(
        Tag(0x3006, 0x0026),
        VR::LO,
        PrimitiveValue::from("GTV"),
    ));

    // Insert ROI 2 first to exercise sort correctness.
    let roi_seq = DataSetSequence::new(vec![roi_item2, roi_item1], Length::UNDEFINED);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x3006, 0x0002),
        VR::LO,
        PrimitiveValue::from("TwoPlan"),
    ));
    obj.put(DataElement::new(
        Tag(0x3006, 0x0020),
        VR::SQ,
        DicomValue::from(roi_seq),
    ));

    write_rt_struct_file(obj, &path);

    let ss = read_rt_struct(&path).expect("read_rt_struct must succeed");
    assert_eq!(ss.rois.len(), 2, "must contain 2 ROIs");
    assert_eq!(ss.rois[0].roi_number, 1, "first ROI must have number 1");
    assert_eq!(ss.rois[0].roi_name, "GTV", "first ROI must be GTV");
    assert_eq!(ss.rois[1].roi_number, 2, "second ROI must have number 2");
    assert_eq!(ss.rois[1].roi_name, "PTV", "second ROI must be PTV");
}

/// Invariant: ROI Observations Sequence (3006,0080) must populate
/// `roi_interpreted_type` on the matching ROI.
#[test]
fn test_read_rt_struct_roi_interpreted_type() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("roi_type.dcm");

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

    let mut obs_item = InMemDicomObject::new_empty();
    obs_item.put(DataElement::new(
        Tag(0x3006, 0x0084),
        VR::IS,
        PrimitiveValue::from("1"),
    ));
    obs_item.put(DataElement::new(
        Tag(0x3006, 0x00A4),
        VR::CS,
        PrimitiveValue::from("GTV"),
    ));
    let obs_seq = DataSetSequence::new(vec![obs_item], Length::UNDEFINED);

    let mut obj = InMemDicomObject::new_empty();
    obj.put(DataElement::new(
        Tag(0x3006, 0x0002),
        VR::LO,
        PrimitiveValue::from("ObsPlan"),
    ));
    obj.put(DataElement::new(
        Tag(0x3006, 0x0020),
        VR::SQ,
        DicomValue::from(roi_seq),
    ));
    obj.put(DataElement::new(
        Tag(0x3006, 0x0080),
        VR::SQ,
        DicomValue::from(obs_seq),
    ));

    write_rt_struct_file(obj, &path);

    let ss = read_rt_struct(&path).expect("read_rt_struct must succeed");
    assert_eq!(ss.rois.len(), 1, "must contain 1 ROI");
    assert_eq!(
        ss.rois[0].roi_interpreted_type,
        Some("GTV".to_string()),
        "roi_interpreted_type must be Some(\"GTV\")"
    );
}
