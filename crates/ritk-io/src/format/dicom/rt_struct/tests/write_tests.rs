use super::*;
use arrayvec::ArrayString;

/// Invariant: write_rt_struct followed by read_rt_struct preserves all fields
/// for a single ROI with one CLOSED_PLANAR contour, display_color, and
/// roi_interpreted_type.
#[test]
fn test_write_rt_struct_single_roi_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("roundtrip.dcm");

    let ss = RtStructureSet {
        structure_set_label: "SRS-01".into(),
        structure_set_name: Some("Brain SRS".into()),
        rois: vec![RtRoiInfo {
            roi_number: 1,
            roi_name: "GTV".into(),
            roi_description: Some("Gross Tumor Volume".into()),
            roi_interpreted_type: Some("GTV".into()),
            display_color: Some([255, 0, 0]),
            contours: vec![RtContour {
                geometric_type: ArrayString::from("CLOSED_PLANAR").unwrap(),
                points: vec![
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [10.0, 10.0, 0.0],
                    [0.0, 10.0, 0.0],
                ],
            }],
        }],
    };

    write_rt_struct(&path, &ss).expect("write_rt_struct");
    let result = read_rt_struct(&path).expect("read_rt_struct round-trip");

    assert_eq!(result.structure_set_label, "SRS-01", "label preserved");
    assert_eq!(
        result.structure_set_name,
        Some("Brain SRS".into()),
        "name preserved"
    );
    assert_eq!(result.rois.len(), 1, "roi count preserved");

    let roi = &result.rois[0];
    assert_eq!(roi.roi_number, 1, "roi_number preserved");
    assert_eq!(roi.roi_name, "GTV", "roi_name preserved");
    assert_eq!(
        roi.roi_description,
        Some("Gross Tumor Volume".into()),
        "roi_description preserved"
    );
    assert_eq!(
        roi.roi_interpreted_type,
        Some("GTV".into()),
        "roi_interpreted_type preserved"
    );
    assert_eq!(
        roi.display_color,
        Some([255, 0, 0]),
        "display_color preserved"
    );
    assert_eq!(roi.contours.len(), 1, "contour count preserved");
    assert_eq!(
        roi.contours[0].geometric_type.as_str(),
        "CLOSED_PLANAR",
        "geometric_type"
    );
    assert_eq!(roi.contours[0].points.len(), 4, "point count preserved");
    assert_eq!(roi.contours[0].points[0], [0.0, 0.0, 0.0], "point[0]");
    assert_eq!(roi.contours[0].points[2], [10.0, 10.0, 0.0], "point[2]");
}

/// Invariant: two ROIs with different roi_number, contour types, and colors
/// round-trip correctly and are sorted ascending by roi_number.
#[test]
fn test_write_rt_struct_multi_roi_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("multi_roi.dcm");

    let ss = RtStructureSet {
        structure_set_label: "Multi".into(),
        structure_set_name: None,
        rois: vec![
            RtRoiInfo {
                roi_number: 2,
                roi_name: "PTV2".into(),
                roi_description: None,
                roi_interpreted_type: Some("PTV".into()),
                display_color: Some([0, 255, 0]),
                contours: vec![RtContour {
                    geometric_type: ArrayString::from("CLOSED_PLANAR").unwrap(),
                    points: vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
                }],
            },
            RtRoiInfo {
                roi_number: 1,
                roi_name: "PTV1".into(),
                roi_description: Some("Primary PTV".into()),
                roi_interpreted_type: Some("PTV".into()),
                display_color: Some([255, 255, 0]),
                contours: vec![RtContour {
                    geometric_type: ArrayString::from("OPEN_PLANAR").unwrap(),
                    points: vec![[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
                }],
            },
        ],
    };

    write_rt_struct(&path, &ss).expect("write_rt_struct multi");
    let result = read_rt_struct(&path).expect("read_rt_struct multi");

    assert_eq!(result.rois.len(), 2, "must have 2 ROIs");
    // Verify sort ascending by roi_number (ROI 1 first even though ROI 2 was first in input).
    assert_eq!(result.rois[0].roi_number, 1, "first ROI must be number 1");
    assert_eq!(result.rois[0].roi_name, "PTV1", "first ROI name");
    assert_eq!(
        result.rois[0].roi_description,
        Some("Primary PTV".into()),
        "description preserved"
    );
    assert_eq!(
        result.rois[0].display_color,
        Some([255, 255, 0]),
        "color preserved"
    );
    assert_eq!(
        result.rois[0].contours[0].geometric_type.as_str(),
        "OPEN_PLANAR",
        "geo type"
    );
    assert_eq!(
        result.rois[0].contours[0].points.len(),
        2,
        "open planar points"
    );

    assert_eq!(result.rois[1].roi_number, 2, "second ROI must be number 2");
    assert_eq!(result.rois[1].roi_name, "PTV2", "second ROI name");
    assert_eq!(
        result.rois[1].display_color,
        Some([0, 255, 0]),
        "second color"
    );
    assert_eq!(
        result.rois[1].contours[0].points.len(),
        3,
        "triangular contour"
    );
}

/// Invariant: an empty structure set label round-trips as empty string.
#[test]
fn test_write_rt_struct_empty_label_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("empty_label.dcm");

    let ss = RtStructureSet {
        structure_set_label: String::new(),
        structure_set_name: None,
        rois: vec![],
    };

    write_rt_struct(&path, &ss).expect("write_rt_struct empty");
    let result = read_rt_struct(&path).expect("read_rt_struct empty");

    assert!(
        result.structure_set_label.is_empty(),
        "empty label preserved"
    );
    assert!(result.rois.is_empty(), "no ROIs");
    assert!(result.structure_set_name.is_none(), "name is None");
}

/// Invariant: a POINT-type contour (single control point) round-trips.
#[test]
fn test_write_rt_struct_point_contour_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("point.dcm");

    let ss = RtStructureSet {
        structure_set_label: "Point".into(),
        structure_set_name: None,
        rois: vec![RtRoiInfo {
            roi_number: 1,
            roi_name: "Fiducial".into(),
            roi_description: None,
            roi_interpreted_type: None,
            display_color: None,
            contours: vec![RtContour {
                geometric_type: ArrayString::from("POINT").unwrap(),
                points: vec![[42.5, -13.2, 7.0]],
            }],
        }],
    };

    write_rt_struct(&path, &ss).expect("write_rt_struct point");
    let result = read_rt_struct(&path).expect("read_rt_struct point");

    assert_eq!(result.rois.len(), 1, "one ROI");
    assert_eq!(result.rois[0].contours.len(), 1, "one contour");
    assert_eq!(
        result.rois[0].contours[0].geometric_type.as_str(),
        "POINT",
        "geo type"
    );
    assert_eq!(result.rois[0].contours[0].points.len(), 1, "one point");
    assert!(
        (result.rois[0].contours[0].points[0][0] - 42.5).abs() < 1e-6,
        "x preserved"
    );
    assert!(
        (result.rois[0].contours[0].points[0][1] - (-13.2)).abs() < 1e-6,
        "y preserved"
    );
    assert!(
        (result.rois[0].contours[0].points[0][2] - 7.0).abs() < 1e-6,
        "z preserved"
    );
}
