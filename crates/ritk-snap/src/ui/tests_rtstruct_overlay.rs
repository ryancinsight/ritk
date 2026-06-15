use super::*;
use ritk_io::{RtContour, RtRoiInfo, RtRoiInterpretedType, RtStructureSet};

fn make_rt() -> RtStructureSet {
    RtStructureSet {
        structure_set_label: "SS".to_string(),
        structure_set_name: Some("Name".to_string()),
        rois: vec![RtRoiInfo {
            roi_number: 1,
            roi_name: "PTV".to_string(),
            roi_description: None,
            roi_interpreted_type: Some(RtRoiInterpretedType::from_dicom_str("PTV")),
            display_color: Some([255, 0, 0]),
            contours: vec![RtContour {
                geometric_type: ContourGeometricType::ClosedPlanar,
                points: vec![
                    [2.0, 3.0, 4.0],
                    [2.0, 5.0, 4.0],
                    [2.0, 5.0, 6.0],
                    [2.0, 3.0, 6.0],
                ],
            }],
        }],
    }
}

#[test]
fn projects_identity_axial_contour_on_matching_slice() {
    let rt = make_rt();
    let projected = project_rt_struct_contours_for_slice(
        &rt,
        0,
        2,
        [10, 10, 10],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    );
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0].color, [255, 0, 0]);
    assert!(projected[0].closed);
    assert_eq!(projected[0].points_row_col[0], [3.0, 4.0]);
}

#[test]
fn filters_contour_when_slice_does_not_match() {
    let rt = make_rt();
    let projected = project_rt_struct_contours_for_slice(
        &rt,
        0,
        7,
        [10, 10, 10],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    );
    assert!(projected.is_empty());
}

#[test]
fn uses_default_color_when_roi_display_color_missing() {
    let mut rt = make_rt();
    rt.rois[0].display_color = None;
    let projected = project_rt_struct_contours_for_slice(
        &rt,
        0,
        2,
        [10, 10, 10],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    );
    assert_eq!(projected[0].color, [255, 255, 0]);
}

#[test]
fn returns_empty_when_transform_is_singular() {
    let rt = make_rt();
    let projected = project_rt_struct_contours_for_slice(
        &rt,
        0,
        2,
        [10, 10, 10],
        [0.0, 0.0, 0.0],
        [0.0; 9],
        [1.0, 1.0, 1.0],
    );
    assert!(projected.is_empty());
}
