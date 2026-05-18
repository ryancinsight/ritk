use super::*;

/// Invariant: a single CLOSED_PLANAR contour (unit square, 4 points) must
/// produce exactly 1 polygon cell containing 4 point indices, with no lines or vertices.
#[test]
fn test_rt_roi_to_polydata_closed_planar() {
    let roi = RtRoiInfo {
        roi_number: 1,
        roi_name: "GTV".to_string(),
        roi_description: None,
        roi_interpreted_type: None,
        display_color: None,
        contours: vec![RtContour {
            geometric_type: "CLOSED_PLANAR".to_string(),
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        }],
    };

    let poly = rt_roi_to_polydata(&roi);

    assert_eq!(poly.points.len(), 4, "poly.points must contain 4 points");
    assert_eq!(poly.polygons.len(), 1, "must have exactly 1 polygon");
    assert_eq!(
        poly.polygons[0].len(),
        4,
        "polygon cell must reference 4 indices"
    );
    assert_eq!(poly.lines.len(), 0, "lines must be empty");
    assert_eq!(poly.vertices.len(), 0, "vertices must be empty");
    assert_eq!(
        poly.polygons[0],
        vec![0, 1, 2, 3],
        "polygon indices must be [0,1,2,3]"
    );
    assert_eq!(poly.points[0], [0.0_f32, 0.0, 0.0], "point[0]");
    assert_eq!(poly.points[2], [1.0_f32, 1.0, 0.0], "point[2]");
}

/// Invariant: a single OPEN_PLANAR contour (3 points) must produce exactly
/// 1 line cell and zero polygon or vertex cells.
#[test]
fn test_rt_roi_to_polydata_open_planar() {
    let roi = RtRoiInfo {
        roi_number: 2,
        roi_name: "Wire".to_string(),
        roi_description: None,
        roi_interpreted_type: None,
        display_color: None,
        contours: vec![RtContour {
            geometric_type: "OPEN_PLANAR".to_string(),
            points: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0]],
        }],
    };

    let poly = rt_roi_to_polydata(&roi);

    assert_eq!(poly.points.len(), 3, "poly.points must contain 3 points");
    assert_eq!(poly.lines.len(), 1, "must have exactly 1 line");
    assert_eq!(poly.lines[0].len(), 3, "line cell must reference 3 indices");
    assert_eq!(poly.polygons.len(), 0, "polygons must be empty");
    assert_eq!(poly.vertices.len(), 0, "vertices must be empty");
    assert_eq!(poly.lines[0], vec![0, 1, 2], "line indices must be [0,1,2]");
}

/// Invariant: an ROI with 1 CLOSED_PLANAR + 1 OPEN_PLANAR contour must
/// populate both poly.polygons and poly.lines, with correct running offsets.
#[test]
fn test_rt_roi_to_polydata_mixed_contours() {
    let roi = RtRoiInfo {
        roi_number: 3,
        roi_name: "Mixed".to_string(),
        roi_description: None,
        roi_interpreted_type: None,
        display_color: None,
        contours: vec![
            RtContour {
                geometric_type: "CLOSED_PLANAR".to_string(),
                points: vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            },
            RtContour {
                geometric_type: "OPEN_PLANAR".to_string(),
                points: vec![[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [7.0, 1.0, 0.0]],
            },
        ],
    };

    let poly = rt_roi_to_polydata(&roi);

    assert_eq!(
        poly.points.len(),
        7,
        "poly.points must contain 7 total points"
    );
    assert_eq!(poly.polygons.len(), 1, "must have 1 polygon");
    assert_eq!(poly.polygons[0], vec![0, 1, 2, 3], "polygon indices");
    assert_eq!(poly.lines.len(), 1, "must have 1 line");
    assert_eq!(poly.lines[0], vec![4, 5, 6], "line indices must be [4,5,6]");
    assert_eq!(poly.vertices.len(), 0, "vertices must be empty");
    assert_eq!(
        poly.points[4],
        [5.0_f32, 0.0, 0.0],
        "point[4] from OPEN_PLANAR"
    );
}
