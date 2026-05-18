use super::*;
use ritk_io::{RtContour, RtDoseGrid, RtRoiInfo, RtStructureSet};

fn square_roi() -> RtStructureSet {
    RtStructureSet {
        structure_set_label: "RT".to_owned(),
        structure_set_name: None,
        rois: vec![RtRoiInfo {
            roi_number: 1,
            roi_name: "PTV".to_owned(),
            roi_description: None,
            roi_interpreted_type: Some("PTV".to_owned()),
            display_color: Some([255, 0, 0]),
            contours: vec![RtContour {
                geometric_type: "CLOSED_PLANAR".to_owned(),
                points: vec![
                    [0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 2.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
            }],
        }],
    }
}

fn uniform_dose_3x3() -> RtDoseGrid {
    RtDoseGrid {
        rows: 3,
        cols: 3,
        n_frames: 1,
        dose_type: "PHYSICAL".to_owned(),
        dose_summation_type: "PLAN".to_owned(),
        dose_grid_scaling: 1.0,
        frame_offsets: vec![0.0],
        dose_gy: vec![2.0; 9],
        image_position: Some([0.0, 0.0, 0.0]),
        image_orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        pixel_spacing: Some([1.0, 1.0]),
        referenced_rt_plan_sop_instance_uid: None,
    }
}

#[test]
fn point_in_polygon_square() {
    let poly = vec![[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]];
    assert!(point_in_polygon(1.5, 1.5, &poly));
    assert!(!point_in_polygon(4.0, 4.0, &poly));
}

#[test]
fn compute_roi_dvh_uniform_dose_is_step_like() {
    let analytics = compute_roi_dose_analytics(
        &square_roi(),
        &uniform_dose_3x3(),
        1,
        [1, 3, 3],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        10,
    )
    .expect("analytics");

    assert!(analytics.voxel_count > 0);
    assert!((analytics.mean_dose_gy - 2.0).abs() < 1e-6);
    assert!((analytics.max_dose_gy - 2.0).abs() < 1e-6);
    assert!((analytics.min_dose_gy - 2.0).abs() < 1e-6);
    assert_eq!(
        analytics.curve.first().map(|p| p.volume_fraction_ge),
        Some(1.0)
    );
    assert_eq!(
        analytics.curve.last().map(|p| p.volume_fraction_ge),
        Some(1.0)
    );
}

#[test]
fn compute_roi_dvh_missing_roi_returns_none() {
    let result = compute_roi_dose_analytics(
        &square_roi(),
        &uniform_dose_3x3(),
        99,
        [1, 3, 3],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        10,
    );
    assert!(result.is_none());
}

#[test]
fn select_nth_smallest_returns_expected_rank_value() {
    let mut data = vec![9.0_f32, 1.0, 5.0, 7.0, 3.0];
    let value = select_nth_smallest(&mut data, 1).expect("rank value");
    assert_eq!(value, 3.0);
}

#[test]
fn build_dvh_curve_histogram_monotonic_volume_fraction() {
    let samples = vec![1.0_f32, 2.0, 2.0, 4.0];
    let curve = build_dvh_curve_histogram(&samples, 4.0, 4);
    assert_eq!(curve.len(), 5);
    for pair in curve.windows(2) {
        assert!(pair[0].volume_fraction_ge >= pair[1].volume_fraction_ge);
    }
    assert_eq!(curve[0].volume_fraction_ge, 1.0);
}
