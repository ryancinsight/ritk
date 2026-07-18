//! RT dose/plan and viewer metadata tests.

use super::*;
use arrayvec::ArrayString;

#[test]
fn load_rt_plan_file_sets_plan_summary_state() {
    let mut app = SnapApp::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("plan.dcm");
    let plan = ritk_io::RtPlanInfo {
        sop_instance_uid: ArrayString::new(),
        rt_plan_label: "PLAN_A".to_owned(),
        rt_plan_name: "Plan A".to_owned(),
        rt_plan_description: "Synthetic plan".to_owned(),
        plan_intent: ArrayString::from("CURATIVE").expect("valid fixed plan intent"),
        beams: vec![ritk_io::RtBeamInfo {
            beam_number: 1,
            beam_name: "BEAM_1".to_owned(),
            beam_description: "Beam one".to_owned(),
            radiation_type: ArrayString::from("PHOTON").expect("valid fixed radiation type"),
            treatment_delivery_type: ArrayString::from("TREATMENT")
                .expect("valid fixed delivery type"),
            n_control_points: 2 }],
        fraction_groups: vec![ritk_io::RtFractionGroup {
            fraction_group_number: 1,
            n_fractions_planned: 30,
            referenced_beam_numbers: vec![1] }] };
    ritk_io::write_rt_plan(&path, &plan).expect("write rt plan");
    app.load_rt_plan_file(path);
    let loaded = app.rt_plan.as_ref().expect("rt plan loaded");
    assert_eq!(loaded.rt_plan_label, "PLAN_A");
    assert_eq!(loaded.beams.len(), 1);
    assert_eq!(loaded.fraction_groups.len(), 1);
    assert!(app.status_message.contains("Loaded RT-PLAN PLAN_A"));
}

#[test]
fn rt_dose_plan_link_status_reports_linked_uid() {
    let mut app = SnapApp::default();
    app.rt_plan = Some(ritk_io::RtPlanInfo {
        sop_instance_uid: ArrayString::from("2.25.9001").expect("valid fixed uid"),
        rt_plan_label: "PLAN_LINK".to_owned(),
        rt_plan_name: String::new(),
        rt_plan_description: String::new(),
        plan_intent: ArrayString::new(),
        beams: vec![],
        fraction_groups: vec![] });
    app.rt_dose = Some(ritk_io::RtDoseGrid {
        rows: 1,
        cols: 1,
        n_frames: 1,
        dose_type: ritk_io::RtDoseType::Physical,
        dose_summation_type: ritk_io::RtDoseSummationType::Plan,
        dose_grid_scaling: 1.0,
        frame_offsets: vec![0.0],
        dose_gy: vec![1.0],
        image_position: None,
        image_orientation: None,
        pixel_spacing: None,
        referenced_rt_plan_sop_instance_uid: Some(
            ArrayString::from("2.25.9001").expect("valid fixed uid"),
        ) });

    let message = app.rt_dose_plan_link_status().expect("link status present");
    assert!(message.contains("linked to loaded RT-PLAN UID 2.25.9001"));
}

#[test]
fn geometry_summary_preserves_dicom_geometry() {
    let metadata = ritk_io::DicomReadMetadata {
        dimensions: [3, 4, 5],
        spacing: [0.8, 0.8, 1.5],
        origin: [10.0, 20.0, -30.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ..Default::default()
    };

    let summary = crate::GeometrySummary::from_dicom(&metadata);

    assert_eq!(summary.dimensions, metadata.dimensions);
    assert_eq!(summary.spacing, metadata.spacing);
    assert_eq!(summary.origin, metadata.origin);
    assert_eq!(summary.direction, metadata.direction);
}

#[test]
fn viewer_status_retains_message() {
    let status = crate::ViewerStatus::new("loaded");
    assert_eq!(status.message, "loaded");
}
