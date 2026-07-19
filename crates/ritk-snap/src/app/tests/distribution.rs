use super::*;
use crate::app::clinical_distribution::{
    build_clinical_distribution_report, report_path, summary_from_loaded_volume,
};
use crate::render::colormap::Colormap;
use crate::tools::kind::ToolKind;
use crate::ui::anatomical_label_for_axis;
use crate::{LoadedVolume, ViewerState};
use arrayvec::ArrayString;
use image::GenericImageView;
use std::path::PathBuf;
use std::sync::Arc;

#[test]
fn clinical_distribution_report_redacts_identifiers_and_lists_media_layout() {
    let mut volume = test_volume([2, 3, 4]);
    volume = LoadedVolume {
        data: Arc::new((0..24).map(|v| v as f32).collect()),
        shape: volume.shape,
        channels: 1,
        spacing: volume.spacing,
        origin: volume.origin,
        direction: volume.direction,
        metadata: None,
        source: Some(PathBuf::from("/secret/patient_a/study.dcm")),
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_name: Some("Jane Roe".to_owned()),
        patient_id: Some("ID-123".to_owned()),
        study_date: Some(ArrayString::from("20240517").unwrap()),
        series_description: Some("Abdomen".to_owned()),
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    };

    let mut viewer_state = ViewerState::default();
    viewer_state.slice_index = 1;
    viewer_state.window_center = Some(40.0);
    viewer_state.window_width = Some(400.0);

    let summary = summary_from_loaded_volume(
        &volume,
        &viewer_state,
        0,
        Colormap::Grayscale,
        ToolKind::WindowLevel,
        3,
        false,
        false,
        true,
    );
    let report = build_clinical_distribution_report(&summary);

    assert!(report.contains("Clinical Distribution Report"));
    let plane_label = anatomical_label_for_axis(Some(&volume), 0);
    assert!(report.contains(&format!("Current plane: {} (axis 0)", plane_label)));
    assert!(report.contains("Volume shape [depth, rows, cols]: 2 Ã— 3 Ã— 4"));
    assert!(report.contains("Window centre: 40.0000"));
    assert!(report.contains("Window width: 400.0000"));
    assert!(report.contains("Patient name: [redacted]"));
    assert!(report.contains("Patient ID: [redacted]"));
    assert!(report.contains("Study date: [redacted]"));
    assert!(report.contains("Series description: [redacted]"));
    assert!(report.contains("Source path: [redacted]"));
    assert!(!report.contains("Jane Roe"));
    assert!(!report.contains("ID-123"));
    assert!(!report.contains("/secret/patient_a/study.dcm"));
    assert!(report.contains("Segmentation: absent"));
    assert!(report.contains("RT-STRUCT overlay: absent"));
    assert!(report.contains("RT-DOSE overlay: present"));
    assert!(report.contains("media/current_slice.png"));
    assert!(report.contains("`media/axial/*.png` (2 files)"));
    assert!(report.contains("`media/coronal/*.png` (3 files)"));
    assert!(report.contains("`media/sagittal/*.png` (4 files)"));
}

#[test]
fn clinical_distribution_export_writes_report_and_media_with_expected_counts() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut app = SnapApp::default();
    app.axis = 0;
    app.viewer_state.slice_index = 1;
    app.viewer_state.window_center = Some(1.0);
    app.viewer_state.window_width = Some(2.0);
    app.active_tool = ToolKind::WindowLevel;
    app.colormap = Colormap::Grayscale;
    app.loaded = Some(LoadedVolume {
        data: Arc::new((0..24).map(|v| v as f32).collect()),
        shape: [2, 3, 4],
        channels: 1,
        spacing: [1.5, 0.75, 0.5],
        origin: [10.0, -2.0, 5.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metadata: None,
        source: Some(PathBuf::from("/secret/patient_a/study.dcm")),
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_name: Some("Jane Roe".to_owned()),
        patient_id: Some("ID-123".to_owned()),
        study_date: Some(ArrayString::from("20240517").unwrap()),
        series_description: Some("Abdomen".to_owned()),
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    });

    let summary = app
        .export_clinical_distribution_to(tmp.path())
        .expect("clinical distribution export");

    let expected_root = tmp.path().join("clinical_distribution");
    assert_eq!(summary.root, expected_root);
    assert!(summary.current_slice_written);
    assert_eq!(summary.mpr_written, 9);
    assert_eq!(summary.mpr_failed, 0);

    let report = std::fs::read_to_string(&summary.report_path).expect("report text");
    assert!(report.contains("Patient name: [redacted]"));
    assert!(!report.contains("Jane Roe"));
    assert!(report.contains("media/current_slice.png"));

    let current_slice = image::open(&summary.current_slice_path).expect("current slice png");
    assert_eq!(current_slice.dimensions(), (4, 3));

    let axial_dir = summary.mpr_root.join("axial");
    let coronal_dir = summary.mpr_root.join("coronal");
    let sagittal_dir = summary.mpr_root.join("sagittal");
    let axial_count = std::fs::read_dir(&axial_dir).expect("axial dir").count();
    let coronal_count = std::fs::read_dir(&coronal_dir)
        .expect("coronal dir")
        .count();
    let sagittal_count = std::fs::read_dir(&sagittal_dir)
        .expect("sagittal dir")
        .count();
    assert_eq!(axial_count, 2);
    assert_eq!(coronal_count, 3);
    assert_eq!(sagittal_count, 4);

    let report_path = report_path(&summary.root);
    assert_eq!(summary.report_path, report_path);
    assert!(
        report_path.is_file(),
        "report must exist at {}",
        report_path.display()
    );
}
