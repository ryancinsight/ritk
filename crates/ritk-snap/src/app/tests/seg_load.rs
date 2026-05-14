//! Segmentation DICOM-SEG loading tests (external fixtures).

use super::*;
use std::path::PathBuf;

#[test]
fn load_external_dcmqi_dicom_seg_into_snap_app() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 512, 512]));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("dcmqi")
        .join("liver.dcm");
    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );
    app.load_segmentation_dicom_seg_file(&path);
    let editor = app
        .label_editor
        .as_ref()
        .expect("label editor loaded from external SEG");
    let map = editor.current_map();
    assert_eq!(map.shape, [3, 512, 512]);
    assert!(map.present_labels().contains(&1));
    assert!(
        map.count_label(1) > 0,
        "external SEG must populate label 1 voxels"
    );
    assert_eq!(
        map.table.get_label(1).map(|e| e.name.as_str()),
        Some("Liver")
    );
    assert_eq!(
        app.status_message,
        format!("Loaded DICOM-SEG from {}", path.display())
    );
}

#[test]
fn load_external_dcmqi_partial_overlap_dicom_seg_into_snap_app() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 512, 512]));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("dcmqi")
        .join("partial_overlaps.dcm");
    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );
    app.load_segmentation_dicom_seg_file(&path);
    let editor = app
        .label_editor
        .as_ref()
        .expect("label editor loaded from external dcmqi partial-overlap SEG");
    let map = editor.current_map();
    assert_eq!(map.shape, [3, 512, 512]);
    let present = map.present_labels();
    for label in [1u32, 2, 3, 4, 5] {
        assert!(present.contains(&label), "label {label} must be present");
        assert!(map.count_label(label) > 0, "label {label} must have voxels");
    }
    assert_eq!(
        app.status_message,
        format!("Loaded DICOM-SEG from {}", path.display())
    );
}

#[test]
fn load_external_highdicom_overlap_dicom_seg_into_snap_app() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([4, 16, 16]));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("highdicom")
        .join("seg_image_ct_binary_overlap.dcm");
    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );
    app.load_segmentation_dicom_seg_file(&path);
    let editor = app
        .label_editor
        .as_ref()
        .expect("label editor loaded from external highdicom SEG");
    let map = editor.current_map();
    assert_eq!(map.shape, [4, 16, 16]);
    assert!(map.present_labels().contains(&1));
    assert!(map.present_labels().contains(&2));
    assert!(
        map.count_label(1) > 0,
        "segment 1 voxels must populate the viewer state"
    );
    assert!(
        map.count_label(2) > 0,
        "segment 2 voxels must populate the viewer state"
    );
    assert_eq!(
        map.table.get_label(1).map(|e| e.name.as_str()),
        Some("first segment")
    );
    assert_eq!(
        map.table.get_label(2).map(|e| e.name.as_str()),
        Some("second segment")
    );
    assert_eq!(
        app.status_message,
        format!("Loaded DICOM-SEG from {}", path.display())
    );
}

#[test]
fn load_external_highdicom_binary_dicom_seg_into_snap_app() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([3, 16, 16]));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("highdicom")
        .join("seg_image_ct_binary.dcm");
    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );
    app.load_segmentation_dicom_seg_file(&path);
    let editor = app
        .label_editor
        .as_ref()
        .expect("label editor loaded from external highdicom binary SEG");
    let map = editor.current_map();
    assert_eq!(map.shape, [3, 16, 16]);
    assert!(map.present_labels().contains(&1));
    assert!(
        map.count_label(1) > 0,
        "segment voxels must populate the viewer state"
    );
    assert_eq!(
        map.table.get_label(1).map(|e| e.name.as_str()),
        Some("first segment")
    );
    assert_eq!(
        app.status_message,
        format!("Loaded DICOM-SEG from {}", path.display())
    );
}

#[test]
fn load_external_rsna_dido_liver_dicom_seg_into_snap_app() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([34, 512, 512]));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("dicom_seg")
        .join("rsna_dido")
        .join("xTtzBC6F6p_rpexuszCnb_01_liver.dcm");
    assert!(
        path.is_file(),
        "external SEG fixture missing: {}",
        path.display()
    );
    app.load_segmentation_dicom_seg_file(&path);
    let editor = app
        .label_editor
        .as_ref()
        .expect("label editor loaded from external rsna dido SEG");
    let map = editor.current_map();
    assert_eq!(map.shape, [34, 512, 512]);
    assert!(map.present_labels().contains(&1));
    assert!(
        map.count_label(1) > 0,
        "segment voxels must populate the viewer state"
    );
    assert_eq!(
        map.table.get_label(1).map(|e| e.name.as_str()),
        Some("liver")
    );
    assert_eq!(
        app.status_message,
        format!("Loaded DICOM-SEG from {}", path.display())
    );
}
