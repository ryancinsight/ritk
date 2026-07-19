use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{decide_dropped_input_action, DroppedInputAction};

fn dropped_with_path(path: &str) -> egui::DroppedFile {
    egui::DroppedFile {
        path: Some(PathBuf::from(path)),
        ..Default::default()
    }
}

fn dropped_pathless_named(name: &str) -> egui::DroppedFile {
    egui::DroppedFile {
        name: name.to_owned(),
        ..Default::default()
    }
}

fn dropped_pathless_named_with_bytes(name: &str, bytes: Vec<u8>) -> egui::DroppedFile {
    egui::DroppedFile {
        name: name.to_owned(),
        bytes: Some(std::sync::Arc::<[u8]>::from(bytes)),
        ..Default::default()
    }
}

fn create_temp_dicom_file() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX_EPOCH")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ritk_snap_drop_test_{unique}"));
    fs::create_dir_all(&dir).expect("create temporary test directory");
    let path = dir.join("slice.dcm");
    fs::write(&path, [0u8; 1]).expect("create temporary dicom file");
    path
}

#[test]
fn empty_drop_batch_returns_none() {
    let action = decide_dropped_input_action(&[]);
    assert_eq!(action, DroppedInputAction::None);
}

#[test]
fn dicom_path_has_priority_over_volume_path() {
    let dicom_path = create_temp_dicom_file();
    let files = vec![
        dropped_with_path("study.nii.gz"),
        dropped_with_path(&dicom_path.to_string_lossy()),
    ];

    let action = decide_dropped_input_action(&files);
    match action {
        DroppedInputAction::QueueDicom(path) => {
            assert_eq!(path, dicom_path);
        }
        other => panic!("expected QueueDicom, got {other:?}"),
    }
}

#[test]
fn supported_volume_path_is_loaded_when_no_dicom_is_present() {
    let files = vec![dropped_with_path("study.nrrd")];
    let action = decide_dropped_input_action(&files);
    assert_eq!(
        action,
        DroppedInputAction::LoadVolume(PathBuf::from("study.nrrd"))
    );
}

#[test]
fn pathless_drop_returns_named_guidance_message() {
    let files = vec![dropped_pathless_named("browser-file.dcm")];
    let action = decide_dropped_input_action(&files);

    match action {
        DroppedInputAction::Message(msg) => {
            assert!(msg.contains("browser-file.dcm"));
            assert!(msg.contains("no filesystem path"));
        }
        other => panic!("expected Message, got {other:?}"),
    }
}

#[test]
fn pathless_nifti_with_bytes_routes_to_in_memory_load() {
    let files = vec![dropped_pathless_named_with_bytes(
        "dropped.nii",
        vec![1, 2, 3],
    )];
    let action = decide_dropped_input_action(&files);

    match action {
        DroppedInputAction::LoadVolumeBytes { name, bytes } => {
            assert_eq!(name, "dropped.nii");
            assert_eq!(bytes.as_ref(), [1, 2, 3]);
        }
        other => panic!("expected LoadVolumeBytes, got {other:?}"),
    }
}

#[test]
fn pathless_dicom_with_bytes_routes_to_dicom_batch_load() {
    let mut bytes = vec![0_u8; 140];
    bytes[128..132].copy_from_slice(b"DICM");
    let files = vec![dropped_pathless_named_with_bytes("slice_001", bytes)];
    let action = decide_dropped_input_action(&files);

    match action {
        DroppedInputAction::LoadDicomSeriesBytes { files } => {
            assert_eq!(files.len(), 1);
            assert_eq!(files[0].0, "slice_001");
        }
        other => panic!("expected LoadDicomSeriesBytes, got {other:?}"),
    }
}

#[test]
fn pathless_dicom_bytes_prefer_dicom_over_nifti_bytes() {
    let mut dicom_bytes = vec![0_u8; 140];
    dicom_bytes[128..132].copy_from_slice(b"DICM");
    let files = vec![
        dropped_pathless_named_with_bytes("dropped.nii", vec![1, 2, 3]),
        dropped_pathless_named_with_bytes("slice_001", dicom_bytes),
    ];
    let action = decide_dropped_input_action(&files);

    assert!(
        matches!(action, DroppedInputAction::LoadDicomSeriesBytes { .. }),
        "expected DICOM byte batch to take precedence when present"
    );
}
