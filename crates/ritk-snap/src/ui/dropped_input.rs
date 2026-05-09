//! Dropped-input routing policy for viewer ingestion.
//!
//! This module is the SSOT for deciding how a batch of dropped files should be
//! handled by the app shell.

use std::path::{Path, PathBuf};

/// Deterministic action chosen from an egui dropped-file batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DroppedInputAction {
    /// No actionable dropped input.
    None,
    /// Queue a DICOM input path for series scan and deferred load.
    QueueDicom(PathBuf),
    /// Load a non-DICOM medical volume path immediately.
    LoadVolume(PathBuf),
    /// Load a pathless dropped medical payload from in-memory bytes.
    LoadVolumeBytes {
        name: String,
        bytes: std::sync::Arc<[u8]>,
    },
    /// Show deterministic user guidance in the status line.
    Message(String),
}

/// Decide one deterministic action for a dropped-file batch.
///
/// Priority order:
/// 1. First DICOM-classified path.
/// 2. First supported non-DICOM medical-image path.
/// 3. Pathless drop guidance message.
/// 4. None.
pub fn decide_dropped_input_action(files: &[egui::DroppedFile]) -> DroppedInputAction {
    if files.is_empty() {
        return DroppedInputAction::None;
    }

    let mut first_supported_volume_path: Option<PathBuf> = None;
    let mut first_supported_volume_bytes: Option<(String, std::sync::Arc<[u8]>)> = None;
    let mut first_pathless_name: Option<String> = None;
    let mut saw_pathless = false;

    for file in files {
        if let Some(path) = file.path.as_ref() {
            if crate::dicom::classify_dicom_input_path(path)
                .dicom_root()
                .is_some()
            {
                return DroppedInputAction::QueueDicom(path.clone());
            }

            if first_supported_volume_path.is_none() && is_supported_volume_path(path) {
                first_supported_volume_path = Some(path.clone());
            }

            continue;
        }

        saw_pathless = true;
        if first_pathless_name.is_none() && !file.name.is_empty() {
            first_pathless_name = Some(file.name.clone());
        }

        if first_supported_volume_bytes.is_none()
            && is_supported_volume_name_for_bytes(&file.name)
            && file.bytes.is_some()
        {
            first_supported_volume_bytes = Some((
                file.name.clone(),
                file.bytes
                    .as_ref()
                    .expect("checked is_some above")
                    .clone(),
            ));
        }
    }

    if let Some(path) = first_supported_volume_path {
        return DroppedInputAction::LoadVolume(path);
    }

    if let Some((name, bytes)) = first_supported_volume_bytes {
        return DroppedInputAction::LoadVolumeBytes { name, bytes };
    }

    if saw_pathless {
        if let Some(name) = first_pathless_name {
            return DroppedInputAction::Message(format!(
                "Dropped '{}' has no filesystem path in this build; use File -> Open for now.",
                name
            ));
        }

        return DroppedInputAction::Message(
            "Dropped file has no filesystem path in this build; use File -> Open for now."
                .to_owned(),
        );
    }

    DroppedInputAction::None
}

fn is_supported_volume_name_for_bytes(name: &str) -> bool {
    let s = name.to_ascii_lowercase();
    s.ends_with(".nii") || s.ends_with(".nii.gz")
}

fn is_supported_volume_path(path: &Path) -> bool {
    let s = path.to_string_lossy().to_lowercase();
    s.ends_with(".nii")
        || s.ends_with(".nii.gz")
        || s.ends_with(".mha")
        || s.ends_with(".mhd")
        || s.ends_with(".nrrd")
        || s.ends_with(".nhdr")
        || s.ends_with(".mgh")
        || s.ends_with(".mgz")
}

#[cfg(test)]
mod tests {
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
        let files = vec![dropped_pathless_named_with_bytes("dropped.nii", vec![1, 2, 3])];
        let action = decide_dropped_input_action(&files);

        match action {
            DroppedInputAction::LoadVolumeBytes { name, bytes } => {
                assert_eq!(name, "dropped.nii");
                assert_eq!(bytes.as_ref(), [1, 2, 3]);
            }
            other => panic!("expected LoadVolumeBytes, got {other:?}"),
        }
    }
}
