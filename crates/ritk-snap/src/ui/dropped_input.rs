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
    /// Load a pathless dropped DICOM byte batch as one assembled series.
    LoadDicomSeriesBytes {
        files: Vec<(String, std::sync::Arc<[u8]>)>,
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
    let mut dicom_bytes_batch: Vec<(String, std::sync::Arc<[u8]>)> = Vec::new();
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

        if let Some(bytes) = file.bytes.as_ref() {
            if is_likely_dicom_payload(&file.name, bytes) {
                let name = if file.name.is_empty() {
                    format!("dropped_{:04}.dcm", dicom_bytes_batch.len())
                } else {
                    file.name.clone()
                };
                dicom_bytes_batch.push((name, bytes.clone()));
                continue;
            }

            if first_supported_volume_bytes.is_none()
                && is_supported_volume_name_for_bytes(&file.name)
            {
                first_supported_volume_bytes = Some((file.name.clone(), bytes.clone()));
            }
        }
    }

    if let Some(path) = first_supported_volume_path {
        return DroppedInputAction::LoadVolume(path);
    }

    if !dicom_bytes_batch.is_empty() {
        return DroppedInputAction::LoadDicomSeriesBytes {
            files: dicom_bytes_batch,
        };
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

fn is_likely_dicom_payload(name: &str, bytes: &[u8]) -> bool {
    let n = name.to_ascii_lowercase();
    if n.ends_with(".dcm") || n.ends_with(".dicom") || n.ends_with(".ima") || n == "dicomdir" {
        return true;
    }
    bytes.len() >= 132 && &bytes[128..132] == b"DICM"
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
#[path = "tests_dropped_input.rs"]
mod tests;
