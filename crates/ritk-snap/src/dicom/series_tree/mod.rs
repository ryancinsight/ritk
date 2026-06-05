//! Hierarchical DICOM series browser model.
//!
//! # Data model
//!
//! DICOM studies are organised as a three-level hierarchy:
//!
//! ```text
//! SeriesTree
//! └── PatientNode (keyed by patient_id)
//!     └── StudyNode (keyed by study_uid or study_date)
//!         └── SeriesEntry (one per DICOM series folder)
//! ```
//!
//! [`SeriesTree::from_entries`] builds this hierarchy from a flat
//! `Vec<SeriesEntry>`. Patients with the same `patient_id` are merged;
//! studies with the same `study_uid` within a patient are merged.
//!
//! # Invariants
//! - [`SeriesTree::total_series`] equals the number of entries passed to
//!   [`SeriesTree::from_entries`].
//! - [`SeriesTree::find_by_folder`] returns `Some` for every folder that
//!   appears in any `SeriesEntry` stored in the tree.

use ritk_io::DicomSeriesInfo;
use std::path::{Path, PathBuf};

// ── SeriesEntry ───────────────────────────────────────────────────────────────

/// A single DICOM series as displayed in the series browser.
#[derive(Debug, Clone)]
pub struct SeriesEntry {
    /// Series Instance UID (may be empty when absent from metadata).
    pub series_uid: String,
    /// Absolute path to the folder containing the DICOM slice files.
    pub folder: PathBuf,
    /// Patient name extracted from series metadata.
    pub patient_name: String,
    /// Patient ID extracted from series metadata.
    pub patient_id: String,
    /// DICOM modality string (e.g. `"CT"`, `"MR"`, `"PT"`).
    pub modality: String,
    /// Series description from tag (0008,103E).
    pub series_description: String,
    /// Number of image slices in the series.
    pub num_slices: usize,
    /// Study date in `YYYYMMDD` format, if present.
    pub study_date: Option<String>,
    /// Study Instance UID, if present.
    pub study_uid: Option<String>,
}

impl SeriesEntry {
    /// Construct a [`SeriesEntry`] from a [`DicomSeriesInfo`] returned by
    /// `ritk_io::scan_dicom_directory`.
    ///
    /// The current public `ritk_io::scan_dicom_directory` surface exposes
    /// series-level identifiers plus file paths. Fields not present in that
    /// summary stay empty until the full series is loaded.
    pub fn from_dicom_series_info(info: DicomSeriesInfo) -> Self {
        let folder = info
            .file_paths
            .first()
            .and_then(|path| path.parent())
            .map(Path::to_path_buf)
            .unwrap_or_default();
        Self {
            series_uid: info.series_instance_uid.to_string(),
            folder,
            patient_name: String::new(),
            patient_id: info.patient_id,
            modality: info.modality.to_string(),
            series_description: info.series_description,
            num_slices: info.file_paths.len(),
            study_date: None,
            study_uid: None,
        }
    }

    /// Short display label used in the series browser tree.
    ///
    /// Format: `"<icon> [<modality>] <description> (<n> slices)"`.
    /// Falls back to the folder name when `series_description` is empty.
    pub fn display_label(&self) -> String {
        let icon = self.modality_icon();
        let desc = if self.series_description.is_empty() {
            self.folder
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("(unknown)")
                .to_string()
        } else {
            self.series_description.clone()
        };
        let mod_tag = if self.modality.is_empty() {
            String::new()
        } else {
            format!("[{}] ", self.modality)
        };
        format!("{icon} {mod_tag}{desc} ({} slices)", self.num_slices)
    }

    /// Short modality icon string (emoji or abbreviated text) for display.
    ///
    /// | Modality prefix | Icon |
    /// |-----------------|------|
    /// | CT              | 🫁   |
    /// | MR              | 🧠   |
    /// | PT              | ☢    |
    /// | NM              | ☢    |
    /// | US              | 〰   |
    /// | CR / DR / DX    | 📷   |
    /// | MG              | 🎗   |
    /// | XA              | 💉   |
    /// | RF              | 📡   |
    /// | (other / empty) | 🗂   |
    pub fn modality_icon(&self) -> &'static str {
        match self.modality.to_uppercase().as_str() {
            "CT" => "🫁",
            "MR" => "🧠",
            "PT" | "NM" => "☢",
            "US" => "〰",
            "CR" | "DR" | "DX" => "📷",
            "MG" => "🎗",
            "XA" => "💉",
            "RF" => "📡",
            _ => "🗂",
        }
    }
}

// ── StudyNode ─────────────────────────────────────────────────────────────────

/// One study within a patient, containing one or more series.
#[derive(Debug, Clone)]
pub struct StudyNode {
    /// Study Instance UID — `None` when absent from metadata.
    pub study_uid: Option<String>,
    /// Study date in `YYYYMMDD` format — `None` when absent.
    pub study_date: Option<String>,
    /// Series belonging to this study, in insertion order.
    pub series: Vec<SeriesEntry>,
}

impl StudyNode {
    /// Canonical grouping key for deduplication inside a patient.
    ///
    /// Prefers the Study UID when available; falls back to the study date;
    /// ultimately falls back to an empty string so studies without either
    /// field still collapse into a single node.
    fn key(&self) -> &str {
        self.study_uid
            .as_deref()
            .or(self.study_date.as_deref())
            .unwrap_or("")
    }
}

// ── PatientNode ───────────────────────────────────────────────────────────────

/// One patient, containing one or more studies.
#[derive(Debug, Clone)]
pub struct PatientNode {
    /// Patient ID string from DICOM metadata.
    pub patient_id: String,
    /// Patient name string from DICOM metadata.
    pub patient_name: String,
    /// Studies belonging to this patient, in insertion order.
    pub studies: Vec<StudyNode>,
}

// ── SeriesTree ────────────────────────────────────────────────────────────────

/// Hierarchical patient → study → series tree for the series browser.
///
/// # Invariants
/// - Every [`SeriesEntry`] passed to [`SeriesTree::from_entries`] appears
///   exactly once in the tree.
/// - [`SeriesTree::total_series`] equals the number of entries provided.
/// - [`SeriesTree::find_by_folder`] is O(n) in the total number of series.
#[derive(Debug, Clone, Default)]
pub struct SeriesTree {
    /// Top-level patient nodes, in insertion order.
    pub patients: Vec<PatientNode>,
}

impl SeriesTree {
    /// Construct an empty tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the hierarchy from a flat list of [`SeriesEntry`] records.
    ///
    /// Grouping rules:
    /// 1. Entries with the same non-empty `patient_id` are placed in the
    ///    same [`PatientNode`]. Entries whose `patient_id` is empty are
    ///    each placed in their own anonymous patient node.
    /// 2. Within a patient, entries with the same non-empty `study_uid`
    ///    (or the same `study_date` when `study_uid` is absent) are placed
    ///    in the same [`StudyNode`].
    /// 3. Insertion order is preserved at every level.
    ///
    /// # Complexity
    /// O(n²) in the number of entries — acceptable for typical DICOM
    /// directory sizes (≤ a few thousand series).
    pub fn from_entries(entries: Vec<SeriesEntry>) -> Self {
        let mut tree = SeriesTree::new();
        for entry in entries {
            tree.insert(entry);
        }
        tree
    }

    /// Insert a single [`SeriesEntry`] into the correct position in the
    /// hierarchy, creating patient and study nodes as needed.
    fn insert(&mut self, entry: SeriesEntry) {
        // ── Patient lookup / creation ──────────────────────────────────────
        let patient_idx = if entry.patient_id.is_empty() {
            // Anonymous patients each get their own node.
            self.patients.push(PatientNode {
                patient_id: entry.patient_id.clone(),
                patient_name: entry.patient_name.clone(),
                studies: Vec::new(),
            });
            self.patients.len() - 1
        } else {
            match self
                .patients
                .iter()
                .position(|p| p.patient_id == entry.patient_id)
            {
                Some(i) => i,
                None => {
                    self.patients.push(PatientNode {
                        patient_id: entry.patient_id.clone(),
                        patient_name: entry.patient_name.clone(),
                        studies: Vec::new(),
                    });
                    self.patients.len() - 1
                }
            }
        };

        // ── Study lookup / creation within patient ─────────────────────────
        let entry_study_key: String = entry
            .study_uid
            .as_deref()
            .or(entry.study_date.as_deref())
            .unwrap_or("")
            .to_string();

        let patient = &mut self.patients[patient_idx];
        let study_idx = if entry_study_key.is_empty() {
            // No study key: create a new anonymous study node.
            patient.studies.push(StudyNode {
                study_uid: entry.study_uid.clone(),
                study_date: entry.study_date.clone(),
                series: Vec::new(),
            });
            patient.studies.len() - 1
        } else {
            match patient
                .studies
                .iter()
                .position(|s| s.key() == entry_study_key.as_str())
            {
                Some(i) => i,
                None => {
                    patient.studies.push(StudyNode {
                        study_uid: entry.study_uid.clone(),
                        study_date: entry.study_date.clone(),
                        series: Vec::new(),
                    });
                    patient.studies.len() - 1
                }
            }
        };

        patient.studies[study_idx].series.push(entry);
    }

    /// Total number of series stored across all patients and studies.
    ///
    /// # Postcondition
    /// Returns the same value as `entries.len()` for any list passed to
    /// [`SeriesTree::from_entries`].
    pub fn total_series(&self) -> usize {
        self.patients
            .iter()
            .flat_map(|p| p.studies.iter())
            .map(|s| s.series.len())
            .sum()
    }

    /// Find the first [`SeriesEntry`] whose `folder` equals `folder`.
    ///
    /// Returns `None` when no matching entry exists.
    pub fn find_by_folder(&self, folder: &Path) -> Option<&SeriesEntry> {
        self.patients
            .iter()
            .flat_map(|p| p.studies.iter())
            .flat_map(|s| s.series.iter())
            .find(|e| e.folder == folder)
    }

    /// Iterate over every [`SeriesEntry`] in the tree in insertion order.
    pub fn iter_series(&self) -> impl Iterator<Item = &SeriesEntry> {
        self.patients
            .iter()
            .flat_map(|p| p.studies.iter())
            .flat_map(|s| s.series.iter())
    }
}

#[cfg(test)]
mod tests;
