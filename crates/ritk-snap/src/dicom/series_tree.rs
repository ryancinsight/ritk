//! Hierarchical DICOM series browser model.
//!
//! # Data model
//!
//! DICOM studies are organised as a three-level hierarchy:
//!
//! ```text
//! SeriesTree
//! └── PatientNode  (keyed by patient_id)
//!     └── StudyNode  (keyed by study_uid or study_date)
//!         └── SeriesEntry  (one per DICOM series folder)
//! ```
//!
//! [`SeriesTree::from_entries`] builds this hierarchy from a flat
//! `Vec<SeriesEntry>`.  Patients with the same `patient_id` are merged;
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
    /// All optional metadata fields are propagated from
    /// `info.metadata`; absent values fall back to empty strings or `None`.
    pub fn from_dicom_series_info(info: DicomSeriesInfo) -> Self {
        let m = &info.metadata;
        Self {
            series_uid: m.series_instance_uid.clone().unwrap_or_default(),
            folder: info.path.clone(),
            patient_name: m.patient_name.clone().unwrap_or_default(),
            patient_id: m.patient_id.clone().unwrap_or_default(),
            modality: m.modality.clone().unwrap_or_default(),
            series_description: m.series_description.clone().unwrap_or_default(),
            num_slices: info.num_slices,
            study_date: m.study_date.clone(),
            study_uid: m.study_instance_uid.clone(),
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
    /// | MG              | 🎗    |
    /// | XA              | 💉   |
    /// | RF              | 📡   |
    /// | (other / empty) | 🗂    |
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
    ///    same [`PatientNode`].  Entries whose `patient_id` is empty are
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic `SeriesEntry` for testing purposes.
    fn make_entry(
        patient_id: &str,
        patient_name: &str,
        study_uid: Option<&str>,
        study_date: Option<&str>,
        series_uid: &str,
        folder: &str,
        modality: &str,
        num_slices: usize,
    ) -> SeriesEntry {
        SeriesEntry {
            series_uid: series_uid.to_string(),
            folder: PathBuf::from(folder),
            patient_name: patient_name.to_string(),
            patient_id: patient_id.to_string(),
            modality: modality.to_string(),
            series_description: format!("{modality} series"),
            num_slices,
            study_date: study_date.map(str::to_string),
            study_uid: study_uid.map(str::to_string),
        }
    }

    /// Three series across two patients must produce exactly two patient nodes.
    ///
    /// Patient A has two series in the same study; patient B has one series.
    /// Postcondition: `tree.patients.len() == 2`, `total_series() == 3`.
    #[test]
    fn test_from_entries_groups_by_patient() {
        let entries = vec![
            make_entry(
                "P001",
                "Alice",
                Some("ST1"),
                Some("20230101"),
                "S1",
                "/a/s1",
                "CT",
                50,
            ),
            make_entry(
                "P001",
                "Alice",
                Some("ST1"),
                Some("20230101"),
                "S2",
                "/a/s2",
                "CT",
                30,
            ),
            make_entry(
                "P002",
                "Bob",
                Some("ST2"),
                Some("20230202"),
                "S3",
                "/b/s1",
                "MR",
                20,
            ),
        ];
        let tree = SeriesTree::from_entries(entries);

        assert_eq!(
            tree.patients.len(),
            2,
            "two distinct patient IDs must produce two PatientNodes"
        );
        assert_eq!(
            tree.total_series(),
            3,
            "total_series() must equal the number of input entries"
        );

        // Patient A must have exactly one study with two series.
        let alice = tree
            .patients
            .iter()
            .find(|p| p.patient_id == "P001")
            .unwrap();
        assert_eq!(alice.studies.len(), 1, "Alice must have one study");
        assert_eq!(
            alice.studies[0].series.len(),
            2,
            "Alice's study must contain both CT series"
        );

        // Patient B must have exactly one study with one series.
        let bob = tree
            .patients
            .iter()
            .find(|p| p.patient_id == "P002")
            .unwrap();
        assert_eq!(bob.studies.len(), 1, "Bob must have one study");
        assert_eq!(
            bob.studies[0].series.len(),
            1,
            "Bob's study must contain exactly one MR series"
        );
    }

    /// `total_series()` must return the exact number of entries inserted.
    ///
    /// Tested with five entries spanning three patients to exercise the
    /// summation path across non-trivial tree depth.
    #[test]
    fn test_total_series_count() {
        let entries = vec![
            make_entry("P1", "Alice", Some("ST1"), None, "S1", "/p1/s1", "CT", 10),
            make_entry("P1", "Alice", Some("ST1"), None, "S2", "/p1/s2", "CT", 20),
            make_entry("P2", "Bob", Some("ST2"), None, "S3", "/p2/s1", "MR", 15),
            make_entry("P3", "Carol", Some("ST3"), None, "S4", "/p3/s1", "PT", 60),
            make_entry("P3", "Carol", Some("ST3"), None, "S5", "/p3/s2", "PT", 60),
        ];
        let tree = SeriesTree::from_entries(entries);
        assert_eq!(
            tree.total_series(),
            5,
            "total_series() must equal 5 for five distinct entries"
        );
    }

    /// `from_entries` with an empty input must produce an empty tree with
    /// zero patients and `total_series() == 0`.
    #[test]
    fn test_from_entries_empty_input() {
        let tree = SeriesTree::from_entries(vec![]);
        assert_eq!(
            tree.patients.len(),
            0,
            "empty input must produce zero patient nodes"
        );
        assert_eq!(
            tree.total_series(),
            0,
            "total_series() must be 0 for empty input"
        );
    }

    /// `find_by_folder` must locate an entry by its exact folder path.
    #[test]
    fn test_find_by_folder_found() {
        let entries = vec![
            make_entry(
                "P1",
                "Alice",
                Some("ST1"),
                None,
                "S1",
                "/data/scan1",
                "CT",
                50,
            ),
            make_entry(
                "P1",
                "Alice",
                Some("ST1"),
                None,
                "S2",
                "/data/scan2",
                "MR",
                30,
            ),
        ];
        let tree = SeriesTree::from_entries(entries);

        let found = tree.find_by_folder(Path::new("/data/scan2"));
        assert!(found.is_some(), "find_by_folder must find '/data/scan2'");
        assert_eq!(
            found.unwrap().series_uid,
            "S2",
            "found entry must be the MR series with uid S2"
        );
    }

    /// `find_by_folder` must return `None` for a path not in the tree.
    #[test]
    fn test_find_by_folder_not_found() {
        let tree = SeriesTree::from_entries(vec![make_entry(
            "P1",
            "Alice",
            Some("ST1"),
            None,
            "S1",
            "/data/s1",
            "CT",
            10,
        )]);
        assert!(
            tree.find_by_folder(Path::new("/data/nonexistent"))
                .is_none(),
            "find_by_folder must return None for an absent path"
        );
    }

    /// `display_label()` must be non-empty and contain the slice count.
    #[test]
    fn test_series_entry_display_label_contains_slice_count() {
        let entry = make_entry("P1", "Alice", None, None, "S1", "/s1", "CT", 42);
        let label = entry.display_label();
        assert!(!label.is_empty(), "display_label() must not be empty");
        assert!(
            label.contains("42"),
            "display_label() must contain the slice count '42'; got: {label}"
        );
    }

    /// `modality_icon()` must return a non-empty string for every supported
    /// modality and for unknown modalities.
    #[test]
    fn test_modality_icon_non_empty() {
        let modalities = [
            "CT", "MR", "PT", "NM", "US", "CR", "DR", "DX", "MG", "XA", "RF", "OT", "",
        ];
        for m in modalities {
            let entry = make_entry("P1", "X", None, None, "S1", "/s1", m, 1);
            let icon = entry.modality_icon();
            assert!(
                !icon.is_empty(),
                "modality_icon() must not be empty for modality '{m}'"
            );
        }
    }

    /// Two series with the same patient_id but different study_uids must
    /// produce two distinct StudyNodes under one PatientNode.
    #[test]
    fn test_from_entries_splits_different_studies() {
        let entries = vec![
            make_entry("P1", "Alice", Some("STUDY-A"), None, "S1", "/s1", "CT", 10),
            make_entry("P1", "Alice", Some("STUDY-B"), None, "S2", "/s2", "MR", 20),
        ];
        let tree = SeriesTree::from_entries(entries);
        assert_eq!(
            tree.patients.len(),
            1,
            "same patient_id must produce one PatientNode"
        );
        let alice = &tree.patients[0];
        assert_eq!(
            alice.studies.len(),
            2,
            "two distinct study_uids must produce two StudyNodes"
        );
        assert_eq!(tree.total_series(), 2);
    }
}
