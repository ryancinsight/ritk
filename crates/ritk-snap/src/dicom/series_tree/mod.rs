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
//!         └── SeriesNode (one per discovered DICOM acquisition)
//! ```
//!
//! [`SeriesTree::from_entries`] builds this hierarchy from a flat
//! `Vec<SeriesEntry>`. Patients with the same `patient_id` are merged;
//! studies with the same `study_uid` within a patient are merged.
//!
//! To eliminate string and path duplicate overhead, parent attributes
//! (such as patient name/ID and study UIDs) are stored only in parent nodes
//! (SSOT), and the leaf nodes (`SeriesNode`) store only series-specific data.
//!
//! # Invariants
//! - [`SeriesTree::total_series`] equals the number of entries passed to
//!   [`SeriesTree::from_entries`].
//! - [`SeriesTree::find_by_uid`] returns `Some` for every Series Instance UID that
//!   appears in any series stored in the tree.

use ritk_io::DicomSeriesInfo;
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

// ── SeriesEntryView ──────────────────────────────────────────────────────────

/// A zero-copy view abstraction over a DICOM series entry.
///
/// Uses Generic Associated Types (GATs) for zero-copy string and path access,
/// allowing implementations to return either borrowed or owned types.
pub trait SeriesEntryView {
    type Str<'b>: AsRef<str> + 'b
    where
        Self: 'b;
    type Path<'b>: AsRef<Path> + 'b
    where
        Self: 'b;

    fn series_uid(&self) -> Self::Str<'_>;
    fn folder(&self) -> Self::Path<'_>;
    fn modality(&self) -> Self::Str<'_>;
    fn series_description(&self) -> Self::Str<'_>;
    fn num_slices(&self) -> usize;
}

// ── ModalityMapper ───────────────────────────────────────────────────────────

/// Helper to map modality strings to emoji or text icons using a const-generic mapping array.
#[derive(Debug, Clone, Copy)]
pub struct ModalityMapper<const N: usize> {
    mappings: [(&'static str, &'static str); N],
}

impl<const N: usize> ModalityMapper<N> {
    /// Create a new modality mapper.
    pub const fn new(mappings: [(&'static str, &'static str); N]) -> Self {
        Self { mappings }
    }

    /// Return the icon for the given modality.
    pub fn get_icon(&self, modality: &str) -> &'static str {
        let upper = modality.to_ascii_uppercase();
        for &(m, icon) in &self.mappings {
            if m == upper {
                return icon;
            }
        }
        "🗂"
    }
}

/// Default list of DICOM modalities and their corresponding icons.
pub const DEFAULT_MODALITY_ICONS: [(&str, &str); 11] = [
    ("CT", "🫁"),
    ("MR", "🧠"),
    ("PT", "☢"),
    ("NM", "☢"),
    ("US", "〰"),
    ("CR", "📷"),
    ("DR", "📷"),
    ("DX", "📷"),
    ("MG", "🎗"),
    ("XA", "💉"),
    ("RF", "📡"),
];

/// Global default instance of the modality mapper.
pub static DEFAULT_MODALITY_MAPPER: ModalityMapper<11> =
    ModalityMapper::new(DEFAULT_MODALITY_ICONS);

/// Formats a series display label in a generic, monomorphized way.
pub fn format_series_label<S: SeriesEntryView, const N: usize>(
    entry: &S,
    mapper: &ModalityMapper<N>,
) -> String {
    let modality = entry.modality();
    let icon = mapper.get_icon(modality.as_ref());
    let desc_ref = entry.series_description();
    let desc_str = desc_ref.as_ref();
    let folder = entry.folder();
    let desc = if desc_str.is_empty() {
        folder
            .as_ref()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)")
    } else {
        desc_str
    };
    let mod_tag = if modality.as_ref().is_empty() {
        String::new()
    } else {
        format!("[{}] ", modality.as_ref())
    };
    format!("{icon} {mod_tag}{desc} ({} slices)", entry.num_slices())
}

// ── SeriesEntry ──────────────────────────────────────────────────────────────

/// A single flat DICOM series representation as populated by directory scanning.
#[derive(Debug, Clone)]
pub struct SeriesEntry<'a> {
    /// Canonical discovery descriptor, retaining exact files and series identity.
    pub acquisition: Arc<DicomSeriesInfo>,
    /// Patient name extracted from series metadata.
    pub patient_name: Cow<'a, str>,
    /// Study date, when known.
    pub study_date: Option<Cow<'a, str>>,
    /// Study Instance UID, when known.
    pub study_uid: Option<Cow<'a, str>>,
}

impl SeriesEntryView for SeriesEntry<'_> {
    type Str<'b>
        = &'b str
    where
        Self: 'b;
    type Path<'b>
        = &'b Path
    where
        Self: 'b;
    fn series_uid(&self) -> &str {
        self.acquisition.series_instance_uid()
    }
    fn folder(&self) -> &Path {
        self.acquisition
            .file_paths
            .first()
            .and_then(|path| path.parent())
            .unwrap_or_else(|| Path::new(""))
    }
    fn modality(&self) -> &str {
        self.acquisition.modality()
    }
    fn series_description(&self) -> &str {
        &self.acquisition.series_description
    }
    fn num_slices(&self) -> usize {
        self.acquisition.file_paths.len()
    }
}

impl SeriesEntry<'_> {
    /// Retain a discovered acquisition without losing its file selection.
    pub fn from_dicom_series_info(mut info: DicomSeriesInfo) -> Self {
        info.file_paths.sort();
        Self {
            acquisition: Arc::new(info),
            patient_name: Cow::Borrowed(""),
            study_date: None,
            study_uid: None,
        }
    }
    /// Display the series description, modality and slice count.
    pub fn display_label(&self) -> String {
        format_series_label(self, &DEFAULT_MODALITY_MAPPER)
    }
    /// Display the modality's icon.
    pub fn modality_icon(&self) -> &'static str {
        DEFAULT_MODALITY_MAPPER.get_icon(self.modality())
    }
}

// ── SeriesNode ───────────────────────────────────────────────────────────────

/// A leaf node in the DICOM series tree, containing only series-specific data.
///
/// Patient and study details are stored strictly in parent nodes (`PatientNode`
/// and `StudyNode`), enforcing DRY and SSOT.
#[derive(Debug, Clone)]
pub struct SeriesNode {
    /// Canonical acquisition descriptor shared by browser, loading and selection.
    pub acquisition: Arc<DicomSeriesInfo>,
}

impl SeriesEntryView for SeriesNode {
    type Str<'b>
        = &'b str
    where
        Self: 'b;
    type Path<'b>
        = &'b Path
    where
        Self: 'b;
    fn series_uid(&self) -> &str {
        self.acquisition.series_instance_uid()
    }
    fn folder(&self) -> &Path {
        self.acquisition
            .file_paths
            .first()
            .and_then(|path| path.parent())
            .unwrap_or_else(|| Path::new(""))
    }
    fn modality(&self) -> &str {
        self.acquisition.modality()
    }
    fn series_description(&self) -> &str {
        &self.acquisition.series_description
    }
    fn num_slices(&self) -> usize {
        self.acquisition.file_paths.len()
    }
}

impl SeriesNode {
    /// Display the series description, modality and slice count.
    pub fn display_label(&self) -> String {
        format_series_label(self, &DEFAULT_MODALITY_MAPPER)
    }
    /// Display the modality's icon.
    pub fn modality_icon(&self) -> &'static str {
        DEFAULT_MODALITY_MAPPER.get_icon(self.modality())
    }
}

// ── StudyNode ────────────────────────────────────────────────────────────────

/// One study within a patient, containing one or more series.
#[derive(Debug, Clone)]
pub struct StudyNode<'a> {
    /// Study Instance UID — `None` when absent from metadata.
    pub study_uid: Option<Cow<'a, str>>,
    /// Study date in `YYYYMMDD` format — `None` when absent.
    pub study_date: Option<Cow<'a, str>>,
    /// Series belonging to this study, in insertion order.
    pub series: Vec<SeriesNode>,
}

impl<'a> StudyNode<'a> {
    /// Canonical grouping key for deduplication inside a patient.
    pub fn key(&self) -> &str {
        self.study_uid
            .as_deref()
            .or(self.study_date.as_deref())
            .unwrap_or("")
    }
}

// ── PatientNode ──────────────────────────────────────────────────────────────

/// One patient, containing one or more studies.
#[derive(Debug, Clone)]
pub struct PatientNode<'a> {
    /// Patient ID string from DICOM metadata.
    pub patient_id: Cow<'a, str>,
    /// Patient name string from DICOM metadata.
    pub patient_name: Cow<'a, str>,
    /// Studies belonging to this patient, in insertion order.
    pub studies: Vec<StudyNode<'a>>,
}

// ── SeriesTree ───────────────────────────────────────────────────────────────

/// Hierarchical patient → study → series tree for the series browser.
#[derive(Debug, Clone, Default)]
pub struct SeriesTree<'a> {
    /// Top-level patient nodes, in insertion order.
    pub patients: Vec<PatientNode<'a>>,
}

impl<'a> SeriesTree<'a> {
    /// Construct an empty tree.
    pub fn new() -> Self {
        Self {
            patients: Vec::new(),
        }
    }

    /// Build the hierarchy from a flat list of [`SeriesEntry`] records.
    pub fn from_entries(entries: Vec<SeriesEntry<'a>>) -> Self {
        let mut tree = Self::new();
        let mut patient_map = std::collections::HashMap::new();
        let mut study_maps = Vec::new(); // maps patient_idx -> HashMap<study_key, study_idx>

        for entry in entries {
            let SeriesEntry {
                acquisition,
                patient_name,
                study_date,
                study_uid,
            } = entry;
            let patient_id: Cow<'_, str> = Cow::Owned(acquisition.patient_id.clone());

            let patient_idx = if patient_id.is_empty() {
                // Anonymous patients each get their own node.
                tree.patients.push(PatientNode {
                    patient_id,
                    patient_name,
                    studies: Vec::new(),
                });
                study_maps.push(std::collections::HashMap::new());
                tree.patients.len() - 1
            } else {
                match patient_map.get(&patient_id) {
                    Some(&idx) => idx,
                    None => {
                        let id_clone = patient_id.clone();
                        tree.patients.push(PatientNode {
                            patient_id,
                            patient_name,
                            studies: Vec::new(),
                        });
                        let idx = tree.patients.len() - 1;
                        patient_map.insert(id_clone, idx);
                        study_maps.push(std::collections::HashMap::new());
                        idx
                    }
                }
            };

            let study_key = match (&study_uid, &study_date) {
                (Some(uid), _) => Some(uid),
                (None, Some(date)) => Some(date),
                (None, None) => None,
            };

            let patient = &mut tree.patients[patient_idx];
            let study_map = &mut study_maps[patient_idx];

            let study_idx = match study_key {
                None => {
                    patient.studies.push(StudyNode {
                        study_uid: None,
                        study_date: None,
                        series: Vec::new(),
                    });
                    patient.studies.len() - 1
                }
                Some(key) => match study_map.get(key) {
                    Some(&idx) => idx,
                    None => {
                        let key_clone = key.clone();
                        patient.studies.push(StudyNode {
                            study_uid,
                            study_date,
                            series: Vec::new(),
                        });
                        let idx = patient.studies.len() - 1;
                        study_map.insert(key_clone, idx);
                        idx
                    }
                },
            };

            patient.studies[study_idx]
                .series
                .push(SeriesNode { acquisition });
        }
        tree
    }

    /// Total number of series stored across all patients and studies.
    pub fn total_series(&self) -> usize {
        self.patients
            .iter()
            .flat_map(|p| p.studies.iter())
            .map(|s| s.series.len())
            .sum()
    }

    /// Find the first [`SeriesNode`] with the requested Series Instance UID.
    pub fn find_by_uid(&self, uid: &str) -> Option<&SeriesNode> {
        self.patients
            .iter()
            .flat_map(|p| p.studies.iter())
            .flat_map(|s| s.series.iter())
            .find(|entry| entry.series_uid() == uid)
    }

    /// Iterate over every [`SeriesNode`] in the tree in insertion order.
    pub fn iter_series(&self) -> impl Iterator<Item = &SeriesNode> {
        self.patients
            .iter()
            .flat_map(|p| p.studies.iter())
            .flat_map(|s| s.series.iter())
    }
}

#[cfg(test)]
mod tests;
