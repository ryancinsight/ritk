//! Hierarchical DICOM series browser model.
//!
//! # Data model
//!
//! DICOM studies are organised as a three-level hierarchy:
//!
//! ```text
//! SeriesTree
//! â””â”€â”€ PatientNode (keyed by patient_id)
//!     â””â”€â”€ StudyNode (keyed by study_uid or study_date)
//!         â””â”€â”€ SeriesNode (one per DICOM series folder)
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
//! - [`SeriesTree::find_by_folder`] returns `Some` for every folder that
//!   appears in any series stored in the tree.

use ritk_io::DicomSeriesInfo;
use std::borrow::Cow;
use std::path::Path;

// â”€â”€ SeriesEntryView â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ ModalityMapper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Helper to map modality strings to emoji or text icons using a const-generic mapping array.
#[derive(Debug, Clone, Copy)]
pub struct ModalityMapper<const N: usize> {
    mappings: [(&'static str, &'static str); N] }

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
        "ðŸ—‚"
    }
}

/// Default list of DICOM modalities and their corresponding icons.
pub const DEFAULT_MODALITY_ICONS: [(&str, &str); 11] = [
    ("CT", "ðŸ«"),
    ("MR", "ðŸ§ "),
    ("PT", "â˜¢"),
    ("NM", "â˜¢"),
    ("US", "ã€°"),
    ("CR", "ðŸ“·"),
    ("DR", "ðŸ“·"),
    ("DX", "ðŸ“·"),
    ("MG", "ðŸŽ—"),
    ("XA", "ðŸ’‰"),
    ("RF", "ðŸ“¡"),
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

// â”€â”€ SeriesEntry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A single flat DICOM series representation as populated by directory scanning.
#[derive(Debug, Clone)]
pub struct SeriesEntry<'a> {
    /// Series Instance UID (may be empty when absent from metadata).
    pub series_uid: Cow<'a, str>,
    /// Absolute path to the folder containing the DICOM slice files.
    pub folder: Cow<'a, Path>,
    /// Patient name extracted from series metadata.
    pub patient_name: Cow<'a, str>,
    /// Patient ID extracted from series metadata.
    pub patient_id: Cow<'a, str>,
    /// DICOM modality string (e.g. `"CT"`, `"MR"`, `"PT"`).
    pub modality: Cow<'a, str>,
    /// Series description from tag (0008,103E).
    pub series_description: Cow<'a, str>,
    /// Number of image slices in the series.
    pub num_slices: usize,
    /// Study date in `YYYYMMDD` format, if present.
    pub study_date: Option<Cow<'a, str>>,
    /// Study Instance UID, if present.
    pub study_uid: Option<Cow<'a, str>> }

impl<'a> SeriesEntryView for SeriesEntry<'a> {
    type Str<'b>
        = &'b str
    where
        Self: 'b;
    type Path<'b>
        = &'b Path
    where
        Self: 'b;

    fn series_uid(&self) -> Self::Str<'_> {
        self.series_uid.as_ref()
    }
    fn folder(&self) -> Self::Path<'_> {
        self.folder.as_ref()
    }
    fn modality(&self) -> Self::Str<'_> {
        self.modality.as_ref()
    }
    fn series_description(&self) -> Self::Str<'_> {
        self.series_description.as_ref()
    }
    fn num_slices(&self) -> usize {
        self.num_slices
    }
}

impl<'a> SeriesEntry<'a> {
    /// Construct a [`SeriesEntry`] from a [`DicomSeriesInfo`] returned by
    /// `ritk_io::scan_dicom_directory`.
    pub fn from_dicom_series_info(info: DicomSeriesInfo) -> Self {
        let series_uid = info.series_instance_uid().to_string();
        let modality = info.modality().to_string();
        let folder = info
            .file_paths
            .first()
            .and_then(|path| path.parent())
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let num_slices = info.file_paths.len();
        Self {
            series_uid: Cow::Owned(series_uid),
            folder: Cow::Owned(folder),
            patient_name: Cow::Borrowed(""),
            patient_id: Cow::Owned(info.patient_id),
            modality: Cow::Owned(modality),
            series_description: Cow::Owned(info.series_description),
            num_slices,
            study_date: None,
            study_uid: None }
    }

    /// Short display label used in the series browser tree.
    pub fn display_label(&self) -> String {
        format_series_label(self, &DEFAULT_MODALITY_MAPPER)
    }

    /// Short modality icon string (emoji or abbreviated text) for display.
    pub fn modality_icon(&self) -> &'static str {
        DEFAULT_MODALITY_MAPPER.get_icon(self.modality.as_ref())
    }
}

// â”€â”€ SeriesNode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A leaf node in the DICOM series tree, containing only series-specific data.
///
/// Patient and study details are stored strictly in parent nodes (`PatientNode`
/// and `StudyNode`), enforcing DRY and SSOT.
#[derive(Debug, Clone)]
pub struct SeriesNode<'a> {
    /// Series Instance UID.
    pub series_uid: Cow<'a, str>,
    /// Absolute path to the folder containing the DICOM slice files.
    pub folder: Cow<'a, Path>,
    /// DICOM modality string.
    pub modality: Cow<'a, str>,
    /// Series description.
    pub series_description: Cow<'a, str>,
    /// Number of image slices.
    pub num_slices: usize }

impl<'a> SeriesEntryView for SeriesNode<'a> {
    type Str<'b>
        = &'b str
    where
        Self: 'b;
    type Path<'b>
        = &'b Path
    where
        Self: 'b;

    fn series_uid(&self) -> Self::Str<'_> {
        self.series_uid.as_ref()
    }
    fn folder(&self) -> Self::Path<'_> {
        self.folder.as_ref()
    }
    fn modality(&self) -> Self::Str<'_> {
        self.modality.as_ref()
    }
    fn series_description(&self) -> Self::Str<'_> {
        self.series_description.as_ref()
    }
    fn num_slices(&self) -> usize {
        self.num_slices
    }
}

impl<'a> SeriesNode<'a> {
    /// Short display label used in the series browser tree.
    pub fn display_label(&self) -> String {
        format_series_label(self, &DEFAULT_MODALITY_MAPPER)
    }

    /// Short modality icon string (emoji or abbreviated text) for display.
    pub fn modality_icon(&self) -> &'static str {
        DEFAULT_MODALITY_MAPPER.get_icon(self.modality.as_ref())
    }
}

// â”€â”€ StudyNode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// One study within a patient, containing one or more series.
#[derive(Debug, Clone)]
pub struct StudyNode<'a> {
    /// Study Instance UID â€” `None` when absent from metadata.
    pub study_uid: Option<Cow<'a, str>>,
    /// Study date in `YYYYMMDD` format â€” `None` when absent.
    pub study_date: Option<Cow<'a, str>>,
    /// Series belonging to this study, in insertion order.
    pub series: Vec<SeriesNode<'a>> }

impl<'a> StudyNode<'a> {
    /// Canonical grouping key for deduplication inside a patient.
    pub fn key(&self) -> &str {
        self.study_uid
            .as_deref()
            .or(self.study_date.as_deref())
            .unwrap_or("")
    }
}

// â”€â”€ PatientNode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// One patient, containing one or more studies.
#[derive(Debug, Clone)]
pub struct PatientNode<'a> {
    /// Patient ID string from DICOM metadata.
    pub patient_id: Cow<'a, str>,
    /// Patient name string from DICOM metadata.
    pub patient_name: Cow<'a, str>,
    /// Studies belonging to this patient, in insertion order.
    pub studies: Vec<StudyNode<'a>> }

// â”€â”€ SeriesTree â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Hierarchical patient â†’ study â†’ series tree for the series browser.
#[derive(Debug, Clone, Default)]
pub struct SeriesTree<'a> {
    /// Top-level patient nodes, in insertion order.
    pub patients: Vec<PatientNode<'a>> }

impl<'a> SeriesTree<'a> {
    /// Construct an empty tree.
    pub fn new() -> Self {
        Self {
            patients: Vec::new() }
    }

    /// Build the hierarchy from a flat list of [`SeriesEntry`] records.
    pub fn from_entries(entries: Vec<SeriesEntry<'a>>) -> Self {
        let mut tree = Self::new();
        let mut patient_map = std::collections::HashMap::new();
        let mut study_maps = Vec::new(); // maps patient_idx -> HashMap<study_key, study_idx>

        for entry in entries {
            let SeriesEntry {
                series_uid,
                folder,
                patient_name,
                patient_id,
                modality,
                series_description,
                num_slices,
                study_date,
                study_uid } = entry;

            let patient_idx = if patient_id.is_empty() {
                // Anonymous patients each get their own node.
                tree.patients.push(PatientNode {
                    patient_id,
                    patient_name,
                    studies: Vec::new() });
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
                            studies: Vec::new() });
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
                (None, None) => None };

            let patient = &mut tree.patients[patient_idx];
            let study_map = &mut study_maps[patient_idx];

            let study_idx = match study_key {
                None => {
                    patient.studies.push(StudyNode {
                        study_uid: None,
                        study_date: None,
                        series: Vec::new() });
                    patient.studies.len() - 1
                }
                Some(key) => match study_map.get(key) {
                    Some(&idx) => idx,
                    None => {
                        let key_clone = key.clone();
                        patient.studies.push(StudyNode {
                            study_uid,
                            study_date,
                            series: Vec::new() });
                        let idx = patient.studies.len() - 1;
                        study_map.insert(key_clone, idx);
                        idx
                    }
                } };

            patient.studies[study_idx].series.push(SeriesNode {
                series_uid,
                folder,
                modality,
                series_description,
                num_slices });
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

    /// Find the first [`SeriesNode`] whose `folder` equals `folder`.
    pub fn find_by_folder(&self, folder: &Path) -> Option<&SeriesNode<'a>> {
        self.patients
            .iter()
            .flat_map(|p| p.studies.iter())
            .flat_map(|s| s.series.iter())
            .find(|e| e.folder.as_ref() == folder)
    }

    /// Iterate over every [`SeriesNode`] in the tree in insertion order.
    pub fn iter_series(&self) -> impl Iterator<Item = &SeriesNode<'a>> {
        self.patients
            .iter()
            .flat_map(|p| p.studies.iter())
            .flat_map(|s| s.series.iter())
    }
}

#[cfg(test)]
mod tests;
