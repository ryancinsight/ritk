//! `DicomSeriesInfo` type — metadata for a discovered DICOM series.

use arrayvec::ArrayString;
use std::path::PathBuf;

/// Metadata for a discovered DICOM series.
#[derive(Debug, Clone)]
pub struct DicomSeriesInfo {
    pub(crate) series_instance_uid: ArrayString<64>,
    pub series_description: String,
    pub(crate) modality: ArrayString<16>,
    pub patient_id: String,
    pub(crate) patient_name: String,
    pub(crate) study_instance_uid: Option<ArrayString<64>>,
    pub(crate) study_date: Option<ArrayString<8>>,
    pub file_paths: Vec<PathBuf>,
}

impl DicomSeriesInfo {
    /// Construct a `DicomSeriesInfo` from string slices.
    ///
    /// # Panics
    /// Panics if `series_instance_uid` exceeds 64 characters or `modality` exceeds 16.
    pub fn new(
        series_instance_uid: &str,
        series_description: String,
        modality: &str,
        patient_id: String,
        file_paths: Vec<PathBuf>,
    ) -> Self {
        Self {
            series_instance_uid: ArrayString::from(series_instance_uid)
                .expect("invariant: series_instance_uid must not exceed 64 characters"),
            series_description,
            modality: ArrayString::from(modality)
                .expect("invariant: modality must not exceed 16 characters"),
            patient_id,
            patient_name: String::new(),
            study_instance_uid: None,
            study_date: None,
            file_paths,
        }
    }

    /// Returns the SeriesInstanceUID as a string slice.
    pub fn series_instance_uid(&self) -> &str {
        self.series_instance_uid.as_str()
    }

    /// Returns the Modality as a string slice.
    pub fn modality(&self) -> &str {
        self.modality.as_str()
    }

    /// Returns the PatientName value retained during directory scanning.
    pub fn patient_name(&self) -> &str {
        &self.patient_name
    }

    /// Returns a syntactically valid StudyInstanceUID, when present.
    pub fn study_instance_uid(&self) -> Option<&str> {
        self.study_instance_uid.as_ref().map(ArrayString::as_str)
    }

    /// Returns an eight-digit StudyDate, when present.
    pub fn study_date(&self) -> Option<&str> {
        self.study_date.as_ref().map(ArrayString::as_str)
    }
}
