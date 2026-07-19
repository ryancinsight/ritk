//! PACS query domain types: request, response, result rows, and state machine.
//!
//! # Overview
//!
//! - [`FindResultRow`] — a decoded C-FIND response dataset.
//! - [`PacsRequest`] — outbound request variant sent to the background worker.
//! - [`PacsResponse`] — response variant returned from the background worker.
//! - [`QueryState`] — viewer-side state machine: `Idle → Pending → Results | Error`.

use ritk_io::format::dicom::networking::parse_dataset_ivr_le;
use ritk_io::{FindLevel, FindQuery, MoveResponse};

// ── FindResultRowSeries ───────────────────────────────────────────────────────

/// A single result row decoded from a SERIES-level C-FIND response dataset.
///
/// # DICOM attributes decoded (IVR-LE, Series Root query level)
///
/// | Tag | Attribute |
/// |--------------|-------------------------------------|
/// | (0008,0060) | Modality |
/// | (0008,103E) | SeriesDescription |
/// | (0020,000E) | SeriesInstanceUID |
/// | (0020,0011) | SeriesNumber |
/// | (0020,1209) | NumberOfSeriesRelatedInstances |
/// | (0008,0021) | SeriesDate |
/// | (0008,0031) | SeriesTime |
/// | (0008,0050) | AccessionNumber |
/// | (0020,000D) | StudyInstanceUID (required query key) |
#[derive(Debug, Clone, Default)]
pub struct FindResultRowSeries {
    pub study_instance_uid: String,
    pub series_instance_uid: String,
    pub series_number: String,
    pub modality: String,
    pub series_description: String,
    pub num_instances: String,
    pub series_date: String,
    pub series_time: String,
    pub accession_number: String,
}

impl FindResultRowSeries {
    /// Decode a `FindResultRowSeries` from a raw IVR-LE C-FIND response dataset.
    ///
    /// Missing or non-UTF-8 attributes produce empty strings.
    /// Malformed trailing bytes are silently ignored (graceful degradation per
    /// DICOM PS 3.5 §7.1 — incomplete elements at end of stream are skipped).
    ///
    /// Uses a `HashMap` for O(1) per-field lookup (O(n) single pass to build
    /// the map, then O(1) per field — total O(n + fields) vs the naive O(n×fields)).
    pub fn from_raw_bytes(bytes: &[u8]) -> Self {
        let attr_map: std::collections::HashMap<(u16, u16), Vec<u8>> =
            parse_dataset_ivr_le(bytes).into_iter().collect();

        let get = |group: u16, element: u16| -> String {
            attr_map
                .get(&(group, element))
                .map(|v| {
                    std::str::from_utf8(v)
                        .unwrap_or_default()
                        .trim_end_matches(['\0', ' '])
                        .to_owned()
                })
                .unwrap_or_default()
        };

        Self {
            study_instance_uid: get(0x0020, 0x000D),
            series_instance_uid: get(0x0020, 0x000E),
            series_number: get(0x0020, 0x0011),
            modality: get(0x0008, 0x0060),
            series_description: get(0x0008, 0x103E),
            num_instances: get(0x0020, 0x1209), // NumberOfSeriesRelatedInstances (series scope)
            series_date: get(0x0008, 0x0021),
            series_time: get(0x0008, 0x0031),
            accession_number: get(0x0008, 0x0050),
        }
    }

    /// Build a series-level C-FIND query dataset for drilling into a study.
    ///
    /// `study_instance_uid` is the required filter key — only series belonging
    /// to this study will be returned by the SCP.
    ///
    /// Return keys cover all nine attributes decoded by
    /// [`FindResultRowSeries::from_raw_bytes`].
    pub fn build_series_query(study_instance_uid: &str) -> FindQuery {
        FindQuery::new(FindLevel::Series)
            .with_key(0x0020, 0x000D, study_instance_uid) // StudyInstanceUID (required filter)
            .with_key(0x0008, 0x0060, "") // Modality (return)
            .with_key(0x0008, 0x103E, "") // SeriesDescription (return)
            .with_key(0x0020, 0x000E, "") // SeriesInstanceUID (return)
            .with_key(0x0020, 0x0011, "") // SeriesNumber (return)
            .with_key(0x0020, 0x1209, "") // NumberOfSeriesRelatedInstances (return)
            .with_key(0x0008, 0x0021, "") // SeriesDate (return)
            .with_key(0x0008, 0x0031, "") // SeriesTime (return)
            .with_key(0x0008, 0x0050, "") // AccessionNumber (return)
    }
}

// ── FindResultRow ──────────────────────────────────────────────────────────────

/// A single result row decoded from a C-FIND response dataset.
///
/// # DICOM attributes decoded (IVR-LE, Study Root query level)
///
/// | Tag          | Attribute                           |
/// |--------------|-------------------------------------|
/// | (0008,0020)  | StudyDate                           |
/// | (0008,0050)  | AccessionNumber                     |
/// | (0008,0060)  | Modality                            |
/// | (0008,1030)  | StudyDescription                    |
/// | (0010,0010)  | PatientName                         |
/// | (0010,0020)  | PatientID                           |
/// | (0020,000D)  | StudyInstanceUID                    |
/// | (0020,1206)  | NumberOfStudyRelatedSeries          |
/// | (0020,1208)  | NumberOfStudyRelatedInstances       |
///
/// # Study-level query note
///
/// `SeriesDescription` (0008,103E) and `SeriesInstanceUID` (0020,000E) are
/// SERIES-level attributes and are never returned by a Study Root STUDY-level
/// C-FIND query.  They are not decoded by this struct.
/// `num_instances` uses tag (0020,1208) — `NumberOfStudyRelatedInstances`
/// (study scope) — not (0020,1209) which is series-scoped.
#[derive(Debug, Clone, Default)]
pub struct FindResultRow {
    pub patient_name: String,
    pub patient_id: String,
    pub study_date: String,
    pub study_description: String,
    pub modality: String,
    pub accession_number: String,
    pub study_instance_uid: String,
    pub num_series: String,
    pub num_instances: String,
}

impl FindResultRow {
    /// Decode a `FindResultRow` from a raw IVR-LE C-FIND response dataset.
    ///
    /// Missing or non-UTF-8 attributes produce empty strings.
    /// Malformed trailing bytes are silently ignored (graceful degradation per
    /// DICOM PS 3.5 §7.1 — incomplete elements at end of stream are skipped).
    ///
    /// Uses a `HashMap` for O(1) per-field lookup (O(n) single pass to build
    /// the map, then O(1) per field — total O(n + fields) vs the naive O(n×fields)).
    pub fn from_raw_bytes(bytes: &[u8]) -> Self {
        let attr_map: std::collections::HashMap<(u16, u16), Vec<u8>> =
            parse_dataset_ivr_le(bytes).into_iter().collect();
        let get = |group: u16, element: u16| -> String {
            attr_map
                .get(&(group, element))
                .map(|v| {
                    std::str::from_utf8(v)
                        .unwrap_or_default()
                        .trim_end_matches(['\0', ' '])
                        .to_owned()
                })
                .unwrap_or_default()
        };

        Self {
            patient_name: get(0x0010, 0x0010),
            patient_id: get(0x0010, 0x0020),
            study_date: get(0x0008, 0x0020),
            study_description: get(0x0008, 0x1030),
            modality: get(0x0008, 0x0060),
            accession_number: get(0x0008, 0x0050),
            study_instance_uid: get(0x0020, 0x000D),
            num_series: get(0x0020, 0x1206),
            num_instances: get(0x0020, 0x1208), // NumberOfStudyRelatedInstances (study scope)
        }
    }

    /// Build a study-level C-FIND query dataset.
    ///
    /// `patient_name` accepts DICOM wildcard characters (`*`, `?`).
    /// An empty `modality` string requests all modalities (key set to `""`).
    /// `study_date` accepts DICOM date range format (`YYYYMMDD-YYYYMMDD`,
    /// `YYYYMMDD-`, `-YYYYMMDD`); empty string = return all dates.
    /// `accession_number` is an exact-match filter; empty string = all.
    ///
    /// Return keys cover all nine attributes decoded by [`FindResultRow::from_raw_bytes`].
    pub fn build_study_query(
        patient_name: &str,
        modality: &str,
        study_date: &str,
        accession_number: &str,
    ) -> FindQuery {
        let mut q = FindQuery::new(FindLevel::Study)
            .with_key(0x0010, 0x0010, patient_name) // PatientName (filter / return)
            .with_key(0x0010, 0x0020, "") // PatientID (return)
            .with_key(0x0008, 0x0020, study_date) // StudyDate (range filter if non-empty; return key)
            .with_key(0x0008, 0x0050, accession_number) // AccessionNumber (filter / return)
            .with_key(0x0008, 0x1030, "") // StudyDescription (return)
            .with_key(0x0020, 0x000D, "") // StudyInstanceUID (return)
            .with_key(0x0020, 0x1206, "") // NumberOfStudyRelatedSeries (return)
            .with_key(0x0020, 0x1208, ""); // NumberOfStudyRelatedInstances (return)
        if modality.is_empty() {
            q = q.with_key(0x0008, 0x0060, ""); // return all modalities
        } else {
            q = q.with_key(0x0008, 0x0060, modality); // filter by modality
        }
        q
    }
}

// ── Request / Response ────────────────────────────────────────────────────────

/// Outbound PACS request sent to the background worker thread.
#[derive(Debug)]
pub enum PacsRequest {
    /// C-ECHO connectivity verification (PS 3.4 §A.5).
    Echo,
    /// C-FIND study-level query (Study Root Query/Retrieve, PS 3.4 §C.4.1).
    ///
    /// `study_date` — DICOM date range format (`YYYYMMDD-YYYYMMDD`, `YYYYMMDD-`,
    /// `-YYYYMMDD`); empty string means no date filter (return all).
    /// `accession_number` — exact-match filter; empty string = all.
    FindStudies {
        patient_name: String,
        modality: String,
        study_date: String,
        accession_number: String,
    },
    /// C-FIND series-level drill-down query (Study Root Query/Retrieve, PS 3.4 §C.4.1).
    ///
    /// Returns all series within the specified study.
    FindSeries { study_instance_uid: String },
    /// C-MOVE study retrieval to a configured destination AE (PS 3.4 §C.4.2).
    ///
    /// The PACS will forward matching instances to `move_destination` via
    /// C-STORE sub-operations. This SCU does not receive them directly.
    RetrieveStudy {
        study_instance_uid: String,
        move_destination: String,
    },
    /// C-MOVE series-level retrieval to a configured destination AE.
    ///
    /// Requests that the PACS transfer only instances belonging to the
    /// specified series within the given study.
    RetrieveSeries {
        study_instance_uid: String,
        series_instance_uid: String,
        move_destination: String,
    },
}

/// Response returned from the background worker thread.
///
/// Exactly one response is produced per request.
#[derive(Debug)]
pub enum PacsResponse {
    /// C-ECHO succeeded; `status` is the DIMSE status code (0x0000 = Success).
    EchoOk { status: u16 },
    /// C-ECHO failed with a human-readable error description.
    EchoErr(String),
    /// C-FIND returned decoded result rows (may be empty — zero matches).
    FindOk(Vec<FindResultRow>),
    /// C-FIND series-level returned decoded result rows.
    FindSeriesOk(Vec<FindResultRowSeries>),
    /// C-FIND failed with a human-readable error description.
    FindErr(String),
    /// C-MOVE (study-level) completed; `MoveResponse` carries sub-operation counters.
    RetrieveOk(MoveResponse),
    /// C-MOVE failed with a human-readable error description.
    RetrieveErr(String),
    /// C-MOVE (series-level) completed; `MoveResponse` carries sub-operation counters.
    RetrieveSeriesOk(MoveResponse),
    /// C-MOVE (series-level) failed with a human-readable error description.
    RetrieveSeriesErr(String),
}

// ── QueryState ────────────────────────────────────────────────────────────────

/// PACS panel query state machine.
///
/// # Transitions
///
/// ```text
/// Idle → Pending → Results
///               → Error
/// Results → Idle  (user presses Clear)
/// Error   → Idle  (user presses Clear)
/// ```
#[derive(Debug, Default)]
pub enum QueryState {
    /// No request in-flight; panel shows the query form.
    #[default]
    Idle,
    /// A request is pending; panel shows a spinner and the operation label.
    Pending { label: String },
    /// C-FIND returned decoded result rows.
    Results(Vec<FindResultRow>),
    /// Series-level drill-down results for a specific study.
    SeriesResults {
        study_instance_uid: String,
        series: Vec<FindResultRowSeries>,
    },
    /// The last operation failed; panel shows the error description.
    Error(String),
}
