//! Unit tests for the PACS module.
//!
//! Tests cover IVR-LE response parsing, configuration defaults, query state
//! transitions, and C-FIND query construction. All tests are value-semantic:
//! they assert on computed field values, not on Result/Option existence alone.
//!
//! Extended query builder, SCP configuration, and filter propagation tests
//! are in `tests_query.rs`.
//!
//! No network connections are required — all tests run fully offline.

use super::config::PacsConfig;
use super::query::{FindResultRow, FindResultRowSeries, PacsResponse, QueryState};

// ── IVR-LE encoding helper ────────────────────────────────────────────────────

/// Encode a single DICOM IVR-LE element as bytes.
///
/// Format: `[group:u16-LE][element:u16-LE][length:u32-LE][value:bytes]`
/// (PS 3.5 §7.1 Table 7.1-1, Implicit VR Little Endian).
fn encode_ivr_le_tag(group: u16, element: u16, value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + value.len());
    buf.extend_from_slice(&group.to_le_bytes());
    buf.extend_from_slice(&element.to_le_bytes());
    buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
    buf.extend_from_slice(value);
    buf
}

// ── FindResultRow parsing ─────────────────────────────────────────────────────

/// Boundary: zero-length input → all fields default to empty string.
///
/// Analytical basis: `parse_dataset_ivr_le(&[])` returns an empty attribute
/// list; `get(g, e)` on an empty list returns the `Default` for `String` = `""`.
#[test]
fn test_find_result_row_from_empty_bytes_all_fields_empty() {
    let row = FindResultRow::from_raw_bytes(&[]);
    assert!(
        row.patient_name.is_empty(),
        "patient_name must be empty for empty bytes"
    );
    assert!(row.patient_id.is_empty(), "patient_id must be empty");
    assert!(row.study_date.is_empty(), "study_date must be empty");
    assert!(
        row.study_description.is_empty(),
        "study_description must be empty"
    );
    assert!(row.modality.is_empty(), "modality must be empty");
    assert!(
        row.study_instance_uid.is_empty(),
        "study_instance_uid must be empty"
    );
    assert!(row.num_series.is_empty(), "num_series must be empty");
    assert!(row.num_instances.is_empty(), "num_instances must be empty");
}

/// Positive: single PatientName tag (0010,0010) → `patient_name` field populated.
///
/// Analytical basis: IVR-LE element with group=0x0010, element=0x0010,
/// value="DOE^JOHN" (8 bytes). All other fields absent → empty.
#[test]
fn test_find_result_row_patient_name_parsed() {
    let value = b"DOE^JOHN";
    let bytes = encode_ivr_le_tag(0x0010, 0x0010, value);
    let row = FindResultRow::from_raw_bytes(&bytes);
    assert_eq!(row.patient_name, "DOE^JOHN");
    assert!(
        row.patient_id.is_empty(),
        "patient_id must be empty when only PatientName tag present"
    );
    assert!(
        row.modality.is_empty(),
        "modality must be empty when only PatientName tag present"
    );
}

/// Positive: four tags encoded in tag-ascending order → all decoded correctly.
///
/// Tags encoded: StudyDate (0008,0020), Modality (0008,0060),
/// PatientName (0010,0010), StudyInstanceUID (0020,000D).
#[test]
fn test_find_result_row_multiple_tags_parsed() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x0020, b"20240115"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x0060, b"CT"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0010, 0x0010, b"DOE^JANE"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0020, 0x000D, b"1.2.840.10008.99"));
    let row = FindResultRow::from_raw_bytes(&bytes);
    assert_eq!(row.study_date, "20240115");
    assert_eq!(row.modality, "CT");
    assert_eq!(row.patient_name, "DOE^JANE");
    assert_eq!(row.study_instance_uid, "1.2.840.10008.99");
    // Unset fields remain empty.
    assert!(row.patient_id.is_empty());
    assert!(row.study_description.is_empty());
}

/// Boundary: DICOM string values are null-padded to even length (PS 3.5 §6.2).
/// Trailing null byte must be stripped by `from_raw_bytes`.
///
/// PatientID tag (0010,0020) with value "P123\0" (null-padded to 5 bytes is odd,
/// so DICOM pads to 6: "P123\0 " — here we test with a direct trailing null).
#[test]
fn test_find_result_row_null_padded_value_trimmed() {
    // Value "P123" padded with a trailing null byte.
    let value = b"P123\x00";
    let bytes = encode_ivr_le_tag(0x0010, 0x0020, value);
    let row = FindResultRow::from_raw_bytes(&bytes);
    assert_eq!(
        row.patient_id, "P123",
        "trailing null must be stripped from PatientID"
    );
}

/// Boundary: space-padded string values (PS 3.5 §6.2 CS/LO/SH VR even-length padding).
/// Trailing space must be stripped.
#[test]
fn test_find_result_row_space_padded_modality_trimmed() {
    // Modality "MR " (space-padded to 3 bytes — would be odd, but test trimming).
    let value = b"MR ";
    let bytes = encode_ivr_le_tag(0x0008, 0x0060, value);
    let row = FindResultRow::from_raw_bytes(&bytes);
    assert_eq!(
        row.modality, "MR",
        "trailing space must be stripped from Modality"
    );
}

// ── PacsConfig ────────────────────────────────────────────────────────────────

/// Verify default calling AE title is "RITKSNAP".
///
/// This value is the self-identification of the RITK viewer in DICOM
/// associations. It must never be empty (PS 3.8 §7.1.1 requires 1–16 chars).
#[test]
fn test_pacs_config_default_calling_ae_title() {
    let cfg = PacsConfig::default();
    assert_eq!(cfg.calling_ae_title, "RITKSNAP");
    assert!(
        !cfg.calling_ae_title.is_empty(),
        "calling AE title must be non-empty"
    );
    assert!(
        cfg.calling_ae_title.len() <= 16,
        "calling AE title must be ≤ 16 characters per PS 3.8"
    );
}

/// Verify default port is 4242 (Orthanc test default).
#[test]
fn test_pacs_config_default_port() {
    let cfg = PacsConfig::default();
    assert_eq!(cfg.port, 4242);
}

/// Verify `to_association_config` copies all fields correctly.
///
/// `AssociationConfig` must carry the exact calling AE title, called AE title,
/// host, and port from `PacsConfig`.
#[test]
fn test_pacs_config_to_association_config_copies_fields() {
    let cfg = PacsConfig {
        calling_ae_title: "CALLER".to_owned(),
        called_ae_title: "PACS01".to_owned(),
        host: "192.168.1.10".to_owned(),
        port: 11112,
        move_destination: "STORE01".to_owned(),
        scp_ae_title: "STORE01".to_owned(),
        scp_port: 11112,
        timeout_secs: 60,
        auto_load_received: true,
        auto_load_limit: 512,
    };
    let assoc = cfg.to_association_config();
    assert_eq!(assoc.calling_ae_title, "CALLER");
    assert_eq!(assoc.called_ae_title, "PACS01");
    assert_eq!(assoc.host, "192.168.1.10");
    assert_eq!(assoc.port, 11112);
    assert_eq!(assoc.timeout, std::time::Duration::from_secs(60));
}

// ── QueryState ────────────────────────────────────────────────────────────────

/// Default `QueryState` must be `Idle` (no pending request, no results).
#[test]
fn test_query_state_default_is_idle() {
    let state = QueryState::default();
    assert!(
        matches!(state, QueryState::Idle),
        "QueryState::default() must be Idle; got {:?}",
        state
    );
}

// ── FindQuery builder ─────────────────────────────────────────────────────────

// ── FindResultRowSeries parsing ──────────────────────────────────────────────

/// Boundary: zero-length input → all series fields default to empty string.
#[test]
fn test_find_series_row_from_empty_bytes_all_fields_empty() {
    let row = FindResultRowSeries::from_raw_bytes(&[]);
    assert!(
        row.study_instance_uid.is_empty(),
        "study_instance_uid must be empty for empty bytes"
    );
    assert!(
        row.series_instance_uid.is_empty(),
        "series_instance_uid must be empty"
    );
    assert!(row.series_number.is_empty(), "series_number must be empty");
    assert!(row.modality.is_empty(), "modality must be empty");
    assert!(
        row.series_description.is_empty(),
        "series_description must be empty"
    );
    assert!(row.num_instances.is_empty(), "num_instances must be empty");
    assert!(row.series_date.is_empty(), "series_date must be empty");
    assert!(row.series_time.is_empty(), "series_time must be empty");
    assert!(
        row.accession_number.is_empty(),
        "accession_number must be empty"
    );
}

/// Positive: all 9 series-level attributes decoded from a synthetic IVR-LE dataset.
#[test]
fn test_find_series_row_all_fields_parsed() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x0060, b"CT"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x103E, b"CHEST ROUTINE"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0020, 0x000E, b"1.2.840.10008.200.1"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0020, 0x0011, b"3"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0020, 0x1209, b"150"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x0021, b"20240201"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x0031, b"143000"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x0050, b"ACC-2024-042"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0020, 0x000D, b"1.2.840.99.1"));
    let row = FindResultRowSeries::from_raw_bytes(&bytes);
    assert_eq!(row.modality, "CT", "modality");
    assert_eq!(
        row.series_description, "CHEST ROUTINE",
        "series_description"
    );
    assert_eq!(
        row.series_instance_uid, "1.2.840.10008.200.1",
        "series_instance_uid"
    );
    assert_eq!(row.series_number, "3", "series_number");
    assert_eq!(row.num_instances, "150", "num_instances");
    assert_eq!(row.series_date, "20240201", "series_date");
    assert_eq!(row.series_time, "143000", "series_time");
    assert_eq!(row.accession_number, "ACC-2024-042", "accession_number");
    assert_eq!(row.study_instance_uid, "1.2.840.99.1", "study_instance_uid");
}

// ── build_series_query ─────────────────────────────────────────────────────

/// `build_series_query` must include StudyInstanceUID as the mandatory filter
/// key and all 8 series-level return keys.
#[test]
fn test_build_series_query_includes_all_return_keys() {
    let q = FindResultRowSeries::build_series_query("1.2.840.99.1");
    let has_key = |group: u16, element: u16| -> bool {
        q.keys.iter().any(|(g, e, _)| *g == group && *e == element)
    };
    assert!(
        has_key(0x0020, 0x000D),
        "StudyInstanceUID (0020,000D) must be the filter key"
    );
    let filter_val = q
        .keys
        .iter()
        .find(|(g, e, _)| *g == 0x0020 && *e == 0x000D)
        .map(|(_, _, v)| v.as_str());
    assert_eq!(
        filter_val,
        Some("1.2.840.99.1"),
        "StudyInstanceUID filter value must match"
    );
    assert!(has_key(0x0008, 0x0060), "Modality (0008,0060) return key");
    assert!(
        has_key(0x0008, 0x103E),
        "SeriesDescription (0008,103E) return key"
    );
    assert!(
        has_key(0x0020, 0x000E),
        "SeriesInstanceUID (0020,000E) return key"
    );
    assert!(
        has_key(0x0020, 0x0011),
        "SeriesNumber (0020,0011) return key"
    );
    assert!(
        has_key(0x0020, 0x1209),
        "NumberOfSeriesRelatedInstances (0020,1209) return key"
    );
    assert!(has_key(0x0008, 0x0021), "SeriesDate (0008,0021) return key");
    assert!(has_key(0x0008, 0x0031), "SeriesTime (0008,0031) return key");
    assert!(
        has_key(0x0008, 0x0050),
        "AccessionNumber (0008,0050) return key"
    );
}

// ── PacsResponse ────────────────────────────────────────────────────────────

/// `PacsResponse::RetrieveSeriesOk` round-trips through debug formatting.
#[test]
fn test_pacs_response_retrieve_series_ok_message() {
    let rsp = PacsResponse::RetrieveSeriesOk(ritk_io::MoveResponse {
        completed: 5,
        failed: 0,
        warning: 0,
        final_status: 0x0000,
    });
    match &rsp {
        PacsResponse::RetrieveSeriesOk(m) => {
            assert_eq!(m.completed, 5, "completed count must round-trip");
            assert_eq!(m.failed, 0, "failed count must be 0");
            assert_eq!(m.final_status, 0x0000, "status must be Success (0x0000)");
        }
        other => panic!("expected RetrieveSeriesOk, got {other:?}"),
    }
}

/// `PacsResponse::RetrieveSeriesErr` stores the error description.
#[test]
fn test_pacs_response_retrieve_series_err_stores_message() {
    let err = "association rejected: no matching AE".to_owned();
    let rsp = PacsResponse::RetrieveSeriesErr(err.clone());
    match &rsp {
        PacsResponse::RetrieveSeriesErr(msg) => {
            assert_eq!(msg, &err, "error message must round-trip")
        }
        other => panic!("expected RetrieveSeriesErr, got {other:?}"),
    }
}

/// `build_study_query` with a patient name wildcard must include that wildcard
/// as the PatientName key (0010,0010).
///
/// Analytical basis: the patient name key is the primary filter — a DICOM SCP
/// matches it against the PatientName attribute using wildcard matching.
#[test]
fn test_build_study_query_contains_patient_name_wildcard() {
    let q = FindResultRow::build_study_query("DOE*", "", "", "");
    let patient_name_key = q
        .keys
        .iter()
        .find(|(group, element, _)| *group == 0x0010 && *element == 0x0010)
        .map(|(_, _, v)| v.as_str());
    assert_eq!(
        patient_name_key,
        Some("DOE*"),
        "PatientName key (0010,0010) must equal the supplied wildcard"
    );
}

/// When modality is empty, the query must include a Modality key (0008,0060)
/// with value `""` (request all modalities — DICOM matching semantics).
#[test]
fn test_build_study_query_empty_modality_uses_empty_key() {
    let q = FindResultRow::build_study_query("*", "", "", "");
    let modality_key = q
        .keys
        .iter()
        .find(|(group, element, _)| *group == 0x0008 && *element == 0x0060)
        .map(|(_, _, v)| v.as_str());
    assert_eq!(
        modality_key,
        Some(""),
        "empty modality must produce Modality key (0008,0060) with empty value"
    );
}

/// When modality is "CT", the Modality key (0008,0060) must carry "CT".
#[test]
fn test_build_study_query_ct_modality_uses_ct_filter() {
    let q = FindResultRow::build_study_query("*", "CT", "", "");
    let modality_key = q
        .keys
        .iter()
        .find(|(group, element, _)| *group == 0x0008 && *element == 0x0060)
        .map(|(_, _, v)| v.as_str());
    assert_eq!(
        modality_key,
        Some("CT"),
        "Modality key (0008,0060) must equal 'CT' when filtering by CT"
    );
}

// ── QueryState transitions ────────────────────────────────────────────────────

/// `QueryState::Pending` stores the operation label string.
///
/// Analytical basis: Pending is a named-field variant; `label` is rendered in
/// the panel spinner on every egui frame while a request is in-flight.
#[test]
fn test_query_state_pending_has_label() {
    let state = QueryState::Pending {
        label: "C-FIND\u{2026}".to_owned(),
    };
    match state {
        QueryState::Pending { label } => {
            assert_eq!(
                label, "C-FIND\u{2026}",
                "Pending label must equal the supplied string"
            );
        }
        other => panic!("expected Pending, got {:?}", other),
    }
}

/// `QueryState::Error` stores the error message string.
///
/// Analytical basis: Error is a tuple variant holding the DIMSE failure
/// description displayed in red in the results section.
#[test]
fn test_query_state_error_stores_message() {
    let msg = "connection refused: 127.0.0.1:4242".to_owned();
    let state = QueryState::Error(msg.clone());
    match state {
        QueryState::Error(m) => assert_eq!(m, msg, "Error must store the exact message"),
        other => panic!("expected Error, got {:?}", other),
    }
}
