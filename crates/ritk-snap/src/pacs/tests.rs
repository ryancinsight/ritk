//! Unit tests for the PACS module.
//!
//! Tests cover IVR-LE response parsing, configuration defaults, query state
//! transitions, and C-FIND query construction.  All tests are value-semantic:
//! they assert on computed field values, not on Result/Option existence alone.
//!
//! No network connections are required — all tests run fully offline.

use super::config::PacsConfig;
use super::query::{FindResultRow, QueryState};

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
    assert!(row.patient_name.is_empty(),      "patient_name must be empty for empty bytes");
    assert!(row.patient_id.is_empty(),        "patient_id must be empty");
    assert!(row.study_date.is_empty(),        "study_date must be empty");
    assert!(row.study_description.is_empty(), "study_description must be empty");
    assert!(row.modality.is_empty(),          "modality must be empty");
    assert!(row.study_instance_uid.is_empty(),"study_instance_uid must be empty");
    assert!(row.num_series.is_empty(),        "num_series must be empty");
    assert!(row.num_instances.is_empty(),     "num_instances must be empty");
}

/// Positive: single PatientName tag (0010,0010) → `patient_name` field populated.
///
/// Analytical basis: IVR-LE element with group=0x0010, element=0x0010,
/// value="DOE^JOHN" (8 bytes).  All other fields absent → empty.
#[test]
fn test_find_result_row_patient_name_parsed() {
    let value = b"DOE^JOHN";
    let bytes = encode_ivr_le_tag(0x0010, 0x0010, value);
    let row = FindResultRow::from_raw_bytes(&bytes);
    assert_eq!(row.patient_name, "DOE^JOHN");
    assert!(row.patient_id.is_empty(), "patient_id must be empty when only PatientName tag present");
    assert!(row.modality.is_empty(), "modality must be empty when only PatientName tag present");
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
    assert_eq!(row.patient_id, "P123", "trailing null must be stripped from PatientID");
}

/// Boundary: space-padded string values (PS 3.5 §6.2 CS/LO/SH VR even-length padding).
/// Trailing space must be stripped.
#[test]
fn test_find_result_row_space_padded_modality_trimmed() {
    // Modality "MR " (space-padded to 3 bytes — would be odd, but test trimming).
    let value = b"MR ";
    let bytes = encode_ivr_le_tag(0x0008, 0x0060, value);
    let row = FindResultRow::from_raw_bytes(&bytes);
    assert_eq!(row.modality, "MR", "trailing space must be stripped from Modality");
}

// ── PacsConfig ────────────────────────────────────────────────────────────────

/// Verify default calling AE title is "RITKSNAP".
///
/// This value is the self-identification of the RITK viewer in DICOM
/// associations.  It must never be empty (PS 3.8 §7.1.1 requires 1–16 chars).
#[test]
fn test_pacs_config_default_calling_ae_title() {
    let cfg = PacsConfig::default();
    assert_eq!(cfg.calling_ae_title, "RITKSNAP");
    assert!(!cfg.calling_ae_title.is_empty(), "calling AE title must be non-empty");
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
    let state = QueryState::Pending { label: "C-FIND\u{2026}".to_owned() };
    match state {
        QueryState::Pending { label } => {
            assert_eq!(label, "C-FIND\u{2026}", "Pending label must equal the supplied string");
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

// ── PacsConfig defaults ───────────────────────────────────────────────────────

/// Default `PacsConfig::timeout_secs` must be 30.
///
/// Analytical basis: 30 seconds is the standard DICOM association connect
/// timeout for interactive workstation use; longer timeouts produce unacceptable
/// UI blocking, shorter timeouts cause false negatives on busy networks.
#[test]
fn test_pacs_config_default_timeout_secs() {
    let cfg = PacsConfig::default();
    assert_eq!(cfg.timeout_secs, 30, "default timeout must be 30 seconds");
}

/// Default `PacsConfig::called_ae_title` must be "ORTHANC" (\u{2264} 16 chars).
#[test]
fn test_pacs_config_default_called_ae_title() {
    let cfg = PacsConfig::default();
    assert_eq!(cfg.called_ae_title, "ORTHANC");
    assert!(!cfg.called_ae_title.is_empty(), "called AE title must be non-empty");
    assert!(
        cfg.called_ae_title.len() <= 16,
        "called AE title must be \u{2264} 16 characters per PS 3.8"
    );
}

/// Default `PacsConfig::host` must be "localhost".
#[test]
fn test_pacs_config_default_host() {
    let cfg = PacsConfig::default();
    assert_eq!(cfg.host, "localhost", "default host must be localhost");
}

/// Default `PacsConfig::move_destination` must be "RITKSNAP".
#[test]
fn test_pacs_config_default_move_destination() {
    let cfg = PacsConfig::default();
    assert_eq!(
        cfg.move_destination, "RITKSNAP",
        "default move_destination must be RITKSNAP (same as calling AE)"
    );
}

// ── FindResultRow — all study-level fields ────────────────────────────────────

/// Positive: all 8 study-level attributes decoded from a synthetic IVR-LE dataset.
///
/// Analytical basis: each tag is encoded individually and verified against the
/// known literal value. The test exercises the complete decoding path without
/// requiring a network connection.
#[test]
fn test_find_result_row_all_study_fields_parsed() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x0020, b"20240101"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x0060, b"MR"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x1030, b"BRAIN MRI"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0010, 0x0010, b"SMITH^JOHN"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0010, 0x0020, b"PT00042"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0020, 0x000D, b"1.2.840.99.1"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0020, 0x1206, b"4"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0020, 0x1208, b"128"));

    let row = FindResultRow::from_raw_bytes(&bytes);
    assert_eq!(row.study_date,         "20240101",    "study_date");
    assert_eq!(row.modality,           "MR",          "modality");
    assert_eq!(row.study_description,  "BRAIN MRI",   "study_description");
    assert_eq!(row.patient_name,       "SMITH^JOHN",  "patient_name");
    assert_eq!(row.patient_id,         "PT00042",     "patient_id");
    assert_eq!(row.study_instance_uid, "1.2.840.99.1","study_instance_uid");
    assert_eq!(row.num_series,         "4",           "num_series");
    assert_eq!(row.num_instances,      "128",         "num_instances via (0020,1208)");
}

// ── build_study_query — complete return-key coverage ─────────────────────────

/// `build_study_query` must include return keys for all 8 study-level attributes
/// decoded by `FindResultRow::from_raw_bytes`.
///
/// Analytical basis: any return key absent from the C-FIND query dataset will
/// produce an empty field in the decoded result row even when the PACS holds a
/// value, because SCPs only return requested keys.
#[test]
fn test_build_study_query_includes_all_return_keys() {
    let q = FindResultRow::build_study_query("*", "CT", "", "");
    let has_key = |group: u16, element: u16| -> bool {
        q.keys.iter().any(|(g, e, _)| *g == group && *e == element)
    };
    assert!(has_key(0x0010, 0x0010), "PatientName (0010,0010) must be a return key");
    assert!(has_key(0x0010, 0x0020), "PatientID (0010,0020) must be a return key");
    assert!(has_key(0x0008, 0x0020), "StudyDate (0008,0020) must be a return key");
    assert!(has_key(0x0008, 0x1030), "StudyDescription (0008,1030) must be a return key");
    assert!(has_key(0x0008, 0x0060), "Modality (0008,0060) must be a return key");
    assert!(has_key(0x0020, 0x000D), "StudyInstanceUID (0020,000D) must be a return key");
    assert!(has_key(0x0020, 0x1206), "NumberOfStudyRelatedSeries (0020,1206) must be a return key");
    assert!(has_key(0x0020, 0x1208), "NumberOfStudyRelatedInstances (0020,1208) must be a return key");
    assert!(has_key(0x0008, 0x0050), "AccessionNumber (0008,0050) must be a return key");
}

// ── PacsPanelAction default ───────────────────────────────────────────────────

/// `PacsPanelAction::default()` must be `None`.
///
/// Analytical basis: the panel function returns `PacsPanelAction::None` on
/// frames with no user interaction; `Default` must match this expectation so
/// callers can initialize and compare correctly.
#[test]
fn test_pacs_panel_action_default_is_none() {
    let action = crate::ui::pacs_panel::PacsPanelAction::default();
    assert!(
        matches!(action, crate::ui::pacs_panel::PacsPanelAction::None),
        "PacsPanelAction::default() must be None"
    );
}

// ── Sprint 283: AccessionNumber + StudyDate range tests ───────────────────────────

/// `FindResultRow::from_raw_bytes` must decode AccessionNumber (0008,0050).
///
/// Analytical basis: AccessionNumber is a study-level C-FIND return attribute;
/// the field was absent before Sprint 283.
#[test]
fn test_find_result_row_accession_number_decoded() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0008, 0x0050, b"ACC-2024-001"));
    bytes.extend_from_slice(&encode_ivr_le_tag(0x0010, 0x0010, b"TEST^PATIENT"));
    let row = FindResultRow::from_raw_bytes(&bytes);
    assert_eq!(
        row.accession_number, "ACC-2024-001",
        "AccessionNumber (0008,0050) must be decoded into accession_number field"
    );
}

/// `FindResultRow::default()` must have an empty `accession_number`.
#[test]
fn test_find_result_row_default_has_empty_accession() {
    let row = FindResultRow::default();
    assert!(row.accession_number.is_empty(), "default accession_number must be empty");
}

/// `build_study_query` with a non-empty `study_date` must pass that value as
/// the (0008,0020) StudyDate key value.
///
/// Analytical basis: DICOM C-FIND date range filter is expressed as the
/// key value of (0008,0020), e.g. `"20240101-20241231"` (inclusive range).
#[test]
fn test_build_study_query_study_date_filter_propagated() {
    let q = FindResultRow::build_study_query("*", "", "20240101-20241231", "");
    let date_val = q
        .keys
        .iter()
        .find(|(g, e, _)| *g == 0x0008 && *e == 0x0020)
        .map(|(_, _, v)| v.as_str());
    assert_eq!(
        date_val,
        Some("20240101-20241231"),
        "StudyDate key value must equal the supplied date range filter"
    );
}

/// `build_study_query` with a non-empty `accession_number` must pass that
/// value as the (0008,0050) AccessionNumber key value.
#[test]
fn test_build_study_query_accession_filter_propagated() {
    let q = FindResultRow::build_study_query("*", "", "", "ACC-999");
    let acc_val = q
        .keys
        .iter()
        .find(|(g, e, _)| *g == 0x0008 && *e == 0x0050)
        .map(|(_, _, v)| v.as_str());
    assert_eq!(
        acc_val,
        Some("ACC-999"),
        "AccessionNumber key value must equal the supplied filter"
    );
}

/// Empty `study_date` must produce an empty-string value for (0008,0020),
/// which DICOM C-FIND semantics interpret as \"return all dates\".
#[test]
fn test_build_study_query_empty_date_is_wildcard() {
    let q = FindResultRow::build_study_query("*", "", "", "");
    let date_val = q
        .keys
        .iter()
        .find(|(g, e, _)| *g == 0x0008 && *e == 0x0020)
        .map(|(_, _, v)| v.as_str());
    assert_eq!(
        date_val,
        Some(""),
        "Empty study_date must produce empty value for (0008,0020) — match-all semantics"
    );
}

/// `PacsRequest::FindStudies` must carry `study_date` and `accession_number`.
#[test]
fn test_pacs_request_find_studies_has_new_filter_fields() {
    let req = crate::pacs::query::PacsRequest::FindStudies {
        patient_name: "*".to_owned(),
        modality: "CT".to_owned(),
        study_date: "20240101-".to_owned(),
        accession_number: "ACC-001".to_owned(),
    };
    match req {
        crate::pacs::query::PacsRequest::FindStudies {
            study_date,
            accession_number,
            ..
        } => {
            assert_eq!(study_date, "20240101-", "study_date field must round-trip");
            assert_eq!(accession_number, "ACC-001", "accession_number field must round-trip");
        }
        _ => panic!("expected FindStudies variant"),
    }
}

// ── Sprint 284: Embedded SCP configuration tests ─────────────────────────────

/// `PacsConfig::scp_ae_title` default must be "RITKSNAP".
///
/// Analytical basis: The embedded SCP's AE title must match the `calling_ae_title`
/// default so that C-MOVE `move_destination` points to the embedded SCP by default
/// without user configuration.
#[test]
fn test_pacs_config_scp_ae_title_default() {
    let cfg = PacsConfig::default();
    assert_eq!(
        cfg.scp_ae_title, "RITKSNAP",
        "default scp_ae_title must be RITKSNAP"
    );
    assert!(
        !cfg.scp_ae_title.is_empty(),
        "scp_ae_title must be non-empty"
    );
    assert!(
        cfg.scp_ae_title.len() <= 16,
        "scp_ae_title must be \u{2264}16 chars per PS 3.8"
    );
}

/// `PacsConfig::scp_port` default must be 11112.
///
/// Analytical basis: 11112 is the DICOM standard test/development port and
/// avoids the privileged 104 port that requires elevated OS permissions.
#[test]
fn test_pacs_config_scp_port_default() {
    let cfg = PacsConfig::default();
    assert_eq!(cfg.scp_port, 11112, "default scp_port must be 11112");
    assert_ne!(cfg.scp_port, 0, "scp_port must not be 0 in default config");
}

/// `PacsConfig` scp_ae_title and move_destination share the same default.
///
/// Analytical basis: the embedded SCP's AE title must equal `move_destination`
/// so that a C-MOVE issued with default config directs instances to the embedded SCP.
#[test]
fn test_pacs_config_scp_ae_matches_move_destination_default() {
    let cfg = PacsConfig::default();
    assert_eq!(
        cfg.scp_ae_title, cfg.move_destination,
        "scp_ae_title must equal move_destination in default config"
    );
}
