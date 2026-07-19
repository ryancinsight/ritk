//! Extended PACS query builder, state transition, and SCP configuration tests.
//!
//! This file covers:
//! - Remaining PacsConfig default-field tests
//! - Full study-level field decoding
//! - Complete return-key coverage for build_study_query
//! - PacsPanelAction default
//! - Sprint 283: AccessionNumber + StudyDate range filter propagation
//! - Sprint 284: Embedded SCP configuration defaults
//! - auto_load_policy / auto_load_limit defaults
//!
//! No network connections are required â€” all tests run fully offline.

use super::config::PacsConfig;
use super::query::FindResultRow;

// â”€â”€ IVR-LE encoding helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Encode a single DICOM IVR-LE element as bytes.
///
/// Format: `[group:u16-LE][element:u16-LE][length:u32-LE][value:bytes]`
/// (PS 3.5 Â§7.1 Table 7.1-1, Implicit VR Little Endian).
fn encode_ivr_le_tag(group: u16, element: u16, value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + value.len());
    buf.extend_from_slice(&group.to_le_bytes());
    buf.extend_from_slice(&element.to_le_bytes());
    buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
    buf.extend_from_slice(value);
    buf
}

// â”€â”€ PacsConfig defaults (continued) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

/// Default `PacsConfig::called_ae_title` must be "ORTHANC" (â‰¤ 16 chars).
#[test]
fn test_pacs_config_default_called_ae_title() {
    let cfg = PacsConfig::default();
    assert_eq!(cfg.called_ae_title, "ORTHANC");
    assert!(
        !cfg.called_ae_title.is_empty(),
        "called AE title must be non-empty"
    );
    assert!(
        cfg.called_ae_title.len() <= 16,
        "called AE title must be â‰¤ 16 characters per PS 3.8"
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

// â”€â”€ FindResultRow â€” all study-level fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    assert_eq!(row.study_date, "20240101", "study_date");
    assert_eq!(row.modality, "MR", "modality");
    assert_eq!(row.study_description, "BRAIN MRI", "study_description");
    assert_eq!(row.patient_name, "SMITH^JOHN", "patient_name");
    assert_eq!(row.patient_id, "PT00042", "patient_id");
    assert_eq!(row.study_instance_uid, "1.2.840.99.1", "study_instance_uid");
    assert_eq!(row.num_series, "4", "num_series");
    assert_eq!(row.num_instances, "128", "num_instances via (0020,1208)");
}

// â”€â”€ build_study_query â€” complete return-key coverage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    assert!(
        has_key(0x0010, 0x0010),
        "PatientName (0010,0010) must be a return key"
    );
    assert!(
        has_key(0x0010, 0x0020),
        "PatientID (0010,0020) must be a return key"
    );
    assert!(
        has_key(0x0008, 0x0020),
        "StudyDate (0008,0020) must be a return key"
    );
    assert!(
        has_key(0x0008, 0x1030),
        "StudyDescription (0008,1030) must be a return key"
    );
    assert!(
        has_key(0x0008, 0x0060),
        "Modality (0008,0060) must be a return key"
    );
    assert!(
        has_key(0x0020, 0x000D),
        "StudyInstanceUID (0020,000D) must be a return key"
    );
    assert!(
        has_key(0x0020, 0x1206),
        "NumberOfStudyRelatedSeries (0020,1206) must be a return key"
    );
    assert!(
        has_key(0x0020, 0x1208),
        "NumberOfStudyRelatedInstances (0020,1208) must be a return key"
    );
    assert!(
        has_key(0x0008, 0x0050),
        "AccessionNumber (0008,0050) must be a return key"
    );
}

// â”€â”€ PacsPanelAction default â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ Sprint 283: AccessionNumber + StudyDate range tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    assert!(
        row.accession_number.is_empty(),
        "default accession_number must be empty"
    );
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
/// which DICOM C-FIND semantics interpret as "return all dates".
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
        "Empty study_date must produce empty value for (0008,0020) â€” match-all semantics"
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
            assert_eq!(
                accession_number, "ACC-001",
                "accession_number field must round-trip"
            );
        }
        _ => panic!("expected FindStudies variant"),
    }
}

// â”€â”€ Sprint 284: Embedded SCP configuration tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        "scp_ae_title must be â‰¤16 chars per PS 3.8"
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

/// `PacsConfig::auto_load_policy` defaults to `AutoLoadPolicy::Automatic`.
///
/// Analytical basis: when the embedded SCP receives instances via C-STORE,
/// the most common user intent is immediate loading into the viewer.
/// Default-automatic eliminates the manual step while the opt-out checkbox remains
/// available in the PACS panel for workflows that require deferred loading.
#[test]
fn test_pacs_config_auto_load_policy_defaults_to_automatic() {
    use super::config::AutoLoadPolicy;
    let cfg = PacsConfig::default();
    assert_eq!(
        cfg.auto_load_policy,
        AutoLoadPolicy::Automatic,
        "auto_load_policy must default to Automatic so SCP-received instances load automatically"
    );
}

/// `PacsConfig::auto_load_limit` defaults to 512.
///
/// Analytical basis: 512 covers a typical CT series (~300-500 slices) while
/// providing a safeguard against accidentally loading thousands of instances
/// from a large C-MOVE. The user can adjust the limit or click "Load Received"
/// to override.
#[test]
fn test_pacs_config_auto_load_limit_default() {
    let cfg = PacsConfig::default();
    assert_eq!(
        cfg.auto_load_limit, 512,
        "auto_load_limit must default to 512 (covers typical CT series)"
    );
}

/// `PacsConfig::auto_load_limit` is a `u32`.
///
/// Analytical basis: the field is declared as `u32` in the struct definition.
/// This test ensures the type has not been accidentally changed by asserting
/// that the value can be assigned to a `u32` variable.
#[test]
fn test_pacs_config_auto_load_limit_is_u32() {
    let cfg = PacsConfig::default();
    let _limit: u32 = cfg.auto_load_limit;
    assert_eq!(_limit, 512, "auto_load_limit must be u32 with default 512");
}
