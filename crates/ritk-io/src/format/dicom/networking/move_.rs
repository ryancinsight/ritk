//! C-MOVE SCU â€” Study Root Query/Retrieve: MOVE (PS3.4 Â§C.4.2).
//!
//! # Protocol
//! 1. TCP connect â†’ A-ASSOCIATE-RQ with Study Root QR-MOVE SOP Class.
//! 2. A-ASSOCIATE-AC.
//! 3. C-MOVE-RQ command PDV (IVR-LE) + data PDV (IVR-LE query dataset).
//! 4. Zero or more C-MOVE-RSP progress updates (status 0xFF00).
//! 5. Final C-MOVE-RSP with status 0x0000 (Success) or failure code.
//! 6. A-RELEASE-RQ â†’ A-RELEASE-RP.
//!
//! The PACS sends the matching objects via C-STORE to the `destination` AE.
//! This SCU only issues the C-MOVE request; it does not receive the objects.
//! To also receive the objects, the caller must run a C-STORE SCP concurrently.
//!
//! # Usage
//! ```ignore
//! let config = AssociationConfig::new(...);
//! let dest = MoveDestination::new(AeTitle::new("MY_SCP").unwrap());
//! let resp = retrieve(&config, &dest, "1.2.3.4.5")?;
//! assert_eq!(resp.final_status, 0x0000);
//! ```

use super::association::{release_client_association, AeTitle, MoveResponse, NetworkingError};
use super::command::{
    build_command_pdu, build_dataset_ivr_le, encode_str, parse_command_response,
    CommandElementValue, C_MOVE_RQ, C_MOVE_RSP, HAS_DATASET, IMPLICIT_VR_LE_TS, PRIORITY_MEDIUM,
    STATUS_PENDING, STATUS_PENDING_WARN, STUDY_ROOT_MOVE_SOP_CLASS,
};
use super::context::AssociationConfig;
use super::echo::{find_ctx_id, receive_command_pdv};
use dicom_ul::association::client::ClientAssociationOptions;
use dicom_ul::pdu::{PDataValue, PDataValueType, Pdu};

// â”€â”€ MoveDestination â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Validated destination AE title for a C-MOVE request.
///
/// The PACS will send the requested objects to this AE via C-STORE.
#[derive(Debug, Clone)]
pub struct MoveDestination(AeTitle);

impl MoveDestination {
    /// Construct a `MoveDestination` from a validated `AeTitle`.
    pub fn new(ae_title: AeTitle) -> Self {
        Self(ae_title)
    }

    /// Returns the AE title string.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

// â”€â”€ C-MOVE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Issue a C-MOVE request at the specified QR level.
///
/// Requests that the PACS transfer matching objects to the `destination` AE
/// via C-STORE. Progress is tracked via pending C-MOVE-RSP messages.
///
/// `query_keys` provides the attribute filter pairs `(group, element, value)`
/// that identify the objects to move (e.g. `(0x0020,0x000D,study_uid)` for
/// study level, or additionally `(0x0020,0x000E,series_uid)` for series level).
///
/// The QR level string (`level_cs`) determines the Retrieve Level:
/// `"STUDY"`, `"SERIES"`, or `"IMAGE"`.
///
/// Returns a `MoveResponse` aggregating the completion/failure/warning counts
/// from the final C-MOVE-RSP.
fn retrieve_impl(
    config: &AssociationConfig,
    destination: &MoveDestination,
    level_cs: &str,
    query_keys: &[(u16, u16, &str)],
) -> Result<MoveResponse, NetworkingError> {
    let mut assoc = ClientAssociationOptions::new()
        .calling_ae_title(config.calling_ae_title.as_str())
        .called_ae_title(config.called_ae_title.as_str())
        .with_presentation_context(STUDY_ROOT_MOVE_SOP_CLASS, vec![IMPLICIT_VR_LE_TS])
        .establish(format!("{}:{}", config.host, config.port))
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    let ctx_id = find_ctx_id(&assoc)?;

    let cmd_bytes = build_command_pdu(&[
        (
            0x0000_0002,
            CommandElementValue::Ui(STUDY_ROOT_MOVE_SOP_CLASS),
        ),
        (0x0000_0100, CommandElementValue::Us(C_MOVE_RQ)),
        (0x0000_0110, CommandElementValue::Us(1u16)),
        (0x0000_0600, CommandElementValue::Str(destination.as_str())),
        (0x0000_0700, CommandElementValue::Us(PRIORITY_MEDIUM)),
        (0x0000_0800, CommandElementValue::Us(HAS_DATASET)),
    ]);

    let level_bytes = encode_str(level_cs);
    let mut dataset_entries: Vec<(u32, Vec<u8>)> = vec![(0x0008_0052, level_bytes)];
    for &(group, element, value) in query_keys {
        let tag = ((group as u32) << 16) | (element as u32);
        dataset_entries.push((tag, encode_str(value)));
    }
    dataset_entries.sort_by_key(|(tag, _)| *tag);
    let dataset_refs: Vec<(u32, &[u8])> = dataset_entries
        .iter()
        .map(|(tag, v)| (*tag, v.as_slice()))
        .collect();
    let dataset_bytes = build_dataset_ivr_le(&dataset_refs);

    assoc
        .send(&Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: ctx_id,
                value_type: PDataValueType::Command,
                is_last: true,
                data: cmd_bytes,
            }],
        })
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    assoc
        .send(&Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: ctx_id,
                value_type: PDataValueType::Data,
                is_last: true,
                data: dataset_bytes,
            }],
        })
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    let mut completed: u16 = 0;
    let mut failed: u16 = 0;
    let mut warning: u16 = 0;

    let final_status = loop {
        let rsp_cmd_bytes = receive_command_pdv(&mut assoc, ctx_id)?;
        let rsp = parse_command_response(&rsp_cmd_bytes)?;

        if rsp.command_field != C_MOVE_RSP {
            return Err(NetworkingError::Protocol(format!(
                "expected C-MOVE-RSP (0x{:04X}), got 0x{:04X}",
                C_MOVE_RSP, rsp.command_field
            )));
        }

        if let Some(c) = rsp.number_completed {
            completed = c;
        }
        if let Some(f) = rsp.number_failed {
            failed = f;
        }
        if let Some(w) = rsp.number_warning {
            warning = w;
        }

        match rsp.status {
            STATUS_PENDING | STATUS_PENDING_WARN => continue,
            s => break s,
        }
    };

    release_client_association(assoc)?;

    Ok(MoveResponse {
        completed,
        failed,
        warning,
        final_status,
    })
}

/// Issue a Study-level C-MOVE request to the configured PACS.
///
/// Requests that the PACS transfer the study identified by `study_instance_uid`
/// to the `destination` AE via C-STORE. Progress is tracked via pending
/// C-MOVE-RSP messages.
//
/// Returns a `MoveResponse` aggregating the completion/failure/warning counts
/// from the final C-MOVE-RSP.
pub fn retrieve(
    config: &AssociationConfig,
    destination: &MoveDestination,
    study_instance_uid: &str,
) -> Result<MoveResponse, NetworkingError> {
    retrieve_impl(
        config,
        destination,
        "STUDY",
        &[(0x0020, 0x000D, study_instance_uid)],
    )
}

/// Issue a Series-level C-MOVE request to the configured PACS.
///
/// Requests that the PACS transfer the series identified by
/// `series_instance_uid` within `study_instance_uid` to the `destination` AE
/// via C-STORE. Progress is tracked via pending C-MOVE-RSP messages.
///
/// Returns a `MoveResponse` aggregating the completion/failure/warning counts
/// from the final C-MOVE-RSP.
pub fn retrieve_series(
    config: &AssociationConfig,
    destination: &MoveDestination,
    study_instance_uid: &str,
    series_instance_uid: &str,
) -> Result<MoveResponse, NetworkingError> {
    retrieve_impl(
        config,
        destination,
        "SERIES",
        &[
            (0x0020, 0x000D, study_instance_uid),
            (0x0020, 0x000E, series_instance_uid),
        ],
    )
}
