//! C-MOVE SCU — Study Root Query/Retrieve: MOVE (PS3.4 §C.4.2).
//!
//! # Protocol
//! 1. TCP connect → A-ASSOCIATE-RQ with Study Root QR-MOVE SOP Class.
//! 2. A-ASSOCIATE-AC.
//! 3. C-MOVE-RQ command PDV (IVR-LE) + data PDV (IVR-LE query dataset).
//! 4. Zero or more C-MOVE-RSP progress updates (status 0xFF00).
//! 5. Final C-MOVE-RSP with status 0x0000 (Success) or failure code.
//! 6. A-RELEASE-RQ → A-RELEASE-RP.
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

use super::association::{AeTitle, MoveResponse, NetworkingError};
use super::context::AssociationConfig;
use super::command::{
    build_command_pdu, build_dataset_ivr_le, encode_str, encode_ui, encode_us,
    parse_command_response, C_MOVE_RQ, C_MOVE_RSP, HAS_DATASET, IMPLICIT_VR_LE_TS,
    PRIORITY_MEDIUM, STATUS_PENDING, STATUS_PENDING_WARN,
    STUDY_ROOT_MOVE_SOP_CLASS,
};
use super::echo::{find_ctx_id, receive_command_pdv};
use dicom_ul::association::client::ClientAssociationOptions;
use dicom_ul::pdu::{PDataValue, PDataValueType, Pdu};

// ── MoveDestination ───────────────────────────────────────────────────────────

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

// ── C-MOVE ────────────────────────────────────────────────────────────────────

/// Issue a Study-level C-MOVE request to the configured PACS.
///
/// Requests that the PACS transfer the study identified by `study_instance_uid`
/// to the `destination` AE via C-STORE. Progress is tracked via pending
/// C-MOVE-RSP messages.
///
/// Returns a `MoveResponse` aggregating the completion/failure/warning counts
/// from the final C-MOVE-RSP.
pub fn retrieve(
    config: &AssociationConfig,
    destination: &MoveDestination,
    study_instance_uid: &str,
) -> Result<MoveResponse, NetworkingError> {
    let mut assoc = ClientAssociationOptions::new()
        .calling_ae_title(config.calling_ae_title.as_str())
        .called_ae_title(config.called_ae_title.as_str())
        .with_presentation_context(STUDY_ROOT_MOVE_SOP_CLASS, vec![IMPLICIT_VR_LE_TS])
        .establish(&format!("{}:{}", config.host, config.port))
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    let ctx_id = find_ctx_id(&assoc)?;

    // Build C-MOVE-RQ command set (PS3.7 C.4.2.1.1):
    //   (0000,0002) UI = Study Root QR MOVE SOP Class
    //   (0000,0100) US = 0x0021 (C-MOVE-RQ)
    //   (0000,0110) US = 1       (Message ID)
    //   (0000,0600) AE = Move Destination AE title
    //   (0000,0700) US = 0x0002  (Priority: MEDIUM)
    //   (0000,0800) US = 0xFEFF  (Has Dataset)
    let sop_uid_bytes = encode_ui(STUDY_ROOT_MOVE_SOP_CLASS);
    let dest_bytes = encode_str(destination.as_str());
    let cmd_bytes = build_command_pdu(&[
        (0x0000_0002, sop_uid_bytes.as_slice()),
        (0x0000_0100, &encode_us(C_MOVE_RQ)),
        (0x0000_0110, &encode_us(1u16)),
        (0x0000_0600, dest_bytes.as_slice()),
        (0x0000_0700, &encode_us(PRIORITY_MEDIUM)),
        (0x0000_0800, &encode_us(HAS_DATASET)),
    ]);

    // Build C-MOVE query dataset (IVR-LE):
    //   (0008,0052) CS = "STUDY"
    //   (0020,000D) UI = StudyInstanceUID
    let level_bytes = encode_str("STUDY");
    let uid_bytes = encode_ui(study_instance_uid);
    let dataset_bytes = build_dataset_ivr_le(&[
        (0x0008_0052, level_bytes.as_slice()),
        (0x0020_000D, uid_bytes.as_slice()),
    ]);

    // Send command PDV.
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

    // Send query dataset PDV.
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

    // Collect C-MOVE-RSP messages until terminal status.
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

        // Accumulate progress counters from pending responses.
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
            STATUS_PENDING | STATUS_PENDING_WARN => {
                // Sub-operation in progress; continue.
                continue;
            }
            s => {
                // Terminal response (success, warning, or failure).
                break s;
            }
        }
    };

    assoc
        .release()
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    Ok(MoveResponse {
        completed,
        failed,
        warning,
        final_status,
    })
}
