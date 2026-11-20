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
//! let dest = MoveDestination::new(AeTitle::new("MY_SCP").expect("infallible: validated precondition"));
//! let resp = retrieve(&config, &dest, "1.2.3.4.5")?;
//! assert_eq!(resp.final_status, 0x0000);
//! ```

use super::association::{networking_error, AeTitle, Association, MoveResponse, NetworkingError};
use super::command::{encode_query_identifier, IMPLICIT_VR_LE_TS, STUDY_ROOT_MOVE_SOP_CLASS};
use super::context::AssociationConfig;

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
    let dataset = encode_query_identifier(level_cs, query_keys.iter().copied());
    let mut association =
        Association::for_service(config, STUDY_ROOT_MOVE_SOP_CLASS, IMPLICIT_VR_LE_TS)?;
    let result = association
        .c_move(STUDY_ROOT_MOVE_SOP_CLASS, destination.as_str(), dataset)
        .map_err(networking_error)?;
    association.release().map_err(networking_error)?;
    Ok(MoveResponse {
        completed: result.completed,
        failed: result.failed,
        warning: result.warning,
        final_status: result.status,
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
