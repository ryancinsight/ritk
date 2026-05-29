//! C-ECHO SCU — DICOM Verification Service Class (PS3.4 §A.5).

use super::association::{EchoResponse, NetworkingError};
use super::context::AssociationConfig;
use super::command::{
    build_command_pdu, parse_command_response, CommandElementValue, C_ECHO_RQ, C_ECHO_RSP,
    EXPLICIT_VR_LE_TS, NO_DATASET, VERIFICATION_SOP_CLASS,
};
use dicom_ul::association::client::ClientAssociationOptions;
use dicom_ul::pdu::{PDataValue, PDataValueType, PresentationContextResultReason, Pdu};
use std::net::TcpStream;

/// Send a C-ECHO to the configured PACS endpoint and return the response.
pub fn echo(config: &AssociationConfig) -> Result<EchoResponse, NetworkingError> {
    let mut assoc = ClientAssociationOptions::new()
        .calling_ae_title(config.calling_ae_title.as_str())
        .called_ae_title(config.called_ae_title.as_str())
        .with_presentation_context(VERIFICATION_SOP_CLASS, vec![EXPLICIT_VR_LE_TS])
        .establish(format!("{}:{}", config.host, config.port))
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    let ctx_id = find_ctx_id(&assoc)?;

    let cmd_bytes = build_command_pdu(&[
        (0x0000_0002, CommandElementValue::Ui(VERIFICATION_SOP_CLASS)),
        (0x0000_0100, CommandElementValue::Us(C_ECHO_RQ)),
        (0x0000_0110, CommandElementValue::Us(1u16)),
        (0x0000_0800, CommandElementValue::Us(NO_DATASET)),
    ]);

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

    let rsp_bytes = receive_command_pdv(&mut assoc, ctx_id)?;
    let rsp = parse_command_response(&rsp_bytes)?;

    if rsp.command_field != C_ECHO_RSP {
        return Err(NetworkingError::Protocol(format!(
            "expected C-ECHO-RSP (0x{:04X}), got 0x{:04X}",
            C_ECHO_RSP, rsp.command_field
        )));
    }

    assoc
        .release()
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    Ok(EchoResponse { status: rsp.status })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Find the first accepted presentation context ID (reason == 0).
///
/// In dicom-ul v0.8.2, `PresentationContextResult` has fields
/// `id`, `reason`, `transfer_syntax` only. `reason == 0` means accepted.
pub(super) fn find_ctx_id(
    assoc: &dicom_ul::association::client::ClientAssociation<TcpStream>,
) -> Result<u8, NetworkingError> {
    assoc
        .presentation_contexts()
        .iter()
        .find(|pc| pc.reason == PresentationContextResultReason::Acceptance)
        .map(|pc| pc.id)
        .ok_or_else(|| NetworkingError::NoPresentationContext(
            "no accepted presentation context in association".to_owned()
        ))
}

/// Collect all command PDV fragments for one command message.
pub(super) fn receive_command_pdv(
    assoc: &mut dicom_ul::association::client::ClientAssociation<TcpStream>,
    ctx_id: u8,
) -> Result<Vec<u8>, NetworkingError> {
    let mut buf = Vec::new();
    loop {
        let pdu = assoc
            .receive()
            .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
        match pdu {
            Pdu::PData { data } => {
                for pdv in data {
                    if pdv.presentation_context_id == ctx_id
                        && pdv.value_type == PDataValueType::Command
                    {
                        buf.extend_from_slice(&pdv.data);
                        if pdv.is_last {
                            return Ok(buf);
                        }
                    }
                }
            }
            other => {
                return Err(NetworkingError::Protocol(format!(
                    "unexpected PDU while waiting for command: {:?}",
                    std::mem::discriminant(&other)
                )));
            }
        }
    }
}

/// Collect all data PDV fragments for one data message.
pub(super) fn receive_data_pdv(
    assoc: &mut dicom_ul::association::client::ClientAssociation<TcpStream>,
    ctx_id: u8,
) -> Result<Vec<u8>, NetworkingError> {
    let mut buf = Vec::new();
    loop {
        let pdu = assoc
            .receive()
            .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
        match pdu {
            Pdu::PData { data } => {
                for pdv in data {
                    if pdv.presentation_context_id == ctx_id
                        && pdv.value_type == PDataValueType::Data
                    {
                        buf.extend_from_slice(&pdv.data);
                        if pdv.is_last {
                            return Ok(buf);
                        }
                    }
                }
            }
            other => {
                return Err(NetworkingError::Protocol(format!(
                    "unexpected PDU while waiting for data: {:?}",
                    std::mem::discriminant(&other)
                )));
            }
        }
    }
}
