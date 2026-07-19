//! Per-connection handler: association negotiation, C-STORE-RQ processing, and PDU I/O.

use arrayvec::ArrayString;

use super::super::association::NetworkingError;
use super::super::context::transfer_syntax;
use super::super::dimse::{CommandField, DimseMessage, DimseStatus};
use super::super::pdu::{
    AbortPdu, AbortSource, AssociateAcPdu, CommandType, FragmentPosition,
    ImplementationClassUidSubItem, ImplementationVersionNameSubItem, MaximumLengthSubItem,
    MessageControlHeader, PDataTfPdu, Pdu, PresentationContextItemAc, PresentationDataValueItem,
    ReleaseRpPdu, UserInformation, APPLICATION_CONTEXT_NAME, RITK_IMPLEMENTATION_CLASS_UID,
    RITK_IMPLEMENTATION_VERSION,
};
use super::config::{ScpConfig, StoredInstance};
use crate::format::dicom::reader::types::literal_arraystring;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;

// ── Connection handler ────────────────────────────────────────────────────────

pub(super) fn handle_connection(
    mut stream: TcpStream,
    config: &ScpConfig,
    tx: &mpsc::SyncSender<StoredInstance>,
) -> Result<(), NetworkingError> {
    stream
        .set_nonblocking(false)
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
    stream
        .set_read_timeout(Some(config.read_timeout))
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
    stream
        .set_write_timeout(Some(config.read_timeout))
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    // 1. Read A-ASSOCIATE-RQ.
    let rq = match read_pdu_stream(&mut stream)? {
        Pdu::AssociateRq(rq) => rq,
        other => {
            return Err(NetworkingError::Protocol(format!(
                "SCP expected A-ASSOCIATE-RQ, got {other:?}"
            )));
        }
    };

    // 2. Accept every offered presentation context with its first transfer syntax.
    let mut ctx_map: HashMap<u8, (ArrayString<64>, ArrayString<64>)> =
        HashMap::with_capacity(rq.presentation_contexts.len());
    let pc_acs: Vec<PresentationContextItemAc> = rq
        .presentation_contexts
        .iter()
        .map(|pc| {
            let ts = pc
                .transfer_syntax_uids
                .first()
                .cloned()
                .unwrap_or_else(|| literal_arraystring(transfer_syntax::IMPLICIT_VR_LE));
            ctx_map.insert(pc.presentation_context_id, (pc.abstract_syntax_uid, ts));
            PresentationContextItemAc {
                presentation_context_id: pc.presentation_context_id,
                result_reason: 0,
                transfer_syntax_uid: ts,
            }
        })
        .collect();

    // 3. Send A-ASSOCIATE-AC.
    write_pdu_stream(
        &mut stream,
        &Pdu::AssociateAc(AssociateAcPdu {
            protocol_version: 1,
            called_ae_title: rq.called_ae_title,
            calling_ae_title: rq.calling_ae_title,
            application_context_name: literal_arraystring(APPLICATION_CONTEXT_NAME),
            presentation_contexts: pc_acs,
            user_information: UserInformation {
                maximum_length: MaximumLengthSubItem {
                    maximum_length_received: config.max_pdu_length,
                },
                implementation_class_uid: ImplementationClassUidSubItem {
                    implementation_class_uid: literal_arraystring(RITK_IMPLEMENTATION_CLASS_UID),
                },
                implementation_version_name: Some(ImplementationVersionNameSubItem {
                    implementation_version_name: literal_arraystring(RITK_IMPLEMENTATION_VERSION),
                }),
                ..Default::default()
            },
        }),
    )?;

    // 4. Message loop: receive C-STORE-RQs until A-RELEASE-RQ.
    loop {
        match recv_dimse_message(&mut stream)? {
            ScpMessageResult::Released => break,
            ScpMessageResult::Aborted => {
                return Err(NetworkingError::Protocol("remote aborted".to_string()));
            }
            ScpMessageResult::Message { cid, msg } => match msg.command_field() {
                Some(CommandField::CStoreRq) => {
                    handle_store_rq(&mut stream, cid, msg, &ctx_map, tx, config)?;
                }
                other => {
                    let _ = write_pdu_stream(
                        &mut stream,
                        &Pdu::Abort(AbortPdu {
                            source: AbortSource::DicomUlServiceUser,
                        }),
                    );
                    return Err(NetworkingError::Protocol(format!(
                        "SCP unexpected command: {other:?}"
                    )));
                }
            },
        }
    }
    Ok(())
}

// ── C-STORE-RQ handler ────────────────────────────────────────────────────────

fn handle_store_rq(
    stream: &mut TcpStream,
    cid: u8,
    msg: DimseMessage,
    ctx_map: &HashMap<u8, (ArrayString<64>, ArrayString<64>)>,
    tx: &mpsc::SyncSender<StoredInstance>,
    config: &ScpConfig,
) -> Result<(), NetworkingError> {
    let sop_class_uid = msg.affected_sop_class_uid().unwrap_or_default();
    let sop_instance_uid = msg.affected_sop_instance_uid().unwrap_or_default();
    let msg_id = msg.message_id().unwrap_or(1);
    let dataset_bytes = msg.data_set.unwrap_or_default();
    let transfer_syntax_uid = ctx_map.get(&cid).map(|(_, ts)| *ts).unwrap_or_default();

    // Always respond Success — protocol requires a response regardless of channel state.
    let rsp = DimseMessage::c_store_rsp(
        msg_id,
        &sop_class_uid,
        &sop_instance_uid,
        DimseStatus::Success as u16,
    );
    send_command_pdv(stream, cid, &rsp)?;

    let instance = StoredInstance {
        sop_class_uid,
        sop_instance_uid,
        dataset_bytes,
        transfer_syntax_uid,
    };
    if tx.try_send(instance).is_err() {
        tracing::warn!(
            "SCP channel full: instance {} discarded (queue_capacity={})",
            sop_instance_uid,
            config.queue_capacity,
        );
    }
    Ok(())
}

// ── Message reception ─────────────────────────────────────────────────────────

/// Outcome of receiving one DIMSE protocol event.
enum ScpMessageResult {
    /// A complete DIMSE message (command set + optional dataset) was assembled.
    Message { cid: u8, msg: DimseMessage },
    /// SCU sent A-RELEASE-RQ; SCP has already responded with A-RELEASE-RP.
    Released,
    /// SCU sent A-ABORT.
    Aborted,
}

/// Read PDUs until one complete DIMSE message is assembled.
///
/// Handles A-RELEASE-RQ and A-ABORT as domain control-flow outcomes via
/// [`ScpMessageResult`], not as errors. Fragmented command and dataset PDVs
/// spread across multiple P-DATA-TF PDUs are transparently reassembled.
fn recv_dimse_message(stream: &mut TcpStream) -> Result<ScpMessageResult, NetworkingError> {
    let mut cmd_buf: Vec<u8> = Vec::with_capacity(256);
    let mut data_buf: Vec<u8> = Vec::with_capacity(4096);
    let mut cid: u8 = 0;
    let mut cmd_complete = false;

    loop {
        match read_pdu_stream(stream)? {
            Pdu::PDataTf(pd) => {
                let data_last = pd.presentation_data_value_items.iter().any(|p| {
                    p.message_control_header.message_type == CommandType::DataSet
                        && p.message_control_header.fragment_position == FragmentPosition::Last
                });
                for pdv in &pd.presentation_data_value_items {
                    cid = pdv.presentation_context_id;
                    match pdv.message_control_header.message_type {
                        CommandType::Command => {
                            cmd_buf.extend_from_slice(&pdv.data);
                            if pdv.message_control_header.fragment_position
                                == FragmentPosition::Last
                            {
                                cmd_complete = true;
                            }
                        }
                        CommandType::DataSet => {
                            data_buf.extend_from_slice(&pdv.data);
                        }
                    }
                }
                if !cmd_complete {
                    continue;
                }
                let mut msg = DimseMessage::decode_command_set(&cmd_buf)
                    .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
                let has_ds = msg.command_data_set_type().is_some_and(|v| v != 0x0101);
                if !has_ds || data_last {
                    msg.data_set = if data_buf.is_empty() {
                        None
                    } else {
                        Some(std::mem::take(&mut data_buf))
                    };
                    return Ok(ScpMessageResult::Message { cid, msg });
                }
                // Command complete but dataset fragments are still in transit.
                return recv_data_fragments(stream, cid, msg, data_buf);
            }
            Pdu::ReleaseRq(_) => {
                let _ = write_pdu_stream(stream, &Pdu::ReleaseRp(ReleaseRpPdu));
                return Ok(ScpMessageResult::Released);
            }
            Pdu::Abort(_) => return Ok(ScpMessageResult::Aborted),
            _ => {}
        }
    }
}

/// Continue reading P-DATA-TF PDUs until the dataset last-fragment is received.
fn recv_data_fragments(
    stream: &mut TcpStream,
    cid: u8,
    mut msg: DimseMessage,
    mut data_buf: Vec<u8>,
) -> Result<ScpMessageResult, NetworkingError> {
    loop {
        match read_pdu_stream(stream)? {
            Pdu::PDataTf(pd) => {
                for p in &pd.presentation_data_value_items {
                    if p.message_control_header.message_type == CommandType::DataSet {
                        data_buf.extend_from_slice(&p.data);
                        if p.message_control_header.fragment_position == FragmentPosition::Last {
                            msg.data_set = Some(data_buf);
                            return Ok(ScpMessageResult::Message { cid, msg });
                        }
                    }
                }
            }
            Pdu::ReleaseRq(_) => {
                let _ = write_pdu_stream(stream, &Pdu::ReleaseRp(ReleaseRpPdu));
                return Ok(ScpMessageResult::Released);
            }
            Pdu::Abort(_) => return Ok(ScpMessageResult::Aborted),
            _ => {}
        }
    }
}

// ── PDU I/O helpers ───────────────────────────────────────────────────────────

pub(super) fn read_pdu_stream(stream: &mut TcpStream) -> Result<Pdu, NetworkingError> {
    let mut hdr = [0u8; 6];
    stream
        .read_exact(&mut hdr)
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
    let body_len = u32::from_be_bytes([hdr[2], hdr[3], hdr[4], hdr[5]]) as usize;
    let mut buf = Vec::with_capacity(6 + body_len);
    buf.extend_from_slice(&hdr);
    buf.resize(6 + body_len, 0);
    stream
        .read_exact(&mut buf[6..])
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
    Pdu::decode(&buf).map_err(|e| NetworkingError::Protocol(e.to_string()))
}

pub(super) fn write_pdu_stream(stream: &mut TcpStream, pdu: &Pdu) -> Result<(), NetworkingError> {
    stream
        .write_all(&pdu.encode())
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
    stream
        .flush()
        .map_err(|e| NetworkingError::Protocol(e.to_string()))
}

/// Send a command-only response PDV in a single P-DATA-TF PDU.
///
/// C-STORE-RSP has CommandDataSetType = 0x0101 (no dataset), so the command
/// bytes always fit in a single P-DATA-TF without fragmentation.
pub(super) fn send_command_pdv(
    stream: &mut TcpStream,
    cid: u8,
    msg: &DimseMessage,
) -> Result<(), NetworkingError> {
    let cmd_bytes = msg.encode_command_set();
    write_pdu_stream(
        stream,
        &Pdu::PDataTf(PDataTfPdu {
            presentation_data_value_items: vec![PresentationDataValueItem {
                presentation_context_id: cid,
                message_control_header: MessageControlHeader {
                    message_type: CommandType::Command,
                    fragment_position: FragmentPosition::Last,
                },
                data: cmd_bytes,
            }],
        }),
    )
}
