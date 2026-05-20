//! DICOM Association SCU — association lifecycle per PS 3.8.
//!
//! Establishes a TCP-level association with a remote SCP, exchanges PDU
//! frames, sends/receives DIMSE messages over negotiated presentation
//! contexts, and releases or aborts the association.

use super::dimse::*;
use super::pdu::*;
use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use thiserror::Error;

// ── Transfer Syntax UIDs ──────────────────────────────────────────────────────
pub mod transfer_syntax {
    pub const IMPLICIT_VR_LE: &str = "1.2.840.10008.1.2";
    pub const EXPLICIT_VR_LE: &str = "1.2.840.10008.1.2.1";
    pub const EXPLICIT_VR_BE: &str = "1.2.840.10008.1.2.2";
    pub const JPEG_BASELINE: &str = "1.2.840.10008.1.2.4.50";
    pub const JPEG_LOSSLESS: &str = "1.2.840.10008.1.2.4.70";
    pub const JPEG_LS_LOSSLESS: &str = "1.2.840.10008.1.2.4.80";
    pub const JPEG_2000_LOSSLESS: &str = "1.2.840.10008.1.2.4.90";
    pub const JPEG_2000: &str = "1.2.840.10008.1.2.4.91";
}

// ── Configuration ─────────────────────────────────────────────────────────────
/// Configuration for a DICOM Association.
#[derive(Debug, Clone)]
pub struct AssociationConfig {
    /// Called AE Title (remote PACS/server).
    pub called_ae_title: String,
    /// Calling AE Title (our AE Title).
    pub calling_ae_title: String,
    /// Remote host address.
    pub host: String,
    /// Remote port (typically 104).
    pub port: u16,
    /// Maximum PDU length we can receive.
    pub max_pdu_length: u32,
    /// Timeout for read/write operations.
    pub timeout: Duration,
    /// Requested presentation contexts.
    pub presentation_contexts: Vec<RequestedPresentationContext>,
    /// User identity (optional, for PACS requiring authentication).
    pub user_identity: Option<UserIdentity>,
}

impl Default for AssociationConfig {
    fn default() -> Self {
        Self {
            called_ae_title: "ANYSCP".to_string(),
            calling_ae_title: "RITK".to_string(),
            host: "127.0.0.1".to_string(),
            port: 104,
            max_pdu_length: DEFAULT_MAXIMUM_LENGTH,
            timeout: Duration::from_secs(30),
            presentation_contexts: Vec::new(),
            user_identity: None,
        }
    }
}

impl AssociationConfig {
    /// Legacy constructor from `AeTitle` and `DicomAddress`.
    pub fn new(calling_ae_title: AeTitle, remote: DicomAddress) -> Self {
        Self {
            called_ae_title: remote.ae_title.as_str().to_string(),
            calling_ae_title: calling_ae_title.as_str().to_string(),
            host: remote.host.clone(),
            port: remote.port,
            max_pdu_length: DEFAULT_MAXIMUM_LENGTH,
            timeout: Duration::from_secs(30),
            presentation_contexts: Vec::new(),
            user_identity: None,
        }
    }
    pub fn with_connect_timeout(mut self, t: Duration) -> Self { self.timeout = t; self }
    pub fn with_read_timeout(mut self, t: Duration) -> Self { self.timeout = t; self }
}

/// A requested presentation context for association negotiation.
#[derive(Debug, Clone)]
pub struct RequestedPresentationContext {
    /// Abstract Syntax (SOP Class) UID.
    pub abstract_syntax_uid: String,
    /// Requested Transfer Syntax UIDs (in order of preference).
    pub transfer_syntax_uids: Vec<String>,
}

// ── Negotiated context ────────────────────────────────────────────────────────
/// A negotiated presentation context.
#[derive(Debug, Clone)]
pub struct NegotiatedContext {
    pub presentation_context_id: u8,
    pub abstract_syntax_uid: String,
    pub transfer_syntax_uid: String,
}

// ── Operation results ─────────────────────────────────────────────────────────
/// Result of a C-FIND operation.
#[derive(Debug, Clone)]
pub struct FindResult {
    /// Individual matching results (one per C-FIND-RSP with Status=Pending).
    pub matches: Vec<Vec<u8>>,
    /// Final status code.
    pub status: u16,
}

/// Result of a C-MOVE operation.
#[derive(Debug, Clone)]
pub struct MoveResult {
    /// Number of completed sub-operations.
    pub completed: u16,
    /// Number of failed sub-operations.
    pub failed: u16,
    /// Number of warning sub-operations.
    pub warning: u16,
    /// Remaining sub-operations (if still in progress).
    pub remaining: u16,
    /// Final status code.
    pub status: u16,
}

// ── Legacy types for sibling module compatibility ──────────────────────────────
/// Errors produced by the DIMSE SCU layer.
#[derive(Debug, Error)]
pub enum NetworkingError {
    #[error("invalid AE title '{0}': {1}")]
    InvalidAeTitle(String, &'static str),
    #[error("TCP connection failed: {0}")]
    Connection(#[from] std::io::Error),
    #[error("association rejected: rejection_source={rejection_source} reason={reason}")]
    AssociationRejected { rejection_source: u8, reason: u8 },
    #[error("no accepted presentation context for abstract syntax {0}")]
    NoPresentationContext(String),
    #[error("DIMSE protocol error: {0}")]
    Protocol(String),
    #[error("unexpected DIMSE status 0x{0:04X}")]
    UnexpectedStatus(u16),
    #[error("PDU parse error: {0}")]
    ParseError(String),
}

/// Validated DICOM Application Entity Title (PS3.7 §7.1.3).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AeTitle(String);

impl AeTitle {
    pub fn new(s: &str) -> Result<Self, NetworkingError> {
        if s.is_empty() || s.len() > 16 {
            return Err(NetworkingError::InvalidAeTitle(
                s.to_owned(), "length must be 1-16 characters"));
        }
        if s.bytes().any(|b| b < 0x20 || b >= 0x7F || b == b'\\') {
            return Err(NetworkingError::InvalidAeTitle(
                s.to_owned(), "must be printable ASCII excluding backslash"));
        }
        Ok(Self(s.to_owned()))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl std::fmt::Display for AeTitle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}

impl TryFrom<&str> for AeTitle {
    type Error = NetworkingError;
    fn try_from(s: &str) -> Result<Self, Self::Error> { Self::new(s) }
}

/// Remote DICOM endpoint: host, port, called AE title.
#[derive(Debug, Clone)]
pub struct DicomAddress {
    pub host: String,
    pub port: u16,
    pub ae_title: AeTitle,
}

impl DicomAddress {
    pub fn new(host: impl Into<String>, port: u16, ae_title: AeTitle) -> Self {
        Self { host: host.into(), port, ae_title }
    }
    pub fn socket_addr(&self) -> String { format!("{}:{}", self.host, self.port) }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EchoResponse { pub status: u16 }

#[derive(Debug, Clone)]
pub struct StoreResponse {
    pub status: u16,
    pub affected_sop_instance_uid: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveResponse {
    pub completed: u16,
    pub failed: u16,
    pub warning: u16,
    pub final_status: u16,
}

// ── Association ───────────────────────────────────────────────────────────────
/// An established DICOM Association.
///
/// Holds the TCP stream and negotiated association parameters.
/// Created by `Association::connect()` and provides methods for
/// sending/receiving DIMSE messages.
pub struct Association {
    stream: TcpStream,
    config: AssociationConfig,
    /// Negotiated presentation contexts (indexed by ID).
    negotiated_contexts: Vec<NegotiatedContext>,
    /// Next available presentation context ID (odd numbers only per PS 3.8).
    next_context_id: u8,
    /// Maximum PDU length the remote peer can receive.
    pub remote_max_pdu_length: u32,
    /// Whether the association is still active.
    active: bool,
}

impl Association {
    /// Establish a DICOM association by connecting to the remote peer.
    ///
    /// 1. Open TCP connection
    /// 2. Send A-ASSOCIATE-RQ
    /// 3. Receive A-ASSOCIATE-AC (or A-ASSOCIATE-RJ)
    /// 4. Return the established Association
    pub fn connect(config: AssociationConfig) -> Result<Self> {
        let addr = format!("{}:{}", config.host, config.port);
        let stream = TcpStream::connect_timeout(&addr.parse()?, config.timeout)
            .with_context(|| format!("TCP connect to {}", addr))?;
        stream.set_read_timeout(Some(config.timeout))?;
        stream.set_write_timeout(Some(config.timeout))?;

        let mut next_id: u8 = 1;
        let mut pc_items = Vec::new();
        for rpc in &config.presentation_contexts {
            let mut ts_uids: Vec<String> = rpc.transfer_syntax_uids.clone();
            // Always include Implicit VR LE as fallback if not present.
            if !ts_uids.iter().any(|t| t == transfer_syntax::IMPLICIT_VR_LE) {
                ts_uids.push(transfer_syntax::IMPLICIT_VR_LE.to_string());
            }
            pc_items.push(PresentationContextItemRq {
                presentation_context_id: next_id,
                abstract_syntax_uid: rpc.abstract_syntax_uid.clone(),
                transfer_syntax_uids: ts_uids,
            });
            next_id = next_id.checked_add(2)
                .ok_or_else(|| anyhow::anyhow!("presentation context ID overflow"))?;
        }

        let user_info = UserInformation {
            maximum_length: MaximumLengthSubItem {
                maximum_length_received: config.max_pdu_length,
            },
            implementation_class_uid: ImplementationClassUidSubItem {
                implementation_class_uid: RITK_IMPLEMENTATION_CLASS_UID.to_string(),
            },
            implementation_version_name: Some(ImplementationVersionNameSubItem {
                implementation_version_name: RITK_IMPLEMENTATION_VERSION.to_string(),
            }),
            user_identity: config.user_identity.clone(),
            ..Default::default()
        };

        let rq = Pdu::AssociateRq(AssociateRqPdu {
            protocol_version: 1,
            called_ae_title: config.called_ae_title.clone(),
            calling_ae_title: config.calling_ae_title.clone(),
            application_context_name: APPLICATION_CONTEXT_NAME.to_string(),
            presentation_contexts: pc_items,
            user_information: user_info,
        });

        let mut assoc = Self {
            stream,
            config,
            negotiated_contexts: Vec::new(),
            next_context_id: next_id,
            remote_max_pdu_length: DEFAULT_MAXIMUM_LENGTH,
            active: false,
        };
        assoc.send_pdu(&rq)?;

        let resp = assoc.recv_pdu()?;
        match resp {
            Pdu::AssociateAc(ac) => {
                // Extract negotiated contexts — only accepted (result == 0).
                let mut negotiated = Vec::new();
                // Build a map from rq: id → abstract_syntax_uid.
                let rq_map: std::collections::HashMap<u8, String> = rq_iter_abstracts(&rq);
                for pc_ac in &ac.presentation_contexts {
                    if pc_ac.result_reason == 0 {
                        let abstract_uid = rq_map.get(&pc_ac.presentation_context_id)
                            .cloned()
                            .unwrap_or_default();
                        negotiated.push(NegotiatedContext {
                            presentation_context_id: pc_ac.presentation_context_id,
                            abstract_syntax_uid: abstract_uid,
                            transfer_syntax_uid: pc_ac.transfer_syntax_uid.clone(),
                        });
                    }
                }
                assoc.remote_max_pdu_length =
                    ac.user_information.maximum_length.maximum_length_received;
                assoc.negotiated_contexts = negotiated;
                assoc.active = true;
                Ok(assoc)
            }
            Pdu::AssociateRj(rj) => {
                bail!("association rejected: result={:?} source={:?} reason={}",
                      rj.result, rj.source, rj.reason)
            }
            other => bail!("unexpected PDU in response to A-ASSOCIATE-RQ: {:?}", other),
        }
    }

    /// Send a C-ECHO and wait for the response. Returns the status code.
    pub fn c_echo(&mut self) -> Result<u16> {
        let ctx_id = self.context_for_sop_class(
            sop_class::VERIFICATION, &[transfer_syntax::IMPLICIT_VR_LE])?;
        let msg_id = self.next_message_id();
        let rq = DimseMessage::c_echo_rq(msg_id);
        self.send_message(ctx_id, &rq)?;
        let (_, rsp) = self.recv_message()?;
        let cf = rsp.command_field()
            .context("missing CommandField in C-ECHO-RSP")?;
        if cf != CommandField::CEchoRsp {
            bail!("expected C-ECHO-RSP, got {:?}", cf);
        }
        rsp.status().context("missing Status in C-ECHO-RSP")
    }

    /// Send a C-FIND request.
    pub fn c_find(&mut self, sop_class_uid: &str, identifier: Vec<u8>) -> Result<FindResult> {
        let ctx_id = self.context_for_sop_class(
            sop_class_uid, &[transfer_syntax::IMPLICIT_VR_LE])?;
        let msg_id = self.next_message_id();
        let rq = DimseMessage::c_find_rq(msg_id, sop_class_uid, identifier);
        self.send_message(ctx_id, &rq)?;

        let mut matches = Vec::new();
        let final_status: u16;
        loop {
            let (_, rsp) = self.recv_message()?;
            let status = rsp.status().context("missing Status in C-FIND-RSP")?;
            let is_pending = status == DimseStatus::Pending as u16
                || status == DimseStatus::PendingWarning as u16;
            if is_pending {
                if let Some(ref data) = rsp.data_set {
                    matches.push(data.clone());
                }
            } else {
                final_status = status;
                break;
            }
        }
        Ok(FindResult { matches, status: final_status })
    }

    /// Send a C-STORE request. Returns the status code.
    pub fn c_store(&mut self, sop_class_uid: &str, sop_instance_uid: &str, data_set: Vec<u8>) -> Result<u16> {
        let ctx_id = self.context_for_sop_class(
            sop_class_uid, &[transfer_syntax::IMPLICIT_VR_LE,
                             transfer_syntax::EXPLICIT_VR_LE])?;
        let msg_id = self.next_message_id();
        let rq = DimseMessage::c_store_rq(msg_id, sop_class_uid, sop_instance_uid, 0x0000, data_set);
        self.send_message(ctx_id, &rq)?;
        let (_, rsp) = self.recv_message()?;
        let cf = rsp.command_field()
            .context("missing CommandField in C-STORE-RSP")?;
        if cf != CommandField::CStoreRsp {
            bail!("expected C-STORE-RSP, got {:?}", cf);
        }
        rsp.status().context("missing Status in C-STORE-RSP")
    }

    /// Send a C-MOVE request.
    pub fn c_move(&mut self, sop_class_uid: &str, move_destination: &str, identifier: Vec<u8>) -> Result<MoveResult> {
        let ctx_id = self.context_for_sop_class(
            sop_class_uid, &[transfer_syntax::IMPLICIT_VR_LE])?;
        let msg_id = self.next_message_id();
        let rq = DimseMessage::c_move_rq(msg_id, sop_class_uid, move_destination, identifier);
        self.send_message(ctx_id, &rq)?;

        let mut completed: u16 = 0;
        let mut failed: u16 = 0;
        let mut warning: u16 = 0;
        let mut remaining: u16 = 0;
        let final_status: u16;
        loop {
            let (_, rsp) = self.recv_message()?;
            let status = rsp.status().context("missing Status in C-MOVE-RSP")?;
            // Extract sub-operation counts from optional command elements.
            if let Some(elem) = rsp.find_element((0x0000, 0x1021)) {
                if elem.value.len() >= 2 {
                    completed = u16::from_le_bytes([elem.value[0], elem.value[1]]);
                }
            }
            if let Some(elem) = rsp.find_element((0x0000, 0x1022)) {
                if elem.value.len() >= 2 {
                    failed = u16::from_le_bytes([elem.value[0], elem.value[1]]);
                }
            }
            if let Some(elem) = rsp.find_element((0x0000, 0x1023)) {
                if elem.value.len() >= 2 {
                    warning = u16::from_le_bytes([elem.value[0], elem.value[1]]);
                }
            }
            if let Some(elem) = rsp.find_element((0x0000, 0x1020)) {
                if elem.value.len() >= 2 {
                    remaining = u16::from_le_bytes([elem.value[0], elem.value[1]]);
                }
            }
            let is_pending = status == DimseStatus::Pending as u16
                || status == DimseStatus::PendingWarning as u16;
            if !is_pending {
                final_status = status;
                break;
            }
        }
        Ok(MoveResult { completed, failed, warning, remaining, status: final_status })
    }

    /// Release the association gracefully.
    pub fn release(mut self) -> Result<()> {
        if !self.active { return Ok(()); }
        self.send_pdu(&Pdu::ReleaseRq(ReleaseRqPdu))?;
        let resp = self.recv_pdu()?;
        match resp {
            Pdu::ReleaseRp(_) => {},
            other => bail!("expected A-RELEASE-RP, got {:?}", other),
        }
        self.active = false;
        Ok(())
    }

    /// Abort the association (ungraceful).
    pub fn abort(mut self) -> Result<()> {
        if !self.active { return Ok(()); }
        let _ = self.send_pdu(&Pdu::Abort(AbortPdu {
            source: AbortSource::DicomUlServiceUser,
        }));
        self.active = false;
        Ok(())
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Send a PDU over the TCP stream.
    fn send_pdu(&mut self, pdu: &Pdu) -> Result<()> {
        let bytes = pdu.encode();
        self.stream.write_all(&bytes)
            .with_context(|| "PDU write")?;
        self.stream.flush()?;
        Ok(())
    }

    /// Receive a PDU from the TCP stream.
    fn recv_pdu(&mut self) -> Result<Pdu> {
        let mut hdr = [0u8; 6];
        self.stream.read_exact(&mut hdr)
            .with_context(|| "PDU header read")?;
        let len = u32::from_be_bytes([hdr[2], hdr[3], hdr[4], hdr[5]]) as usize;
        let mut body = vec![0u8; len];
        self.stream.read_exact(&mut body)
            .with_context(|| "PDU body read")?;
        let mut full = Vec::with_capacity(6 + len);
        full.extend_from_slice(&hdr);
        full.extend_from_slice(&body);
        Pdu::decode(&full).with_context(|| "PDU decode")
    }

    /// Send a DIMSE message (command set + optional data set) over the association.
    fn send_message(&mut self, context_id: u8, message: &DimseMessage) -> Result<()> {
        let cmd_bytes = message.encode_command_set();
        // Fragment command PDVs.
        let max_data = self.remote_max_pdu_length.saturating_sub(6) as usize;
        let max_data = if max_data == 0 { DEFAULT_MAXIMUM_LENGTH as usize - 6 } else { max_data };
        let max_data = max_data.max(1);
        self.fragment_and_send(context_id, &cmd_bytes, CommandType::Command, max_data)?;
        // Fragment data set PDVs if present.
        if let Some(ref ds) = message.data_set {
            self.fragment_and_send(context_id, ds, CommandType::DataSet, max_data)?;
        }
        Ok(())
    }

    /// Fragment data into PDV items and send each as a P-DATA-TF PDU.
    fn fragment_and_send(&mut self, context_id: u8, data: &[u8], cmd_type: CommandType, max_data: usize) -> Result<()> {
        if data.is_empty() {
            // Send single empty PDV with last bit set.
            let pdv = PresentationDataValueItem {
                presentation_context_id: context_id,
                message_control_header: MessageControlHeader {
                    message_type: cmd_type, last_fragment: true },
                data: Vec::new(),
            };
            self.send_pdu(&Pdu::PDataTf(PDataTfPdu {
                presentation_data_value_items: vec![pdv] }))?;
            return Ok(());
        }
        let chunks: Vec<&[u8]> = data.chunks(max_data).collect();
        let last_idx = chunks.len() - 1;
        for (i, chunk) in chunks.iter().enumerate() {
            let pdv = PresentationDataValueItem {
                presentation_context_id: context_id,
                message_control_header: MessageControlHeader {
                    message_type: cmd_type,
                    last_fragment: i == last_idx,
                },
                data: chunk.to_vec(),
            };
            self.send_pdu(&Pdu::PDataTf(PDataTfPdu {
                presentation_data_value_items: vec![pdv] }))?;
        }
        Ok(())
    }

    /// Receive a DIMSE message from the association.
    fn recv_message(&mut self) -> Result<(u8, DimseMessage)> {
        let mut cmd_buf = Vec::new();
        let mut data_buf = Vec::new();
        let mut ctx_id: u8 = 0;
        let mut cmd_last = false;
        loop {
            let pdu = self.recv_pdu()?;
            match pdu {
                Pdu::PDataTf(pd) => {
                    let last_data_arrived = pdv_last_data(&pd);
                    for pdv in &pd.presentation_data_value_items {
                        ctx_id = pdv.presentation_context_id;
                        match pdv.message_control_header.message_type {
                            CommandType::Command => {
                                cmd_buf.extend_from_slice(&pdv.data);
                                if pdv.message_control_header.last_fragment {
                                    cmd_last = true;
                                }
                            }
                            CommandType::DataSet => {
                                data_buf.extend_from_slice(&pdv.data);
                            }
                        }
                    }
                    if cmd_last {
                        let mut msg = DimseMessage::decode_command_set(&cmd_buf)?;
                        let has_ds = msg.command_data_set_type().map_or(false, |v| v != 0x0101);
                        if has_ds && !last_data_arrived {
                            // Read remaining P-DATA-TF PDUs until last data fragment.
                            loop {
                                let pdu2 = self.recv_pdu()?;
                                match pdu2 {
                                    Pdu::PDataTf(pd2) => {
                                        for pdv2 in pd2.presentation_data_value_items {
                                            if pdv2.message_control_header.message_type == CommandType::DataSet {
                                                data_buf.extend_from_slice(&pdv2.data);
                                                if pdv2.message_control_header.last_fragment {
                                                    msg.data_set = Some(data_buf);
                                                    return Ok((ctx_id, msg));
                                                }
                                            }
                                        }
                                    }
                                    Pdu::ReleaseRq(_) => {
                                        let _ = self.send_pdu(&Pdu::ReleaseRp(ReleaseRpPdu));
                                        bail!("remote released association during data transfer");
                                    }
                                    Pdu::Abort(_) => bail!("remote aborted association"),
                                    other => bail!("unexpected PDU during data recv: {:?}", other),
                                }
                            }
                        }
                        msg.data_set = if data_buf.is_empty() { None } else { Some(data_buf) };
                        return Ok((ctx_id, msg));
                    }
                }
                Pdu::ReleaseRq(_) => {
                    let _ = self.send_pdu(&Pdu::ReleaseRp(ReleaseRpPdu));
                    bail!("remote released association");
                }
                Pdu::Abort(_) => bail!("remote aborted association"),
                other => bail!("unexpected PDU: {:?}", other),
            }
        }
    }

    /// Find the negotiated context for a given abstract syntax UID.
    fn find_context(&self, abstract_syntax_uid: &str) -> Option<&NegotiatedContext> {
        self.negotiated_contexts.iter()
            .find(|c| c.abstract_syntax_uid == abstract_syntax_uid)
    }

    /// Get or create a presentation context ID for a SOP class.
    fn context_for_sop_class(&mut self, sop_class_uid: &str, _transfer_syntax_uids: &[&str]) -> Result<u8> {
        if let Some(ctx) = self.find_context(sop_class_uid) {
            return Ok(ctx.presentation_context_id);
        }
        bail!("no negotiated presentation context for SOP class {}", sop_class_uid)
    }

    /// Monotonic message ID counter (wraps at u16::MAX).
    fn next_message_id(&mut self) -> u16 {
        let id = self.next_context_id as u16;
        self.next_context_id = self.next_context_id.wrapping_add(1);
        if self.next_context_id == 0 { self.next_context_id = 1; }
        id
    }

    // ── Test-exposed helpers ──────────────────────────────────────────────

    /// Build an A-ASSOCIATE-RQ PDU from the config (exposed for testing).
    pub fn build_associate_rq(config: &AssociationConfig) -> Pdu {
        let mut next_id: u8 = 1;
        let mut pc_items = Vec::new();
        for rpc in &config.presentation_contexts {
            let mut ts_uids: Vec<String> = rpc.transfer_syntax_uids.clone();
            if !ts_uids.iter().any(|t| t == transfer_syntax::IMPLICIT_VR_LE) {
                ts_uids.push(transfer_syntax::IMPLICIT_VR_LE.to_string());
            }
            pc_items.push(PresentationContextItemRq {
                presentation_context_id: next_id,
                abstract_syntax_uid: rpc.abstract_syntax_uid.clone(),
                transfer_syntax_uids: ts_uids,
            });
            next_id += 2;
        }
        let user_info = UserInformation {
            maximum_length: MaximumLengthSubItem {
                maximum_length_received: config.max_pdu_length,
            },
            implementation_class_uid: ImplementationClassUidSubItem {
                implementation_class_uid: RITK_IMPLEMENTATION_CLASS_UID.to_string(),
            },
            implementation_version_name: Some(ImplementationVersionNameSubItem {
                implementation_version_name: RITK_IMPLEMENTATION_VERSION.to_string(),
            }),
            user_identity: config.user_identity.clone(),
            ..Default::default()
        };
        Pdu::AssociateRq(AssociateRqPdu {
            protocol_version: 1,
            called_ae_title: config.called_ae_title.clone(),
            calling_ae_title: config.calling_ae_title.clone(),
            application_context_name: APPLICATION_CONTEXT_NAME.to_string(),
            presentation_contexts: pc_items,
            user_information: user_info,
        })
    }

    /// Extract negotiated contexts from an A-ASSOCIATE-AC PDU given the RQ
    /// abstract syntax map (id → abstract_syntax_uid).
    pub fn negotiated_contexts_from_ac(
        ac: &AssociateAcPdu,
        rq_abstracts: &std::collections::HashMap<u8, String>,
    ) -> Vec<NegotiatedContext> {
        ac.presentation_contexts.iter()
            .filter(|pc| pc.result_reason == 0)
            .map(|pc| NegotiatedContext {
                presentation_context_id: pc.presentation_context_id,
                abstract_syntax_uid: rq_abstracts.get(&pc.presentation_context_id)
                    .cloned().unwrap_or_default(),
                transfer_syntax_uid: pc.transfer_syntax_uid.clone(),
            })
            .collect()
    }

    /// Fragment data into PDV items (no I/O). Returns `(data_len, Vec<PDV>)`.
    pub fn fragment_pdvs(data: &[u8], cmd_type: CommandType, max_data: usize) -> Vec<PresentationDataValueItem> {
        let context_id = 1u8; // placeholder for fragmentation logic test
        if data.is_empty() {
            return vec![PresentationDataValueItem {
                presentation_context_id: context_id,
                message_control_header: MessageControlHeader { message_type: cmd_type, last_fragment: true },
                data: Vec::new(),
            }];
        }
        let chunks: Vec<&[u8]> = data.chunks(max_data).collect();
        let last_idx = chunks.len() - 1;
        chunks.iter().enumerate().map(|(i, chunk)| {
            PresentationDataValueItem {
                presentation_context_id: context_id,
                message_control_header: MessageControlHeader {
                    message_type: cmd_type,
                    last_fragment: i == last_idx,
                },
                data: chunk.to_vec(),
            }
        }).collect()
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

/// Extract abstract syntax UIDs from an A-ASSOCIATE-RQ PDU.
fn rq_iter_abstracts(pdu: &Pdu) -> std::collections::HashMap<u8, String> {
    let mut map = std::collections::HashMap::new();
    if let Pdu::AssociateRq(rq) = pdu {
        for pc in &rq.presentation_contexts {
            map.insert(pc.presentation_context_id, pc.abstract_syntax_uid.clone());
        }
    }
    map
}

/// Check if any PDV in a P-Data-TF is a last data-set fragment.
fn pdv_last_data(pd: &PDataTfPdu) -> bool {
    pd.presentation_data_value_items.iter()
        .any(|pdv| pdv.message_control_header.message_type == CommandType::DataSet
            && pdv.message_control_header.last_fragment)
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = AssociationConfig::default();
        assert_eq!(cfg.port, 104);
        assert_eq!(cfg.max_pdu_length, 16384);
        assert_eq!(cfg.called_ae_title, "ANYSCP");
        assert_eq!(cfg.calling_ae_title, "RITK");
    }

    #[test]
    fn test_requested_context_odd_ids() {
        let cfg = AssociationConfig {
            presentation_contexts: vec![
                RequestedPresentationContext {
                    abstract_syntax_uid: sop_class::VERIFICATION.to_string(),
                    transfer_syntax_uids: vec![transfer_syntax::IMPLICIT_VR_LE.to_string()],
                },
                RequestedPresentationContext {
                    abstract_syntax_uid: sop_class::FIND_STUDY.to_string(),
                    transfer_syntax_uids: vec![transfer_syntax::IMPLICIT_VR_LE.to_string()],
                },
                RequestedPresentationContext {
                    abstract_syntax_uid: sop_class::MOVE_STUDY.to_string(),
                    transfer_syntax_uids: vec![transfer_syntax::IMPLICIT_VR_LE.to_string()],
                },
            ],
            ..Default::default()
        };
        let pdu = Association::build_associate_rq(&cfg);
        if let Pdu::AssociateRq(rq) = pdu {
            let ids: Vec<u8> = rq.presentation_contexts.iter()
                .map(|pc| pc.presentation_context_id).collect();
            assert_eq!(ids, vec![1, 3, 5]);
            assert!(ids.iter().all(|id| id % 2 == 1));
        }
    }

    #[test]
    fn test_build_associate_rq() {
        let cfg = AssociationConfig {
            called_ae_title: "TESTSCP".to_string(),
            presentation_contexts: vec![
                RequestedPresentationContext {
                    abstract_syntax_uid: sop_class::VERIFICATION.to_string(),
                    transfer_syntax_uids: vec![transfer_syntax::EXPLICIT_VR_LE.to_string()],
                },
            ],
            ..Default::default()
        };
        let pdu = Association::build_associate_rq(&cfg);
        if let Pdu::AssociateRq(rq) = pdu {
            assert_eq!(rq.presentation_contexts.len(), 1);
            assert_eq!(rq.called_ae_title, "TESTSCP");
            assert_eq!(rq.presentation_contexts[0].presentation_context_id, 1);
            // Implicit VR LE should be appended as fallback.
            assert_eq!(rq.presentation_contexts[0].transfer_syntax_uids.len(), 2);
            assert_eq!(rq.presentation_contexts[0].transfer_syntax_uids[1], transfer_syntax::IMPLICIT_VR_LE);
        }
        // Round-trip encode/decode.
        let bytes = pdu.encode();
        let decoded = Pdu::decode(&bytes).unwrap();
        assert_eq!(pdu, decoded);
    }

    #[test]
    fn test_fragment_pdv_single() {
        let data = vec![0xABu8; 100];
        let pdvs = Association::fragment_pdvs(&data, CommandType::Command, 16378);
        assert_eq!(pdvs.len(), 1);
        assert!(pdvs[0].message_control_header.last_fragment);
        assert_eq!(pdvs[0].data.len(), 100);
    }

    #[test]
    fn test_fragment_pdv_multiple() {
        // 40000 bytes with max_data = 8186 → ceil(40000/8186) = 5 fragments.
        // But max_pdu_length=8192 → max_data = 8192 - 6 = 8186.
        let data = vec![0xCDu8; 40000];
        let pdvs = Association::fragment_pdvs(&data, CommandType::DataSet, 8186);
        assert!(pdvs.len() >= 5);
        // Only last fragment has last_fragment=true.
        for (i, pdv) in pdvs.iter().enumerate() {
            if i < pdvs.len() - 1 {
                assert!(!pdv.message_control_header.last_fragment);
            } else {
                assert!(pdv.message_control_header.last_fragment);
            }
        }
        // Total data preserved.
        let total: usize = pdvs.iter().map(|p| p.data.len()).sum();
        assert_eq!(total, 40000);
    }

    #[test]
    fn test_find_context() {
        let assoc = Association {
            stream: TcpStream::new().unwrap(), // dummy, never used for I/O
            config: AssociationConfig::default(),
            negotiated_contexts: vec![
                NegotiatedContext {
                    presentation_context_id: 1,
                    abstract_syntax_uid: sop_class::VERIFICATION.to_string(),
                    transfer_syntax_uid: transfer_syntax::IMPLICIT_VR_LE.to_string(),
                },
                NegotiatedContext {
                    presentation_context_id: 3,
                    abstract_syntax_uid: sop_class::FIND_STUDY.to_string(),
                    transfer_syntax_uid: transfer_syntax::EXPLICIT_VR_LE.to_string(),
                },
            ],
            next_context_id: 7,
            remote_max_pdu_length: 16384,
            active: true,
        };
        let ctx = assoc.find_context(sop_class::VERIFICATION).unwrap();
        assert_eq!(ctx.presentation_context_id, 1);
        assert_eq!(ctx.transfer_syntax_uid, transfer_syntax::IMPLICIT_VR_LE);
        assert!(assoc.find_context("1.2.3.4.5.6").is_none());
    }

    #[test]
    fn test_negotiated_context_from_ac() {
        let ac = AssociateAcPdu {
            protocol_version: 1,
            called_ae_title: "SCP".to_string(),
            calling_ae_title: "RITK".to_string(),
            application_context_name: APPLICATION_CONTEXT_NAME.to_string(),
            presentation_contexts: vec![
                PresentationContextItemAc {
                    presentation_context_id: 1,
                    result_reason: 0,
                    transfer_syntax_uid: transfer_syntax::IMPLICIT_VR_LE.to_string(),
                },
                PresentationContextItemAc {
                    presentation_context_id: 3,
                    result_reason: 1, // rejected
                    transfer_syntax_uid: transfer_syntax::EXPLICIT_VR_LE.to_string(),
                },
            ],
            user_information: UserInformation::default(),
        };
        let mut rq_map = std::collections::HashMap::new();
        rq_map.insert(1u8, sop_class::VERIFICATION.to_string());
        rq_map.insert(3u8, sop_class::FIND_STUDY.to_string());
        let negotiated = Association::negotiated_contexts_from_ac(&ac, &rq_map);
        assert_eq!(negotiated.len(), 1);
        assert_eq!(negotiated[0].presentation_context_id, 1);
        assert_eq!(negotiated[0].abstract_syntax_uid, sop_class::VERIFICATION);
        assert_eq!(negotiated[0].transfer_syntax_uid, transfer_syntax::IMPLICIT_VR_LE);
    }

    #[test]
    fn test_transfer_syntax_uids() {
        // All transfer syntax UIDs must start with "1.2.840.10008" and be non-empty.
        let uids = [
            transfer_syntax::IMPLICIT_VR_LE,
            transfer_syntax::EXPLICIT_VR_LE,
            transfer_syntax::EXPLICIT_VR_BE,
            transfer_syntax::JPEG_BASELINE,
            transfer_syntax::JPEG_LOSSLESS,
            transfer_syntax::JPEG_LS_LOSSLESS,
            transfer_syntax::JPEG_2000_LOSSLESS,
            transfer_syntax::JPEG_2000,
        ];
        for uid in &uids {
            assert!(!uid.is_empty(), "transfer syntax UID must not be empty");
            assert!(uid.starts_with("1.2.840.10008"),
                "transfer syntax UID {} must start with 1.2.840.10008", uid);
        }
    }
}
