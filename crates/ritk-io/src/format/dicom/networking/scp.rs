//! Embedded C-STORE SCP — DICOM Storage Service Class Provider (PS3.4 §B).
//!
//! # Overview
//!
//! [`StoreScp::start`] binds a TCP listener, spawns an accept thread, and
//! returns a [`StoreScpHandle`].  For every incoming DICOM association the SCP:
//!
//! 1. Parses A-ASSOCIATE-RQ and accepts all offered presentation contexts
//!    using the first offered transfer syntax per context.
//! 2. Sends A-ASSOCIATE-AC.
//! 3. Receives C-STORE-RQ messages; emits each as a [`StoredInstance`] on a
//!    bounded [`mpsc::sync_channel`]; responds with C-STORE-RSP Success.
//! 4. Sends A-RELEASE-RP on A-RELEASE-RQ and closes the connection.
//!
//! # Invariants
//!
//! - **Bounded memory**: the internal channel capacity is `ScpConfig::queue_capacity`.
//!   When the consumer falls behind, additional instances are acknowledged to the
//!   PACS but discarded with a `tracing::warn` event.
//! - **No async contagion**: all I/O uses blocking `std::net::TcpStream` on
//!   dedicated OS threads.  The accept loop polls the shutdown flag every
//!   [`ACCEPT_POLL_INTERVAL`] to support clean termination.
//! - **Send safety**: `StoredInstance` and `ScpConfig` are `Send + Sync`.
//!   `StoreScpHandle` is `Send + !Sync` (contains `mpsc::Receiver`).

use super::association::NetworkingError;
use super::context::transfer_syntax;
use super::dimse::{CommandField, DimseMessage, DimseStatus};
use super::pdu::{
    AbortPdu, AbortSource, APPLICATION_CONTEXT_NAME, AssociateAcPdu, CommandType,
    DEFAULT_MAXIMUM_LENGTH, ImplementationClassUidSubItem, ImplementationVersionNameSubItem,
    MaximumLengthSubItem, MessageControlHeader, PDataTfPdu, PresentationContextItemAc,
    PresentationDataValueItem, Pdu, ReleaseRpPdu, RITK_IMPLEMENTATION_CLASS_UID,
    RITK_IMPLEMENTATION_VERSION, UserInformation,
};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

/// Polling interval for the non-blocking accept loop between connection attempts.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(5);

// ── StoredInstance ─────────────────────────────────────────────────────────────

/// A DICOM instance received by the embedded C-STORE SCP.
///
/// Produced when a PACS delivers an instance via a C-STORE sub-operation,
/// typically triggered by a preceding C-MOVE request.
#[derive(Debug, Clone)]
pub struct StoredInstance {
    /// Abstract SOP Class UID from the negotiated presentation context.
    pub sop_class_uid: String,
    /// SOP Instance UID from C-STORE-RQ tag (0000,1000).
    pub sop_instance_uid: String,
    /// Raw dataset bytes in the negotiated transfer syntax.
    pub dataset_bytes: Vec<u8>,
    /// Transfer syntax UID negotiated for this presentation context.
    pub transfer_syntax_uid: String,
}

/// Pad a UID value to even length with a null byte, as required by PS3.5.
fn pad_uid(uid: &str) -> Vec<u8> {
    let bytes = uid.as_bytes();
    if bytes.len() % 2 == 0 {
        bytes.to_vec()
    } else {
        let mut v = bytes.to_vec();
        v.push(0x00);
        v
    }
}

impl StoredInstance {
    /// Construct valid DICOM Part 10 bytes from this stored instance.
    ///
    /// Prepends the 128-byte zero preamble, DICM magic, and a File Meta
    /// Information group (group 0002) containing the SOP Class UID,
    /// SOP Instance UID, and Transfer Syntax UID. The `dataset_bytes`
    /// follow unchanged.
    ///
    /// The resulting bytes can be parsed by `dicom::object::from_reader`
    /// or any DICOM Part 10 compliant parser.
    pub fn make_part10_bytes(&self) -> Vec<u8> {
        // 128-byte zero preamble + 4-byte DICM magic
        let preamble = [0u8; 128];
        let dicm = *b"DICM";

        // Build File Meta Information (group 0002) in Explicit VR Little Endian.
        // Tags are written in ascending order per PS3.5.
        let mut meta = Vec::new();

        // (0002,0000) File Meta Information Group Length — placeholder, filled below.
        meta.extend_from_slice(&[0x00, 0x00, 0x02, 0x00]); // tag
        meta.extend_from_slice(b"UL"); // VR
        meta.extend_from_slice(&[0x00, 0x00]); // reserved
        meta.extend_from_slice(&4u32.to_le_bytes()); // value: 4 bytes for length itself

        // The group length value will be corrected after all meta elements are written.
        let group_length_offset = 8; // offset of the 4-byte length value within `meta`

        // (0002,0001) File Meta Information Version
        meta.extend_from_slice(&[0x00, 0x02, 0x00, 0x01]); // tag
        meta.extend_from_slice(b"OB"); // VR
        meta.extend_from_slice(&[0x00, 0x00]); // reserved
        meta.extend_from_slice(&2u32.to_le_bytes()); // length
        meta.extend_from_slice(&[0x00, 0x01]); // version: v1

        // (0002,0002) Media Storage SOP Class UID
        meta.extend_from_slice(&[0x00, 0x02, 0x00, 0x02]); // tag
        meta.extend_from_slice(b"UI"); // VR
        let sop_class_padded = pad_uid(&self.sop_class_uid);
        meta.extend_from_slice(&(sop_class_padded.len() as u16).to_le_bytes()); // length
        meta.extend_from_slice(&sop_class_padded);

        // (0002,0003) Media Storage SOP Instance UID
        meta.extend_from_slice(&[0x00, 0x02, 0x00, 0x03]); // tag
        meta.extend_from_slice(b"UI"); // VR
        let sop_instance_padded = pad_uid(&self.sop_instance_uid);
        meta.extend_from_slice(&(sop_instance_padded.len() as u16).to_le_bytes()); // length
        meta.extend_from_slice(&sop_instance_padded);

        // (0002,0010) Transfer Syntax UID
        meta.extend_from_slice(&[0x00, 0x02, 0x00, 0x10]); // tag
        meta.extend_from_slice(b"UI"); // VR
        let ts_padded = pad_uid(&self.transfer_syntax_uid);
        meta.extend_from_slice(&(ts_padded.len() as u16).to_le_bytes()); // length
        meta.extend_from_slice(&ts_padded);

        // (0002,0012) Implementation Class UID
        meta.extend_from_slice(&[0x00, 0x02, 0x00, 0x12]); // tag
        meta.extend_from_slice(b"UI"); // VR
        let impl_uid = b"1.2.826.0.1.3690043.9.7433.1.0\0"; // padded to even length
        meta.extend_from_slice(&(impl_uid.len() as u16).to_le_bytes());
        meta.extend_from_slice(impl_uid);

        // (0002,0013) Implementation Version Name
        meta.extend_from_slice(&[0x00, 0x02, 0x00, 0x13]); // tag
        meta.extend_from_slice(b"SH"); // VR
        let impl_name = b"RITKSCP1";
        meta.extend_from_slice(&(impl_name.len() as u16).to_le_bytes());
        meta.extend_from_slice(impl_name);

        // Correct the File Meta Information Group Length (0002,0000).
        // This is the byte length of the group 0002 elements EXCLUDING
        // the (0002,0000) element itself, per PS3.10 §7.1.
        let group_length = (meta.len() - 12) as u32; // subtract the 12 bytes of (0002,0000)
        meta[group_length_offset..group_length_offset + 4]
            .copy_from_slice(&group_length.to_le_bytes());

        // Assemble full Part 10 file
        let mut result =
            Vec::with_capacity(128 + 4 + meta.len() + self.dataset_bytes.len());
        result.extend_from_slice(&preamble);
        result.extend_from_slice(&dicm);
        result.extend_from_slice(&meta);
        result.extend_from_slice(&self.dataset_bytes);
        result
    }
}

// ── ScpConfig ─────────────────────────────────────────────────────────────────

/// Configuration for the embedded C-STORE SCP.
#[derive(Debug, Clone)]
pub struct ScpConfig {
    /// AE title this application advertises to connecting SCUs.
    ///
    /// The PACS must be configured to forward C-STORE sub-operations to this
    /// AE title at `0.0.0.0:port`.
    pub ae_title: String,
    /// TCP port to listen on.
    ///
    /// Use `0` to request an OS-assigned ephemeral port; read the actual port
    /// from [`StoreScpHandle::port`] after start.
    pub port: u16,
    /// Maximum PDU length advertised in A-ASSOCIATE-AC.
    pub max_pdu_length: u32,
    /// Bounded channel capacity for buffered [`StoredInstance`] values.
    ///
    /// When the consumer (egui frame loop) falls behind by this many instances,
    /// new arrivals are acknowledged to the PACS and then discarded.
    pub queue_capacity: usize,
    /// Per-connection read/write timeout.
    ///
    /// Prevents zombie connections from blocking a connection thread indefinitely.
    pub read_timeout: Duration,
}

impl Default for ScpConfig {
    fn default() -> Self {
        Self {
            ae_title: "RITKSNAP".to_string(),
            port: 11112,
            max_pdu_length: DEFAULT_MAXIMUM_LENGTH,
            queue_capacity: 512,
            read_timeout: Duration::from_secs(60),
        }
    }
}

// ── StoreScpHandle ─────────────────────────────────────────────────────────────

/// Handle to a running embedded C-STORE SCP.
///
/// Poll [`StoreScpHandle::try_recv`] regularly (e.g., once per egui frame).
/// Dropping the handle signals the accept thread to exit on its next poll cycle.
pub struct StoreScpHandle {
    rx: mpsc::Receiver<StoredInstance>,
    shutdown: Arc<AtomicBool>,
    actual_port: u16,
    ae_title: String,
}

impl StoreScpHandle {
    /// Non-blocking poll: returns the next buffered [`StoredInstance`] or `None`.
    pub fn try_recv(&self) -> Option<StoredInstance> {
        self.rx.try_recv().ok()
    }

    /// TCP port the SCP is listening on.
    ///
    /// Reflects the OS-assigned port when [`ScpConfig::port`] was `0`.
    pub fn port(&self) -> u16 {
        self.actual_port
    }

    /// AE title the SCP advertises to connecting SCUs.
    pub fn ae_title(&self) -> &str {
        &self.ae_title
    }

    /// Signal the SCP to stop and consume the handle.
    ///
    /// The accept thread exits on its next [`ACCEPT_POLL_INTERVAL`] poll.
    /// In-progress connection threads run to completion.
    pub fn stop(self) {
        // Drop triggers the Drop impl which sets shutdown = true.
    }
}

impl Drop for StoreScpHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

// ── StoreScp ──────────────────────────────────────────────────────────────────

/// Embedded C-STORE SCP factory.
///
/// # Example
///
/// ```ignore
/// let handle = StoreScp::start(ScpConfig::default()).expect("SCP bind");
/// // Direct C-MOVE to handle.ae_title() @ handle.port().
/// while let Some(inst) = handle.try_recv() {
///     println!("Received {}", inst.sop_instance_uid);
/// }
/// ```
pub struct StoreScp;

impl StoreScp {
    /// Bind a TCP listener and start the SCP accept thread.
    ///
    /// # Errors
    ///
    /// Returns [`NetworkingError::Protocol`] if the TCP bind or
    /// `set_nonblocking` call fails.
    pub fn start(config: ScpConfig) -> Result<StoreScpHandle, NetworkingError> {
        let listener = TcpListener::bind(("0.0.0.0", config.port)).map_err(|e| {
            NetworkingError::Protocol(format!("SCP bind on port {}: {e}", config.port))
        })?;
        listener.set_nonblocking(true).map_err(|e| {
            NetworkingError::Protocol(format!("SCP set_nonblocking: {e}"))
        })?;

        let actual_port = listener.local_addr().map(|a| a.port()).unwrap_or(config.port);
        let (tx, rx) = mpsc::sync_channel::<StoredInstance>(config.queue_capacity);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = Arc::clone(&shutdown);
        let ae_title = config.ae_title.clone();

        std::thread::spawn(move || {
            scp_accept_loop(listener, config, tx, shutdown_thread);
        });

        Ok(StoreScpHandle { rx, shutdown, actual_port, ae_title })
    }
}

// ── Accept loop ───────────────────────────────────────────────────────────────

fn scp_accept_loop(
    listener: TcpListener,
    config: ScpConfig,
    tx: mpsc::SyncSender<StoredInstance>,
    shutdown: Arc<AtomicBool>,
) {
    use std::io::ErrorKind;
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept() {
            Ok((stream, peer)) => {
                tracing::debug!("SCP: accepted connection from {peer}");
                let cfg = config.clone();
                let tx2 = tx.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, &cfg, &tx2) {
                        tracing::warn!("SCP connection error: {e}");
                    }
                });
            }
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL_INTERVAL);
            }
            Err(e) => {
                if !shutdown.load(Ordering::Relaxed) {
                    tracing::warn!("SCP accept error: {e}");
                }
                std::thread::sleep(ACCEPT_POLL_INTERVAL);
            }
        }
    }
    tracing::debug!("SCP accept loop exited");
}

// ── Connection handler ────────────────────────────────────────────────────────

fn handle_connection(
    mut stream: TcpStream,
    config: &ScpConfig,
    tx: &mpsc::SyncSender<StoredInstance>,
) -> Result<(), NetworkingError> {
    stream.set_nonblocking(false)
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
    stream.set_read_timeout(Some(config.read_timeout))
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
    stream.set_write_timeout(Some(config.read_timeout))
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
    let mut ctx_map: HashMap<u8, (String, String)> = HashMap::new();
    let pc_acs: Vec<PresentationContextItemAc> = rq
        .presentation_contexts
        .iter()
        .map(|pc| {
            let ts = pc
                .transfer_syntax_uids
                .first()
                .cloned()
                .unwrap_or_else(|| transfer_syntax::IMPLICIT_VR_LE.to_string());
            ctx_map.insert(
                pc.presentation_context_id,
                (pc.abstract_syntax_uid.clone(), ts.clone()),
            );
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
            application_context_name: APPLICATION_CONTEXT_NAME.to_string(),
            presentation_contexts: pc_acs,
            user_information: UserInformation {
                maximum_length: MaximumLengthSubItem {
                    maximum_length_received: config.max_pdu_length,
                },
                implementation_class_uid: ImplementationClassUidSubItem {
                    implementation_class_uid: RITK_IMPLEMENTATION_CLASS_UID.to_string(),
                },
                implementation_version_name: Some(ImplementationVersionNameSubItem {
                    implementation_version_name: RITK_IMPLEMENTATION_VERSION.to_string(),
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
            ScpMessageResult::Message { cid, msg } => {
                match msg.command_field() {
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
                }
            }
        }
    }

    Ok(())
}

// ── C-STORE-RQ handler ────────────────────────────────────────────────────────

fn handle_store_rq(
    stream: &mut TcpStream,
    cid: u8,
    msg: DimseMessage,
    ctx_map: &HashMap<u8, (String, String)>,
    tx: &mpsc::SyncSender<StoredInstance>,
    config: &ScpConfig,
) -> Result<(), NetworkingError> {
    let sop_class_uid = msg.affected_sop_class_uid().unwrap_or_default();
    let sop_instance_uid = msg.affected_sop_instance_uid().unwrap_or_default();
    let msg_id = msg.message_id().unwrap_or(1);
    let dataset_bytes = msg.data_set.unwrap_or_default();
    let transfer_syntax_uid = ctx_map
        .get(&cid)
        .map(|(_, ts)| ts.clone())
        .unwrap_or_default();

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
        sop_instance_uid: sop_instance_uid.clone(),
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
    let mut cmd_buf: Vec<u8> = Vec::new();
    let mut data_buf: Vec<u8> = Vec::new();
    let mut cid: u8 = 0;
    let mut cmd_complete = false;

    loop {
        match read_pdu_stream(stream)? {
            Pdu::PDataTf(pd) => {
                let data_last = pd.presentation_data_value_items.iter().any(|p| {
                    p.message_control_header.message_type == CommandType::DataSet
                        && p.message_control_header.last_fragment
                });

                for pdv in &pd.presentation_data_value_items {
                    cid = pdv.presentation_context_id;
                    match pdv.message_control_header.message_type {
                        CommandType::Command => {
                            cmd_buf.extend_from_slice(&pdv.data);
                            if pdv.message_control_header.last_fragment {
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
                let has_ds = msg.command_data_set_type().map_or(false, |v| v != 0x0101);

                if !has_ds || data_last {
                    msg.data_set =
                        if data_buf.is_empty() { None } else { Some(std::mem::take(&mut data_buf)) };
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
                        if p.message_control_header.last_fragment {
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

fn read_pdu_stream(stream: &mut TcpStream) -> Result<Pdu, NetworkingError> {
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

fn write_pdu_stream(stream: &mut TcpStream, pdu: &Pdu) -> Result<(), NetworkingError> {
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
fn send_command_pdv(
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
                    last_fragment: true,
                },
                data: cmd_bytes,
            }],
        }),
    )
}

#[cfg(test)]
#[path = "tests_scp.rs"]
mod tests_scp;
