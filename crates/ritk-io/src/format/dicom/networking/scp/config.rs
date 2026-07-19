//! SCP configuration, stored-instance representation, and the public handle.

use arrayvec::ArrayString;

use super::super::pdu::DEFAULT_MAXIMUM_LENGTH;
use crate::format::dicom::reader::types::literal_arraystring;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// Polling interval for the non-blocking accept loop between connection attempts.
pub(super) const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(5);

// â”€â”€ StoredInstance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A DICOM instance received by the embedded C-STORE SCP.
///
/// Produced when a PACS delivers an instance via a C-STORE sub-operation,
/// typically triggered by a preceding C-MOVE request.
#[derive(Debug, Clone)]
pub struct StoredInstance {
    /// Abstract SOP Class UID from the negotiated presentation context.
    pub sop_class_uid: ArrayString<64>,
    /// SOP Instance UID from C-STORE-RQ tag (0000,1000).
    pub sop_instance_uid: ArrayString<64>,
    /// Raw dataset bytes in the negotiated transfer syntax.
    pub dataset_bytes: Vec<u8>,
    /// Transfer syntax UID negotiated for this presentation context.
    pub transfer_syntax_uid: ArrayString<64>,
}

/// Pad a UID value to even length with a null byte, as required by PS3.5.
pub(crate) fn pad_uid(uid: &str) -> Vec<u8> {
    let bytes = uid.as_bytes();
    if bytes.len().is_multiple_of(2) {
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
        let mut meta = Vec::with_capacity(256);

        // (0002,0000) File Meta Information Group Length â€” placeholder, filled below.
        // The group length value will be corrected after all meta elements are written.
        meta.extend_from_slice(&[0x00, 0x00, 0x02, 0x00]); // tag
        meta.extend_from_slice(b"UL"); // VR
        meta.extend_from_slice(&[0x00, 0x00]); // reserved
        meta.extend_from_slice(&4u32.to_le_bytes()); // value: 4 bytes for length itself
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
        // the (0002,0000) element itself, per PS3.10 Â§7.1.
        let group_length = (meta.len() - 12) as u32; // subtract the 12 bytes of (0002,0000)
        meta[group_length_offset..group_length_offset + 4]
            .copy_from_slice(&group_length.to_le_bytes());

        // Assemble full Part 10 file
        let mut result = Vec::with_capacity(128 + 4 + meta.len() + self.dataset_bytes.len());
        result.extend_from_slice(&preamble);
        result.extend_from_slice(&dicm);
        result.extend_from_slice(&meta);
        result.extend_from_slice(&self.dataset_bytes);
        result
    }
}

// â”€â”€ ScpConfig â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Configuration for the embedded C-STORE SCP.
#[derive(Debug, Clone)]
pub struct ScpConfig {
    /// AE title this application advertises to connecting SCUs.
    ///
    /// The PACS must be configured to forward C-STORE sub-operations to this
    /// AE title at `0.0.0.0:port`.
    pub ae_title: ArrayString<16>,

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
            ae_title: literal_arraystring("RITKSNAP"),
            port: 11112,
            max_pdu_length: DEFAULT_MAXIMUM_LENGTH,
            queue_capacity: 512,
            read_timeout: Duration::from_secs(60),
        }
    }
}

// â”€â”€ StoreScpHandle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Handle to a running embedded C-STORE SCP.
///
/// Poll [`StoreScpHandle::try_recv`] regularly (e.g., once per egui frame).
/// Dropping the handle signals the accept thread to exit on its next poll cycle.
pub struct StoreScpHandle {
    pub(super) rx: mpsc::Receiver<StoredInstance>,
    pub(super) shutdown: std::sync::Arc<AtomicBool>,
    pub(super) actual_port: u16,
    pub(super) ae_title: ArrayString<16>,
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
    /// The accept thread exits on its next `ACCEPT_POLL_INTERVAL` poll.
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
