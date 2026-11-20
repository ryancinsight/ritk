//! SCP configuration, stored-instance representation, and the public handle.

use anyhow::Result;
use arrayvec::ArrayString;
use ritk_dicom::dataset::FileMetaBuilder;

use super::super::pdu::{DEFAULT_MAXIMUM_LENGTH, RITK_IMPLEMENTATION_VERSION};
use crate::format::dicom::reader::types::literal_arraystring;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// Polling interval for the non-blocking accept loop between connection attempts.
pub(super) const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(5);

// ── StoredInstance ─────────────────────────────────────────────────────────────

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

impl StoredInstance {
    /// Wrap the received dataset in a Part 10 file using the negotiated syntax.
    ///
    /// # Errors
    /// Returns an error if the association metadata cannot form a Part 10 header.
    pub fn make_part10_bytes(&self) -> Result<Vec<u8>> {
        let mut meta = FileMetaBuilder::new()
            .transfer_syntax(self.transfer_syntax_uid.as_str())
            .media_storage_sop_class_uid(self.sop_class_uid.as_str())
            .media_storage_sop_instance_uid(self.sop_instance_uid.as_str())
            .build()?;
        meta.implementation_version_name = Some(RITK_IMPLEMENTATION_VERSION.to_owned());

        let mut result = Vec::new();
        meta.write_header(&mut result)?;
        result.extend_from_slice(&self.dataset_bytes);
        Ok(result)
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

// ── StoreScpHandle ─────────────────────────────────────────────────────────────

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

    /// Blocking receive: waits up to `timeout` for the next buffered
    /// [`StoredInstance`] and returns `None` if none arrives in time.
    ///
    /// Use this when a caller has nothing else to do; a UI thread should keep
    /// using [`StoreScpHandle::try_recv`] so it never blocks a frame.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<StoredInstance> {
        self.rx.recv_timeout(timeout).ok()
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
        // Relaxed is the weakest ordering that supplies the needed edge: the
        // flag is a pure cancellation signal and publishes no other data, and
        // the accept loop polls it with `Relaxed` loads. It only has to become
        // visible on the next `ACCEPT_POLL_INTERVAL`, which Relaxed guarantees.
        self.shutdown.store(true, Ordering::Relaxed);
    }
}
