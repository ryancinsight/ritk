//! PACS server configuration for the snap viewer UI.
//!
//! Stores raw string fields edited directly in the egui config form.
//! Conversion to [`AssociationConfig`] is performed at request submission time,
//! keeping the UI layer free of DIMSE protocol details.

use arrayvec::ArrayString;
use ritk_io::AssociationConfig;

// ── PacsConfig ────────────────────────────────────────────────────────────────

/// PACS server configuration (UI-facing).
///
/// All fields are raw strings / primitive values for direct in-place editing
/// inside the `pacs_panel` egui form.  They are converted to [`AssociationConfig`]
/// by [`PacsConfig::to_association_config`] only when a request is submitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacsConfig {
    /// AE title of this application (calling AE title, PS 3.8 §7.1.1).
    pub calling_ae_title: String,
    /// AE title of the remote PACS (called AE title).
    pub called_ae_title: String,
    /// Remote PACS hostname or IP address.
    pub host: String,
    /// Remote PACS TCP port (standard: 104, common test: 11112 / 4242).
    pub port: u16,
    /// Move destination AE title for C-MOVE requests.
    ///
    /// The PACS will forward retrieved objects to this AE via C-STORE
    /// sub-operations.  Configure to the AE title of the receiving SCP.
    pub move_destination: String,
    /// AE title the embedded C-STORE SCP advertises.
    pub scp_ae_title: String,
    /// TCP port the embedded C-STORE SCP listens on. Use 0 for OS-assigned.
    pub scp_port: u16,
    /// TCP connection + read timeout in seconds.
    pub timeout_secs: u64,
    /// Whether received DICOM objects are automatically loaded into the viewer.
    pub auto_load_received: bool,

    /// Maximum number of pending instances that will be auto-loaded.
    ///
    /// When `auto_load_received` is enabled and the pending instance count
    /// exceeds this limit, auto-load is suppressed and the user must
    /// manually click "Load Received". Default: 512 (covers a typical
    /// CT series of ~300-500 slices).
    pub auto_load_limit: u32,
}

impl Default for PacsConfig {
    fn default() -> Self {
        Self {
            calling_ae_title: "RITKSNAP".to_owned(),
            called_ae_title: "ORTHANC".to_owned(),
            host: "localhost".to_owned(),
            port: 4242,
            move_destination: "RITKSNAP".to_owned(),
            scp_ae_title: "RITKSNAP".to_owned(),
            scp_port: 11112,
            timeout_secs: 30,
            auto_load_received: true,
            auto_load_limit: 512,
        }
    }
}

impl PacsConfig {
    /// Convert to [`AssociationConfig`] for use with the DIMSE SCU functions.
    ///
    /// AE titles are forwarded verbatim; the DICOM association layer performs
    /// PS 3.8 validation on establishment.  `presentation_contexts` is left
    /// empty so each SCU function can set the appropriate SOP class.
    pub fn to_association_config(&self) -> AssociationConfig {
        AssociationConfig {
            calling_ae_title: self.calling_ae_title.as_str().try_into().unwrap_or_else(|_| ArrayString::from("RITKSNAP").unwrap()),
            called_ae_title: self.called_ae_title.as_str().try_into().unwrap_or_else(|_| ArrayString::from("ORTHANC").unwrap()),
            host: self.host.clone(),
            port: self.port,
            timeout: std::time::Duration::from_secs(self.timeout_secs),
            ..Default::default()
        }
    }
}
