//! DICOM association context types — presentation context negotiation,
//! transfer syntax UID constants, and association configuration.
//!
//! Extracted from `association.rs` to satisfy the 500-line structural limit.

use super::pdu::{UserIdentity, DEFAULT_MAXIMUM_LENGTH};
use super::types::{AeTitle, DicomAddress};
use arrayvec::ArrayString;
use std::time::Duration;

// ── Transfer syntax UIDs (PS 3.6 Table A-1) ──────────────────────────────────

/// DICOM standard transfer syntax UIDs (PS 3.6 Table A-1).
pub mod transfer_syntax {
    /// Implicit VR Little Endian — default transfer syntax.
    pub const IMPLICIT_VR_LE: &str = "1.2.840.10008.1.2";
    /// Explicit VR Little Endian.
    pub const EXPLICIT_VR_LE: &str = "1.2.840.10008.1.2.1";
    /// Explicit VR Big Endian (retired; included for compatibility).
    pub const EXPLICIT_VR_BE: &str = "1.2.840.10008.1.2.2";
    /// JPEG Baseline (Process 1) — lossy 8-bit.
    pub const JPEG_BASELINE: &str = "1.2.840.10008.1.2.4.50";
    /// JPEG Lossless (Process 14, SV1).
    pub const JPEG_LOSSLESS: &str = "1.2.840.10008.1.2.4.70";
    /// JPEG-LS Lossless.
    pub const JPEG_LS_LOSSLESS: &str = "1.2.840.10008.1.2.4.80";
    /// JPEG 2000 Lossless.
    pub const JPEG_2000_LOSSLESS: &str = "1.2.840.10008.1.2.4.90";
    /// JPEG 2000 (lossy or lossless).
    pub const JPEG_2000: &str = "1.2.840.10008.1.2.4.91";
}

// ── AssociationConfig ─────────────────────────────────────────────────────────

/// Association-level DICOM connection configuration.
///
/// Built from the viewer's UI configuration and passed into
/// [`super::association::Association::connect`] or the high-level SCU
/// convenience functions (`echo`, `find`, `retrieve`, `store`).
#[derive(Debug, Clone)]
pub struct AssociationConfig {
    pub called_ae_title: ArrayString<16>,
    pub calling_ae_title: ArrayString<16>,
    pub host: String,
    pub port: u16,
    pub max_pdu_length: u32,
    pub timeout: Duration,
    pub presentation_contexts: Vec<RequestedPresentationContext>,
    pub user_identity: Option<UserIdentity>,
}

impl Default for AssociationConfig {
    fn default() -> Self {
        Self {
            called_ae_title: ArrayString::from("ANYSCP").unwrap(),
            calling_ae_title: ArrayString::from("RITK").unwrap(),
            host: "127.0.0.1".into(),
            port: 104,
            max_pdu_length: DEFAULT_MAXIMUM_LENGTH,
            timeout: Duration::from_secs(30),
            presentation_contexts: Vec::new(),
            user_identity: None,
        }
    }
}

impl AssociationConfig {
    /// Construct from a validated calling AE title and remote DICOM address.
    pub fn new(calling: AeTitle, remote: DicomAddress) -> Self {
        Self {
            called_ae_title: remote
                .ae_title
                .as_str()
                .try_into()
                .unwrap_or_else(|_| ArrayString::from("ANYSCP").unwrap()),
            calling_ae_title: calling
                .as_str()
                .try_into()
                .unwrap_or_else(|_| ArrayString::from("RITK").unwrap()),
            host: remote.host.clone(),
            port: remote.port,
            ..Default::default()
        }
    }

    /// Override the TCP connect and read timeout.
    pub fn with_connect_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    /// Override the TCP read timeout.
    pub fn with_read_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
}

// ── Presentation context types ────────────────────────────────────────────────

/// A requested presentation context sent in A-ASSOCIATE-RQ (PS 3.8 §9.3.2).
#[derive(Debug, Clone)]
pub struct RequestedPresentationContext {
    pub abstract_syntax_uid: ArrayString<64>,
    pub transfer_syntax_uids: Vec<ArrayString<64>>,
}

/// A negotiated (accepted) presentation context from A-ASSOCIATE-AC (PS 3.8 §9.3.3).
#[derive(Debug, Clone, PartialEq)]
pub struct NegotiatedContext {
    pub presentation_context_id: u8,
    pub abstract_syntax_uid: ArrayString<64>,
    pub transfer_syntax_uid: ArrayString<64>,
}
