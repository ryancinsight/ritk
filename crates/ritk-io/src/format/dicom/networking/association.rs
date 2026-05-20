//! DICOM network addressing and association configuration types.
//!
//! # Invariants (DICOM PS3.7 §7.1)
//! - AE title: 1–16 printable ASCII chars; no control chars; no backslash.
//! - Port: 0–65535 (0 = OS-assigned, typically for tests).

use std::time::Duration;
use thiserror::Error;

// ── Error ─────────────────────────────────────────────────────────────────────

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

// ── AeTitle ───────────────────────────────────────────────────────────────────

/// Validated DICOM Application Entity Title (PS3.7 §7.1.3).
///
/// 1–16 printable ASCII characters, no backslash, no control characters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AeTitle(String);

impl AeTitle {
    /// Validate and construct an `AeTitle`.
    pub fn new(s: &str) -> Result<Self, NetworkingError> {
        if s.is_empty() || s.len() > 16 {
            return Err(NetworkingError::InvalidAeTitle(
                s.to_owned(),
                "length must be 1–16 characters",
            ));
        }
        if s.bytes().any(|b| b < 0x20 || b >= 0x7F || b == b'\\') {
            return Err(NetworkingError::InvalidAeTitle(
                s.to_owned(),
                "must be printable ASCII excluding backslash",
            ));
        }
        Ok(Self(s.to_owned()))
    }

    /// Returns the raw AE title string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AeTitle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<&str> for AeTitle {
    type Error = NetworkingError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

// ── DicomAddress ──────────────────────────────────────────────────────────────

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

    pub fn socket_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

// ── AssociationConfig ─────────────────────────────────────────────────────────

/// Full configuration for a DIMSE SCU association.
#[derive(Debug, Clone)]
pub struct AssociationConfig {
    pub calling_ae_title: AeTitle,
    pub remote: DicomAddress,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
}

impl AssociationConfig {
    pub fn new(calling: AeTitle, remote: DicomAddress) -> Self {
        Self {
            calling_ae_title: calling,
            remote,
            connect_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(60),
        }
    }

    pub fn with_connect_timeout(mut self, t: Duration) -> Self {
        self.connect_timeout = t;
        self
    }

    pub fn with_read_timeout(mut self, t: Duration) -> Self {
        self.read_timeout = t;
        self
    }
}

// ── Response types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EchoResponse {
    pub status: u16,
}

#[derive(Debug, Clone)]
pub struct FindResult {
    pub elements: Vec<((u16, u16), Vec<u8>)>,
}

impl FindResult {
    pub fn get_string(&self, group: u16, element: u16) -> Option<String> {
        self.elements
            .iter()
            .find(|((g, e), _)| *g == group && *e == element)
            .map(|(_, v)| {
                String::from_utf8_lossy(v)
                    .trim_end_matches(['\0', ' '])
                    .to_owned()
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
