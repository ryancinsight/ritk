//! Legacy compatibility types for DICOM networking (GAP-262-IO-01).
//!
//! These types are retained for sibling module (`echo`, `find`, `store`, `move_`)
//! compatibility with the `dicom-ul`-based implementation.

use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

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

// ── AE Title ──────────────────────────────────────────────────────────────────

/// Validated DICOM AE Title (1–16 printable ASCII characters, no backslash).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AeTitle(String);

impl AeTitle {
    /// Construct an AE Title, validating length and character set per PS 3.8.
    pub fn new(s: &str) -> Result<Self, NetworkingError> {
        if s.is_empty() || s.len() > 16 {
            return Err(NetworkingError::InvalidAeTitle(
                s.to_owned(),
                "length must be 1-16 characters",
            ));
        }
        if s.bytes().any(|b| !(0x20..0x7F).contains(&b) || b == b'\\') {
            return Err(NetworkingError::InvalidAeTitle(
                s.to_owned(),
                "must be printable ASCII excluding backslash",
            ));
        }
        Ok(Self(s.to_owned()))
    }

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

/// Remote DICOM endpoint (host, port, AE title).
#[derive(Debug, Clone)]
pub struct DicomAddress {
    pub host: String,
    pub port: u16,
    pub ae_title: AeTitle,
}

impl DicomAddress {
    pub fn new(host: impl Into<String>, port: u16, ae_title: AeTitle) -> Self {
        Self {
            host: host.into(),
            port,
            ae_title,
        }
    }

    pub fn socket_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

// ── Response types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EchoResponse {
    pub status: u16,
}

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
