//! Persisted source references are revalidated before a session replaces a study.
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A path from an earlier session or an exact DICOM acquisition reference.
///
/// The path variant preserves legacy JSON string sources at the deserialize
/// boundary. DICOM references contain no trusted image metadata or pixels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StudySource {
    /// Filesystem source used by legacy sessions and non-DICOM volumes.
    Path(PathBuf),
    /// Exact acquisition membership; every file and UID is checked on restore.
    Dicom {
        /// Expected Series Instance UID.
        series_uid: String,
        /// Exact selected instance paths, without directory expansion.
        files: Vec<PathBuf>,
    },
}

/// Current session format, accepting version 1 at the deserialize boundary.
///
/// Legacy unversioned sessions default to version 1. Both accepted inputs
/// normalize to version 2, whose source can preserve exact DICOM membership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct SessionFormat;

impl TryFrom<u8> for SessionFormat {
    type Error = &'static str;
    fn try_from(version: u8) -> Result<Self, Self::Error> {
        match version {
            1 | 2 => Ok(Self),
            _ => Err("unsupported viewer session format version"),
        }
    }
}

impl From<SessionFormat> for u8 {
    fn from(_: SessionFormat) -> Self {
        2
    }
}
