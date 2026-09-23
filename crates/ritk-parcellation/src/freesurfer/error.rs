//! The error every FreeSurfer reader and writer in this module returns.

use std::fmt;

/// The FreeSurfer file format an error arose in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FreeSurferFormat {
    /// Binary triangle surface (`lh.white`, `lh.pial`, `lh.inflated`).
    Surface,
    /// New-format per-vertex scalar file (`lh.curv`, `lh.thickness`, `lh.sulc`).
    Morphometry,
    /// Binary surface annotation (`lh.aparc.annot`).
    Annotation,
    /// ASCII surface label (`lh.cortex.label`).
    Label,
    /// Text colour lookup table (`FreeSurferColorLUT.txt`).
    ColorLut,
}

impl fmt::Display for FreeSurferFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Surface => "triangle surface",
            Self::Morphometry => "morphometry (curv) file",
            Self::Annotation => "annotation",
            Self::Label => "label file",
            Self::ColorLut => "colour lookup table",
        })
    }
}

/// Error returned when reading or writing a FreeSurfer surface-family file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FreeSurferError {
    /// The underlying reader or writer failed; a file shorter than its header
    /// promises surfaces here as [`std::io::ErrorKind::UnexpectedEof`].
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The leading magic number is not the one the format requires.
    #[error("{format}: magic {got:#08x} is not {expected:#08x}")]
    InvalidMagic {
        /// Format being read.
        format: FreeSurferFormat,
        /// The magic the format requires.
        expected: u32,
        /// The magic that was read.
        got: u32,
    },

    /// A count field lies outside the range any real file can carry.
    #[error("{format}: {field} {count} outside 0..={max}")]
    InvalidCount {
        /// Format being read.
        format: FreeSurferFormat,
        /// The field holding the count.
        field: &'static str,
        /// The count that was read.
        count: i64,
        /// The largest count accepted.
        max: i64,
    },

    /// A format version or layout variant this reader does not implement.
    #[error("{format}: unsupported {field} {got}")]
    Unsupported {
        /// Format being read.
        format: FreeSurferFormat,
        /// The field naming the variant.
        field: &'static str,
        /// The value that was read.
        got: i64,
    },

    /// A record is structurally invalid.
    #[error("{format}: {field} {index}: {reason}")]
    Malformed {
        /// Format being read or written.
        format: FreeSurferFormat,
        /// The kind of record.
        field: &'static str,
        /// Position of the record — element index, or line number for text.
        index: usize,
        /// What is wrong with it.
        reason: String,
    },
}

impl FreeSurferError {
    /// A [`FreeSurferError::Malformed`] with the reason rendered from `reason`.
    pub(super) fn malformed(
        format: FreeSurferFormat,
        field: &'static str,
        index: usize,
        reason: impl Into<String>,
    ) -> Self {
        Self::Malformed {
            format,
            field,
            index,
            reason: reason.into(),
        }
    }
}
