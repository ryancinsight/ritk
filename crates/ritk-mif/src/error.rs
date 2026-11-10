use std::io;
use std::num::ParseIntError;
use std::path::PathBuf;

use thiserror::Error;

/// Failure to read a `.mif` data payload.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MifError {
    /// The payload byte count does not fit `usize`.
    #[error(".mif byte count overflows usize: {count} elements of {elem_size} bytes")]
    ByteCountOverflow {
        /// Declared element count.
        count: usize,
        /// Bytes per element.
        elem_size: usize,
    },
    /// The `file` key does not name a data file and an offset.
    #[error("Invalid 'file' key in .mif header: '{spec}'")]
    InvalidFileKey {
        /// Value of the key.
        spec: String,
    },
    /// The `file` offset is not an integer.
    #[error("Invalid offset in .mif 'file' key: {source}")]
    InvalidFileOffset {
        /// Integer parse failure.
        source: ParseIntError,
    },
    /// The `file` offset lies beyond the end of the data.
    #[error(
        ".mif 'file' key declares offset {offset}, but only {available} bytes follow the header"
    )]
    OffsetOverrun {
        /// Declared offset.
        offset: u64,
        /// Bytes available before the end of the data.
        available: u64,
    },
    /// The payload could not be read.
    #[error("Cannot read .mif data {path:?}: {source}")]
    PayloadRead {
        /// File holding the payload.
        path: PathBuf,
        /// Stream failure.
        source: io::Error,
    },
    /// The payload is shorter than `dim` and `datatype` require.
    #[error(".mif payload has {actual} bytes but dim requires {expected}")]
    PayloadTooShort {
        /// Bytes present.
        actual: usize,
        /// Bytes required.
        expected: usize,
    },
    /// The payload length differs from the element count.
    #[error(
        ".mif payload size mismatch: expected {expected_count} elements ({expected_bytes} bytes), \
         got {actual} bytes"
    )]
    PayloadSizeMismatch {
        /// Elements the header implies.
        expected_count: usize,
        /// Bytes the header implies.
        expected_bytes: usize,
        /// Bytes present.
        actual: usize,
    },
    /// The element size is not 1, 2, 4, or 8 bytes.
    #[error("Unsupported .mif element size {elem_size} (expected 1, 2, 4, or 8)")]
    UnsupportedElementSize {
        /// Bytes per element.
        elem_size: usize,
    },
}
