#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// Error returned when reading or writing a TRX file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrxError {
    /// A header count is too large to size the arrays it describes.
    ///
    /// `header.json` supplies these as unbounded JSON numbers, so the
    /// arithmetic that derives element counts from them runs before the
    /// length checks that would otherwise catch a nonsensical value.
    #[error("TRX header field '{field}' is {value}, too large to size the data it describes")]
    HeaderCountOverflow {
        /// Name of the offending header field.
        field: &'static str,
        /// Value it declared.
        value: u64,
    },

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// The positions array length does not match `nb_points * 3`.
    #[error("positions array length mismatch: expected {expected} elements, got {got}")]
    PositionsLengthMismatch {
        /// Expected number of scalar elements.
        expected: u64,
        /// Actual number of scalar elements read.
        got: u64,
    },

    /// The offsets array length does not match `nb_streamlines + 1`.
    #[error("offsets array length mismatch: expected {expected} elements, got {got}")]
    OffsetsLengthMismatch {
        /// Expected number of offset entries.
        expected: u64,
        /// Actual number read.
        got: u64,
    },

    /// The sentinel offset does not equal `nb_points`.
    #[error("sentinel offset mismatch: expected {expected}, got {got}")]
    SentinelMismatch {
        /// Expected sentinel value (= nb_points).
        expected: u64,
        /// Actual last entry in offsets array.
        got: u64,
    },

    /// An offset entry is out of bounds or non-monotonic.
    #[error("invalid offset at index {index}: {value} (previous: {prev}, max: {max})")]
    InvalidOffset {
        /// Offset index.
        index: usize,
        /// The invalid value.
        value: u64,
        /// Previous offset value.
        prev: u64,
        /// Maximum allowed value (= nb_points).
        max: u64,
    },

    /// Unsupported data type string.
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),

    /// Gaia rejected the point sequence.
    #[error("invalid streamline {index}: {source}")]
    InvalidPolyline {
        /// Streamline index.
        index: usize,
        /// Error from Gaia.
        #[source]
        source: gaia::PolylineError,
    },

    /// Non-finite coordinate.
    #[error("non-finite coordinate in streamline {index}, point {point_index}")]
    NonFiniteCoordinate {
        /// Streamline index.
        index: usize,
        /// Point index.
        point_index: usize,
    },
}
