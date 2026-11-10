use std::io;
use std::num::ParseIntError;
use std::path::PathBuf;

use ritk_codecs::ImageReadBudgetError;
use thiserror::Error;

/// Failure to read a `.mif` file.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use ritk_codecs::ImageReadBudget;
/// use ritk_mif::{read_mif, MifError};
///
/// let error = read_mif("missing.mif", &SequentialBackend, ImageReadBudget::DEFAULT)
///     .expect_err("the file does not exist");
/// assert!(matches!(error, MifError::Open { .. }));
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MifError {
    /// The header file could not be opened.
    #[error("Cannot open .mif file {path:?}: {source}")]
    Open {
        /// Path supplied by the caller.
        path: PathBuf,
        /// Filesystem failure.
        source: io::Error,
    },
    /// The text header is malformed or truncated.
    #[error("{source:#}")]
    Header {
        /// Header parse failure.
        source: anyhow::Error,
    },
    /// A required header key is absent.
    #[error("Missing '{field}' in .mif header")]
    MissingField {
        /// Name of the absent key.
        field: &'static str,
    },
    /// A header value has the wrong syntax.
    #[error("Invalid '{field}' in .mif header: {source:#}")]
    InvalidField {
        /// Name of the malformed key.
        field: &'static str,
        /// Parse failure.
        source: anyhow::Error,
    },
    /// `dim` declares an axis with no voxels.
    #[error(".mif 'dim' declares a {axis} extent of 0; every axis must span at least one voxel")]
    ZeroExtent {
        /// Name of the empty axis.
        axis: &'static str,
    },
    /// `vox` lists fewer than three spatial sizes.
    #[error(".mif 'vox' expected at least 3 spatial sizes, got {found}")]
    TooFewVoxelSizes {
        /// Number of sizes present.
        found: usize,
    },
    /// The per-volume voxel count does not fit `usize`.
    #[error(".mif dim [{nx},{ny},{nz}] voxel count overflows usize")]
    VoxelCountOverflow {
        /// Declared extent along X.
        nx: usize,
        /// Declared extent along Y.
        ny: usize,
        /// Declared extent along Z.
        nz: usize,
    },
    /// The series element count does not fit `usize`.
    #[error(".mif series element count overflows usize")]
    SeriesCountOverflow,
    /// The payload byte count does not fit `usize`.
    #[error(".mif byte count overflows usize: {count} elements of {elem_size} bytes")]
    ByteCountOverflow {
        /// Declared element count.
        count: usize,
        /// Bytes per element.
        elem_size: usize,
    },
    /// A header-declared size exceeds the read budget.
    #[error(transparent)]
    Budget {
        /// Resource ceiling rejected before the payload was read.
        #[from]
        source: ImageReadBudgetError,
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
    /// A single-volume read met a multi-frame file.
    #[error(
        ".mif file has {frames} frames; this reader returns one 3-D volume. \
         Use the series reader for multi-frame files."
    )]
    MultiFrame {
        /// Frames in the file.
        frames: usize,
    },
    /// The decoded voxels could not be assembled into an image.
    #[error("cannot build image from .mif voxels: {source:#}")]
    Image {
        /// Image construction failure.
        source: anyhow::Error,
    },
}
