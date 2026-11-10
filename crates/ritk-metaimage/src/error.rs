use std::io;
use std::num::ParseIntError;
use std::path::PathBuf;

use ritk_codecs::ImageReadBudgetError;
use thiserror::Error;

/// Failure to read a MetaImage file.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use ritk_codecs::ImageReadBudget;
/// use ritk_metaimage::{read_metaimage, MetaImageError};
///
/// let error = read_metaimage("missing.mha", &SequentialBackend, ImageReadBudget::DEFAULT)
///     .expect_err("the file does not exist");
/// assert!(matches!(error, MetaImageError::Open { .. }));
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetaImageError {
    /// The header file could not be opened.
    #[error("cannot open MetaImage file {path:?}: {source}")]
    Open {
        /// Header path supplied by the caller.
        path: PathBuf,
        /// Filesystem failure.
        source: io::Error,
    },
    /// A header line could not be read.
    #[error("cannot read MetaImage header line: {source}")]
    HeaderRead {
        /// Stream failure.
        source: io::Error,
    },
    /// The header ended before the `ElementDataFile` key.
    #[error("ElementDataFile key not found in MetaImage header of {path:?}")]
    MissingElementDataFile {
        /// Header path supplied by the caller.
        path: PathBuf,
    },
    /// A required header key is absent.
    #[error("missing '{field}' in MetaImage header")]
    MissingField {
        /// Name of the absent key.
        field: &'static str,
    },
    /// `NDims` is not an integer.
    #[error("'NDims' is not a valid integer: {source}")]
    InvalidNDims {
        /// Integer parse failure.
        source: ParseIntError,
    },
    /// `NDims` is neither 2 nor 3.
    #[error("expected NDims = 2 or 3, found {found}")]
    UnsupportedNDims {
        /// Declared dimension count.
        found: usize,
    },
    /// A numeric header value has the wrong syntax or element count.
    #[error("invalid MetaImage header field '{field}': {source:#}")]
    InvalidField {
        /// Name of the malformed key.
        field: &'static str,
        /// Parse failure.
        source: anyhow::Error,
    },
    /// `ElementType` names an unsupported sample type.
    #[error("unsupported MetaImage ElementType: '{element_type}'")]
    UnsupportedElementType {
        /// Declared element type.
        element_type: String,
    },
    /// The `DimSize` product does not fit `usize`.
    #[error("MetaImage voxel count overflow: DimSize = {nx} {ny} {nz}")]
    VoxelCountOverflow {
        /// Declared extent along X.
        nx: usize,
        /// Declared extent along Y.
        ny: usize,
        /// Declared extent along Z.
        nz: usize,
    },
    /// The payload byte count does not fit `usize`.
    #[error("MetaImage byte count overflow: {voxels} voxels × {elem_size} bytes")]
    ByteCountOverflow {
        /// Declared voxel count.
        voxels: usize,
        /// Bytes per voxel.
        elem_size: usize,
    },
    /// A header-declared or observed size exceeds the read budget.
    #[error(transparent)]
    Budget {
        /// Resource ceiling rejected before the payload was read or inflated.
        #[from]
        source: ImageReadBudgetError,
    },
    /// The payload could not be read.
    #[error("cannot read MetaImage payload {path:?}: {source}")]
    PayloadRead {
        /// File holding the payload.
        path: PathBuf,
        /// Stream failure.
        source: io::Error,
    },
    /// The zlib payload could not be inflated.
    #[error("failed to inflate zlib-compressed MetaImage payload: {source}")]
    Inflate {
        /// Decompression failure.
        source: io::Error,
    },
    /// The payload length differs from the length `DimSize` and `ElementType` imply.
    #[error(
        "MetaImage payload length mismatch: expected {expected} bytes from DimSize \
         ({voxels} voxels × {elem_size} bytes for {element_type}), got {actual} bytes"
    )]
    PayloadLengthMismatch {
        /// Bytes implied by the header.
        expected: usize,
        /// Bytes present after any inflation.
        actual: usize,
        /// Declared voxel count.
        voxels: usize,
        /// Bytes per voxel.
        elem_size: usize,
        /// Declared element type.
        element_type: String,
    },
    /// The payload bytes could not be converted to `f32` samples.
    #[error("cannot decode MetaImage samples: {source:#}")]
    Decode {
        /// Sample conversion failure.
        source: anyhow::Error,
    },
    /// The decoded samples could not be assembled into an image.
    #[error("cannot build image from MetaImage samples: {source:#}")]
    Image {
        /// Image construction failure.
        source: anyhow::Error,
    },
}
