use std::collections::TryReserveError;
use std::io;
use std::path::PathBuf;

use ritk_spatial::InvalidSpacing;
use thiserror::Error;

/// Failure to read an MGH or MGZ file.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use ritk_mgh::{read_mgh, MghError};
///
/// let error = read_mgh("missing.mgh", &SequentialBackend)
///     .expect_err("the file does not exist");
/// assert!(matches!(error, MghError::Open { .. }));
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MghError {
    /// The file could not be opened or inspected.
    #[error("cannot open MGH/MGZ file {path:?}: {source}")]
    Open {
        /// Path supplied by the caller.
        path: PathBuf,
        /// Filesystem failure.
        source: io::Error,
    },
    /// A header field could not be read.
    #[error("cannot read MGH header: {source}")]
    Header {
        /// Stream failure, including truncation.
        source: io::Error,
    },
    /// The header version is not the supported MGH version.
    #[error("Invalid MGH version: expected {expected}, found {found}")]
    InvalidVersion {
        /// Supported version.
        expected: i32,
        /// Version declared by the file.
        found: i32,
    },
    /// A header extent is zero or negative.
    #[error("Invalid MGH dimensions: width={width}, height={height}, depth={depth}")]
    InvalidDimensions {
        /// Declared width.
        width: i32,
        /// Declared height.
        height: i32,
        /// Declared depth.
        depth: i32,
    },
    /// The header frame count is zero or negative.
    #[error("Invalid MGH nframes: {nframes}")]
    InvalidFrameCount {
        /// Declared frame count.
        nframes: i32,
    },
    /// The header names an unsupported voxel data type.
    #[error("Unsupported MGH data type code: {code}")]
    UnsupportedDataType {
        /// Declared data type code.
        code: i32,
    },
    /// The RAS block declares a voxel spacing that is not positive and finite.
    #[error("Invalid MGH RAS voxel spacing: {source}")]
    InvalidSpacing {
        /// Rejected spacing.
        source: InvalidSpacing,
    },
    /// The per-frame voxel count does not fit `usize`.
    #[error("Volume dimensions overflow: {nx}x{ny}x{nz}")]
    DimensionOverflow {
        /// Declared extent along X.
        nx: usize,
        /// Declared extent along Y.
        ny: usize,
        /// Declared extent along Z.
        nz: usize,
    },
    /// The series voxel count does not fit `usize`.
    #[error("Series voxel count overflow: {voxels} voxels × {frames} frames")]
    SeriesOverflow {
        /// Voxels in one frame.
        voxels: usize,
        /// Declared frame count.
        frames: usize,
    },
    /// The series payload size in bytes does not fit `usize`.
    #[error("Series data size overflow: {voxels} voxels × {bytes_per_voxel} bytes")]
    DataSizeOverflow {
        /// Voxels in the whole series.
        voxels: usize,
        /// Bytes per voxel of the declared data type.
        bytes_per_voxel: usize,
    },
    /// A single-volume read met a multi-frame file.
    #[error(
        "MGH file declares {nframes} frames; this reader returns a 3-D Image, which represents \
         exactly one frame. Use read_mgh_series for this acquisition."
    )]
    MultiFrame {
        /// Declared frame count.
        nframes: usize,
    },
    /// The voxel payload ends before the header's voxel count.
    #[error("MGH voxel payload is truncated at voxel {voxel} of {voxel_count}")]
    TruncatedPayload {
        /// First voxel that is not fully present.
        voxel: usize,
        /// Voxels per frame declared by the header.
        voxel_count: usize,
    },
    /// The voxel payload could not be read.
    #[error("failed to read MGH voxel {voxel} of {voxel_count}: {source}")]
    PayloadRead {
        /// First voxel that was not read.
        voxel: usize,
        /// Voxels per frame declared by the header.
        voxel_count: usize,
        /// Stream failure.
        source: io::Error,
    },
    /// Decoded output storage could not be reserved.
    #[error("cannot allocate MGH decode storage: {source}")]
    Allocation {
        /// Allocation failure.
        source: TryReserveError,
    },
    /// The decoded voxels could not be assembled into an image.
    #[error("cannot build image from MGH voxels: {source:#}")]
    Image {
        /// Image construction failure.
        source: anyhow::Error,
    },
}
