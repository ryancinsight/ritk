use std::collections::TryReserveError;
use std::io;

use thiserror::Error;

/// Failure to decode an MGH or MGZ voxel payload.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MghError {
    /// The header names an unsupported voxel data type.
    #[error("Unsupported MGH data type code: {code}")]
    UnsupportedDataType {
        /// Declared data type code.
        code: i32,
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
}
