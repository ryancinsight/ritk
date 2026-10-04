//! Failures of sample decoding.

use super::SampleType;

/// A byte buffer that cannot be read as packed samples.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SampleError {
    /// The buffer length is not a whole number of samples.
    #[error(
        "{byte_len} bytes is not a whole number of {sample_type} samples of {} bytes",
        sample_type.byte_width()
    )]
    PartialSample {
        /// The type the bytes were read as.
        sample_type: SampleType,
        /// The offending buffer length.
        byte_len: usize,
    },
    /// The packed output length cannot be represented by `usize`.
    #[error("{sample_count} {sample_type} samples do not fit in a packed byte buffer")]
    LengthOverflow {
        /// The sample type being encoded.
        sample_type: SampleType,
        /// Number of samples requested.
        sample_count: usize,
    },
    /// Reserving memory for decoded samples failed.
    #[error("cannot reserve {requested_bytes} bytes for {sample_count} {sample_type} samples")]
    Allocation {
        /// The type being decoded or encoded.
        sample_type: SampleType,
        /// Number of samples requested.
        sample_count: usize,
        /// Number of output bytes reserved.
        requested_bytes: usize,
        /// The allocator's capacity or allocation error.
        #[source]
        source: std::collections::TryReserveError,
    },
}
