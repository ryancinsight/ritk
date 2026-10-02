//! Failures of sample decoding.

use super::SampleType;

/// A byte buffer that cannot be read as packed samples.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
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
}
