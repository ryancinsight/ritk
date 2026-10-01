//! Failures of sample decoding and rescaling.

use super::SampleType;

/// A sample buffer or rescale that cannot be applied as asked.
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
    /// The sample vector could not reserve storage for the decoded payload.
    #[error("cannot allocate {sample_count} {sample_type} samples: {source}")]
    Allocation {
        /// The type the bytes were read as.
        sample_type: SampleType,
        /// Number of samples requested by the byte length.
        sample_count: usize,
        /// The allocator or capacity failure.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// A rescale coefficient is NaN or infinite.
    #[error("rescale slope {slope} and intercept {intercept} must both be finite")]
    NonFiniteRescale {
        /// The multiplier.
        slope: f64,
        /// The offset.
        intercept: f64,
    },
    /// A non-identity rescale was requested into an integer sample type.
    #[error(
        "rescale y = {slope} * x + {intercept} has no faithful {sample_type} result; read the stored samples and the rescale separately"
    )]
    IntegerRescale {
        /// The integer type the samples were requested in.
        sample_type: SampleType,
        /// The multiplier.
        slope: f64,
        /// The offset.
        intercept: f64,
    },
    /// A finite rescale coefficient overflows the floating-point type the
    /// samples were requested in, or a nonzero slope underflows to zero in it.
    #[error(
        "rescale slope {slope:?} and intercept {intercept:?} fall outside the range of {sample_type}"
    )]
    RescaleOutOfRange {
        /// The floating-point type the samples were requested in.
        sample_type: SampleType,
        /// The multiplier.
        slope: f64,
        /// The offset.
        intercept: f64,
    },
}
