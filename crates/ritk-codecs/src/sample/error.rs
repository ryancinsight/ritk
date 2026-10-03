use std::collections::TryReserveError;
use std::error::Error;
use std::fmt;

use super::buffer::{SampleBuffer, SampleType};

/// Failure to decode or encode a complete fixed-width sample buffer.
///
/// # Examples
///
/// ```
/// use ritk_codecs::{ByteOrder, SampleBuffer, SampleError, SampleType};
///
/// let error = SampleBuffer::decode(
///     SampleType::U16,
///     &[0x01, 0x02, 0x03],
///     ByteOrder::LeastSignificantByteFirst,
/// )
/// .expect_err("partial sample");
/// assert!(matches!(
///     error,
///     SampleError::PartialSample {
///         sample_type: SampleType::U16,
///         byte_length: 3,
///         trailing_bytes: 1,
///     }
/// ));
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum SampleError {
    /// The byte length ends with a partial stored sample.
    PartialSample {
        /// The sample representation declared by the format header.
        sample_type: SampleType,
        /// The total number of payload bytes.
        byte_length: usize,
        /// Bytes left after consuming complete samples.
        trailing_bytes: usize,
    },
    /// The requested byte allocation could not be reserved.
    Allocation(TryReserveError),
    /// The output byte length overflowed `usize` arithmetic.
    EncodedLengthOverflow {
        /// Number of samples to encode.
        sample_count: usize,
        /// Width of each stored sample.
        sample_width: usize,
    },
    /// The locked scalar codec rejected an exact-width slice.
    ScalarCodecRejected {
        /// The sample representation being decoded or encoded.
        sample_type: SampleType,
    },
}

impl fmt::Display for SampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PartialSample {
                sample_type,
                byte_length,
                trailing_bytes,
            } => write!(
                formatter,
                "{sample_type:?} payload of {byte_length} bytes has {trailing_bytes} trailing bytes"
            ),
            Self::Allocation(error) => write!(formatter, "cannot reserve sample buffer: {error}"),
            Self::EncodedLengthOverflow {
                sample_count,
                sample_width,
            } => write!(
                formatter,
                "encoded length overflows for {sample_count} samples of {sample_width} bytes"
            ),
            Self::ScalarCodecRejected { sample_type } => write!(
                formatter,
                "fixed-width scalar codec rejected an exact {sample_type:?} sample"
            ),
        }
    }
}

impl Error for SampleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            Self::PartialSample { .. }
            | Self::EncodedLengthOverflow { .. }
            | Self::ScalarCodecRejected { .. } => None,
        }
    }
}

/// A typed extraction failure that retains the original samples.
///
/// # Examples
///
/// ```
/// use ritk_codecs::{SampleBuffer, SampleType};
///
/// let error = SampleBuffer::from_samples(vec![9_007_199_254_740_993_u64])
///     .try_into_samples::<u32>()
///     .expect_err("mismatched sample type");
/// assert_eq!(error.requested_type(), SampleType::U32);
/// assert_eq!(error.actual_type(), SampleType::U64);
/// assert_eq!(error.into_buffer().try_into_samples::<u64>()?, [9_007_199_254_740_993]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct SampleExtractionError {
    requested: SampleType,
    buffer: SampleBuffer,
}

impl SampleExtractionError {
    pub(super) fn new(requested: SampleType, buffer: SampleBuffer) -> Self {
        Self { requested, buffer }
    }

    /// Returns the requested sample representation.
    #[must_use]
    pub const fn requested_type(&self) -> SampleType {
        self.requested
    }

    /// Returns the actual representation retained in the buffer.
    #[must_use]
    pub const fn actual_type(&self) -> SampleType {
        self.buffer.sample_type()
    }

    /// Recovers the unchanged buffer after a type mismatch.
    #[must_use]
    pub fn into_buffer(self) -> SampleBuffer {
        self.buffer
    }
}

impl fmt::Display for SampleExtractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot extract {:?} samples from a {:?} buffer",
            self.requested,
            self.actual_type()
        )
    }
}

impl Error for SampleExtractionError {}
