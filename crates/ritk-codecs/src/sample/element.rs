use consus_core::decode::EndianScalar;

use super::buffer::{SampleBuffer, SampleType, StoredSamples};
use super::error::SampleExtractionError;

mod sealed {
    pub trait Sealed {}
}

/// A fixed-width sample type supported by RITK image formats.
///
/// Implementations are limited to the ten numeric representations listed by
/// [`SampleType::ALL`]. The trait describes stored bytes only; it does not
/// require an algorithm or tensor scalar trait.
///
/// # Examples
///
/// ```
/// use ritk_codecs::{Sample, SampleType};
///
/// assert_eq!(u16::SAMPLE_TYPE, SampleType::U16);
/// let buffer = u16::into_sample_buffer(vec![1, 257]);
/// assert_eq!(u16::try_from_sample_buffer(buffer)?, [1, 257]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Sample: sealed::Sealed + EndianScalar + Copy + 'static {
    /// The stored representation corresponding to this Rust type.
    const SAMPLE_TYPE: SampleType;

    /// Moves samples into a type-preserving buffer.
    fn into_sample_buffer(samples: Vec<Self>) -> SampleBuffer;

    /// Recovers samples from a matching buffer, preserving it on mismatch.
    ///
    /// # Errors
    ///
    /// Returns the unchanged buffer when its stored representation differs
    /// from `Self::SAMPLE_TYPE`.
    fn try_from_sample_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleExtractionError>;
}

macro_rules! impl_sample {
    ($sample:ty, $variant:ident) => {
        impl sealed::Sealed for $sample {}

        impl Sample for $sample {
            const SAMPLE_TYPE: SampleType = SampleType::$variant;

            fn into_sample_buffer(samples: Vec<Self>) -> SampleBuffer {
                SampleBuffer {
                    samples: StoredSamples::$variant(samples),
                }
            }

            fn try_from_sample_buffer(
                buffer: SampleBuffer,
            ) -> Result<Vec<Self>, SampleExtractionError> {
                match buffer.samples {
                    StoredSamples::$variant(samples) => Ok(samples),
                    samples => Err(SampleExtractionError::new(
                        Self::SAMPLE_TYPE,
                        SampleBuffer { samples },
                    )),
                }
            }
        }
    };
}

impl_sample!(u8, U8);
impl_sample!(i8, I8);
impl_sample!(u16, U16);
impl_sample!(i16, I16);
impl_sample!(u32, U32);
impl_sample!(i32, I32);
impl_sample!(u64, U64);
impl_sample!(i64, I64);
impl_sample!(f32, F32);
impl_sample!(f64, F64);
