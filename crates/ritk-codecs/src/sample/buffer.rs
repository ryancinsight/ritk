//! Owned sample buffers whose element type a file header selects.

use consus_core::ByteOrder;

use super::{decode_samples, Sample, SampleError, SampleType};

/// Samples in the type a file stores them, chosen at run time.
///
/// A reader returns this when the element type comes from a header: each
/// variant owns a vector of its primitive, so no sample is converted until a
/// caller asks for a type with [`into_vec`](Self::into_vec). Dispatch over the
/// closed set is an exhaustive `match`, never a vtable.
#[derive(Debug, Clone, PartialEq)]
pub enum SampleBuffer {
    /// `u8` samples.
    U8(Vec<u8>),
    /// `i8` samples.
    I8(Vec<i8>),
    /// `u16` samples.
    U16(Vec<u16>),
    /// `i16` samples.
    I16(Vec<i16>),
    /// `u32` samples.
    U32(Vec<u32>),
    /// `i32` samples.
    I32(Vec<i32>),
    /// `u64` samples.
    U64(Vec<u64>),
    /// `i64` samples.
    I64(Vec<i64>),
    /// `f32` samples.
    F32(Vec<f32>),
    /// `f64` samples.
    F64(Vec<f64>),
}

impl SampleBuffer {
    /// Decode `bytes`, packed samples of `sample_type` stored in `order`.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError::PartialSample`] when `bytes.len()` is not a
    /// whole number of samples.
    pub fn decode(
        bytes: &[u8],
        sample_type: SampleType,
        order: ByteOrder,
    ) -> Result<Self, SampleError> {
        Ok(match sample_type {
            SampleType::U8 => Self::U8(decode_samples(bytes, order)?),
            SampleType::I8 => Self::I8(decode_samples(bytes, order)?),
            SampleType::U16 => Self::U16(decode_samples(bytes, order)?),
            SampleType::I16 => Self::I16(decode_samples(bytes, order)?),
            SampleType::U32 => Self::U32(decode_samples(bytes, order)?),
            SampleType::I32 => Self::I32(decode_samples(bytes, order)?),
            SampleType::U64 => Self::U64(decode_samples(bytes, order)?),
            SampleType::I64 => Self::I64(decode_samples(bytes, order)?),
            SampleType::F32 => Self::F32(decode_samples(bytes, order)?),
            SampleType::F64 => Self::F64(decode_samples(bytes, order)?),
        })
    }

    /// The stored element type.
    #[must_use]
    pub fn sample_type(&self) -> SampleType {
        match self {
            Self::U8(_) => SampleType::U8,
            Self::I8(_) => SampleType::I8,
            Self::U16(_) => SampleType::U16,
            Self::I16(_) => SampleType::I16,
            Self::U32(_) => SampleType::U32,
            Self::I32(_) => SampleType::I32,
            Self::U64(_) => SampleType::U64,
            Self::I64(_) => SampleType::I64,
            Self::F32(_) => SampleType::F32,
            Self::F64(_) => SampleType::F64,
        }
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Whether the buffer holds no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The samples as `T`.
    ///
    /// When the buffer already holds `T` its vector is moved out unchanged,
    /// with no copy. Otherwise every sample converts directly from its stored
    /// type to `T` by [`FromSample`](super::FromSample), which is exact
    /// whenever `T` represents every value of the stored type.
    #[must_use]
    pub fn into_vec<T: Sample>(self) -> Vec<T> {
        match T::try_from_buffer(self) {
            Ok(values) => values,
            Err(Self::U8(values)) => values.into_iter().map(T::cast_from_u8).collect(),
            Err(Self::I8(values)) => values.into_iter().map(T::cast_from_i8).collect(),
            Err(Self::U16(values)) => values.into_iter().map(T::cast_from_u16).collect(),
            Err(Self::I16(values)) => values.into_iter().map(T::cast_from_i16).collect(),
            Err(Self::U32(values)) => values.into_iter().map(T::cast_from_u32).collect(),
            Err(Self::I32(values)) => values.into_iter().map(T::cast_from_i32).collect(),
            Err(Self::U64(values)) => values.into_iter().map(T::cast_from_u64).collect(),
            Err(Self::I64(values)) => values.into_iter().map(T::cast_from_i64).collect(),
            Err(Self::F32(values)) => values.into_iter().map(T::cast_from_f32).collect(),
            Err(Self::F64(values)) => values.into_iter().map(T::cast_from_f64).collect(),
        }
    }
}
