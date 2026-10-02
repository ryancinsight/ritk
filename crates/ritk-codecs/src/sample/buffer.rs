//! Owned sample buffers whose element type a file header selects.

use consus_core::{decode_extend, ByteOrder};

use super::{Sample, SampleConversionError, SampleError, SampleType};

/// Samples in the type a file stores them, chosen at run time.
///
/// A reader returns this when the element type comes from a header: each
/// variant owns a vector of its primitive, so no sample is converted until a
/// caller asks for a type with [`into_vec`](Self::into_vec) or
/// [`cast_into_vec`](Self::cast_into_vec). Dispatch over the
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
    /// The bytes decode in one pass through consus-core's bulk decoder, which
    /// resolves the byte order once per buffer and writes into a vector
    /// reserved to the exact sample count.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError::PartialSample`] when `bytes.len()` is not a
    /// whole number of samples, or [`SampleError::Allocation`] when the
    /// decoded vector cannot reserve the required capacity.
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

    /// The samples as `T`, exactly.
    ///
    /// When the buffer holds `T` its vector is moved out unchanged, with no
    /// copy. When every value of the stored type is exactly a `T` value
    /// ([`SampleType::widens_to`]) the samples convert, each value kept.
    /// Any other request returns the buffer unchanged in the error, so this
    /// read never truncates, wraps, saturates, rounds, or collapses a NaN;
    /// [`cast_into_vec`](Self::cast_into_vec) is the explicit lossy
    /// conversion.
    ///
    /// # Errors
    ///
    /// Returns [`SampleConversionError`] owning `self` when the stored type
    /// does not widen to `T`, or when the destination allocation fails.
    pub fn into_vec<T: Sample>(self) -> Result<Vec<T>, SampleConversionError> {
        match T::try_from_buffer(self) {
            Ok(values) => Ok(values),
            Err(buffer) if buffer.sample_type().widens_to(T::TYPE) => {
                buffer.convert_into_vec::<T>()
            }
            Err(buffer) => Err(SampleConversionError::inexact(buffer, T::TYPE)),
        }
    }

    /// The samples as `T` under Rust's primitive numeric conversion rules.
    ///
    /// Integer sources use exact signed or unsigned 64-bit carriers, and an
    /// `f32` source may widen exactly to `f64`; these carriers preserve the
    /// source value until the conversion to `T`. Integers truncate to a
    /// narrower integer or wrap across signedness, floats round to the nearest
    /// `T` float, and a float into an integer rounds toward zero and saturates,
    /// with NaN at zero.
    /// The conversion is exact for every sample exactly when the stored type
    /// [widens](SampleType::widens_to) to `T`; a caller accepting a lossy
    /// pair decides how to surface that.
    /// # Errors
    ///
    /// Returns [`SampleConversionError`] owning the original buffer when the
    /// destination allocation fails.
    pub fn cast_into_vec<T: Sample>(self) -> Result<Vec<T>, SampleConversionError> {
        self.convert_into_vec::<T>()
    }

    fn convert_into_vec<T: Sample>(self) -> Result<Vec<T>, SampleConversionError> {
        match T::try_from_buffer(self) {
            Ok(values) => Ok(values),
            Err(buffer) => {
                let mut converted = Vec::new();
                if let Err(source) = converted.try_reserve_exact(buffer.len()) {
                    return Err(SampleConversionError::allocation(buffer, T::TYPE, source));
                }
                match buffer {
                    Self::U8(values) => converted.extend(
                        values
                            .into_iter()
                            .map(|value| T::from_unsigned_sample(u64::from(value))),
                    ),
                    Self::I8(values) => converted.extend(
                        values
                            .into_iter()
                            .map(|value| T::from_signed_sample(i64::from(value))),
                    ),
                    Self::U16(values) => converted.extend(
                        values
                            .into_iter()
                            .map(|value| T::from_unsigned_sample(u64::from(value))),
                    ),
                    Self::I16(values) => converted.extend(
                        values
                            .into_iter()
                            .map(|value| T::from_signed_sample(i64::from(value))),
                    ),
                    Self::U32(values) => converted.extend(
                        values
                            .into_iter()
                            .map(|value| T::from_unsigned_sample(u64::from(value))),
                    ),
                    Self::I32(values) => converted.extend(
                        values
                            .into_iter()
                            .map(|value| T::from_signed_sample(i64::from(value))),
                    ),
                    Self::U64(values) => {
                        converted.extend(values.into_iter().map(T::from_unsigned_sample))
                    }
                    Self::I64(values) => {
                        converted.extend(values.into_iter().map(T::from_signed_sample))
                    }
                    Self::F32(values) => converted.extend(
                        values
                            .into_iter()
                            .map(|value| T::from_real_sample(f64::from(value))),
                    ),
                    Self::F64(values) => {
                        converted.extend(values.into_iter().map(T::from_real_sample))
                    }
                }
                Ok(converted)
            }
        }
    }
}

/// Decode `bytes`, packed `T` samples stored in `order`, into a vector of `T`.
fn decode_samples<T: Sample>(bytes: &[u8], order: ByteOrder) -> Result<Vec<T>, SampleError> {
    let sample_width = T::TYPE.byte_width();
    if !bytes.len().is_multiple_of(sample_width) {
        return Err(SampleError::PartialSample {
            sample_type: T::TYPE,
            byte_len: bytes.len(),
        });
    }
    let sample_count = bytes.len() / sample_width;
    let mut values = Vec::new();
    values
        .try_reserve_exact(sample_count)
        .map_err(|source| SampleError::Allocation {
            sample_type: T::TYPE,
            sample_count,
            source,
        })?;
    decode_extend::<T, T>(bytes, order, &mut values, |value| value).ok_or(
        SampleError::PartialSample {
            sample_type: T::TYPE,
            byte_len: bytes.len(),
        },
    )?;
    Ok(values)
}
