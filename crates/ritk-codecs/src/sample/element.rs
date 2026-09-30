//! Compile-time sample types and the conversions between them.

use coeus_core::Scalar;
use num_traits::AsPrimitive;

use super::{SampleBuffer, SampleType};

/// Numeric conversion from every stored sample type into `Self`.
///
/// Each method is Rust's primitive `as` cast from its source type, applied
/// directly with no intermediate type: an integer narrows by truncation and
/// widens by sign or zero extension, an integer becomes a float by rounding to
/// nearest, and a float becomes an integer by rounding toward zero, saturating
/// at the target range, with NaN mapping to zero. Implemented once for every
/// primitive that all ten stored types cast to.
pub trait FromSample: Copy + 'static {
    /// Convert a `u8` sample.
    fn cast_from_u8(value: u8) -> Self;
    /// Convert an `i8` sample.
    fn cast_from_i8(value: i8) -> Self;
    /// Convert a `u16` sample.
    fn cast_from_u16(value: u16) -> Self;
    /// Convert an `i16` sample.
    fn cast_from_i16(value: i16) -> Self;
    /// Convert a `u32` sample.
    fn cast_from_u32(value: u32) -> Self;
    /// Convert an `i32` sample.
    fn cast_from_i32(value: i32) -> Self;
    /// Convert a `u64` sample.
    fn cast_from_u64(value: u64) -> Self;
    /// Convert an `i64` sample.
    fn cast_from_i64(value: i64) -> Self;
    /// Convert an `f32` sample.
    fn cast_from_f32(value: f32) -> Self;
    /// Convert an `f64` sample.
    fn cast_from_f64(value: f64) -> Self;
}

impl<T> FromSample for T
where
    T: Copy + 'static,
    u8: AsPrimitive<T>,
    i8: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    i16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    i64: AsPrimitive<T>,
    f32: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    #[inline]
    fn cast_from_u8(value: u8) -> Self {
        value.as_()
    }
    #[inline]
    fn cast_from_i8(value: i8) -> Self {
        value.as_()
    }
    #[inline]
    fn cast_from_u16(value: u16) -> Self {
        value.as_()
    }
    #[inline]
    fn cast_from_i16(value: i16) -> Self {
        value.as_()
    }
    #[inline]
    fn cast_from_u32(value: u32) -> Self {
        value.as_()
    }
    #[inline]
    fn cast_from_i32(value: i32) -> Self {
        value.as_()
    }
    #[inline]
    fn cast_from_u64(value: u64) -> Self {
        value.as_()
    }
    #[inline]
    fn cast_from_i64(value: i64) -> Self {
        value.as_()
    }
    #[inline]
    fn cast_from_f32(value: f32) -> Self {
        value.as_()
    }
    #[inline]
    fn cast_from_f64(value: f64) -> Self {
        value.as_()
    }
}

/// A fixed-width numeric type a volume format stores samples as.
///
/// Implemented for exactly the ten types [`SampleType`] enumerates. The
/// associated [`TYPE`](Self::TYPE) is the static route from a Rust type to its
/// runtime descriptor, so a `match` on `T::TYPE` folds away in each
/// monomorphization. `Scalar` makes every sample type an [`Image`] element.
///
/// [`Image`]: https://docs.rs/ritk-image
pub trait Sample: Scalar + FromSample {
    /// The runtime descriptor of `Self`.
    const TYPE: SampleType;

    /// Wrap a vector of `Self` as its [`SampleBuffer`] variant without copying.
    fn into_buffer(values: Vec<Self>) -> SampleBuffer;

    /// Take the vector out of `buffer` when it holds `Self`, without copying;
    /// otherwise hand `buffer` back unchanged.
    ///
    /// # Errors
    ///
    /// Returns `buffer` itself when its variant is not `Self`.
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer>;
}

impl Sample for u8 {
    const TYPE: SampleType = SampleType::U8;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::U8(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::U8(values) => Ok(values),
            other => Err(other),
        }
    }
}

impl Sample for i8 {
    const TYPE: SampleType = SampleType::I8;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::I8(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::I8(values) => Ok(values),
            other => Err(other),
        }
    }
}

impl Sample for u16 {
    const TYPE: SampleType = SampleType::U16;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::U16(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::U16(values) => Ok(values),
            other => Err(other),
        }
    }
}

impl Sample for i16 {
    const TYPE: SampleType = SampleType::I16;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::I16(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::I16(values) => Ok(values),
            other => Err(other),
        }
    }
}

impl Sample for u32 {
    const TYPE: SampleType = SampleType::U32;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::U32(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::U32(values) => Ok(values),
            other => Err(other),
        }
    }
}

impl Sample for i32 {
    const TYPE: SampleType = SampleType::I32;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::I32(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::I32(values) => Ok(values),
            other => Err(other),
        }
    }
}

impl Sample for u64 {
    const TYPE: SampleType = SampleType::U64;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::U64(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::U64(values) => Ok(values),
            other => Err(other),
        }
    }
}

impl Sample for i64 {
    const TYPE: SampleType = SampleType::I64;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::I64(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::I64(values) => Ok(values),
            other => Err(other),
        }
    }
}

impl Sample for f32 {
    const TYPE: SampleType = SampleType::F32;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::F32(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::F32(values) => Ok(values),
            other => Err(other),
        }
    }
}

impl Sample for f64 {
    const TYPE: SampleType = SampleType::F64;
    fn into_buffer(values: Vec<Self>) -> SampleBuffer {
        SampleBuffer::F64(values)
    }
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::F64(values) => Ok(values),
            other => Err(other),
        }
    }
}
