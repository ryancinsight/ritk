//! Compile-time sample types and the conversions between them.

use consus_core::EndianScalar;

use super::super::{SampleBuffer, SampleType};
use super::primitive::PrimitiveSample;

mod sealed {
    /// The sample vocabulary is fixed by [`super::SampleType`].
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for i8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A fixed-width numeric type a volume format stores samples as.
///
/// Sealed to the ten types [`SampleType`] enumerates. The associated
/// [`TYPE`](Self::TYPE) is the static route from a Rust type to its runtime
/// descriptor, so a `match` on `T::TYPE` folds away in each monomorphization.
/// `EndianScalar` gives it consus-core's byte-order codec. Image adapters add
/// their tensor backend's scalar bound where they construct an `Image`.
///
/// Conversion methods use exact signed, unsigned, or floating-point source
/// carriers. Integer narrowing wraps to the target width; integer-to-float
/// conversion rounds to nearest, ties to even; float-to-integer conversion
/// truncates toward zero and saturates, with NaN mapping to zero; and
/// binary floating-point narrowing rounds to nearest, ties to even. These
/// rules match [Rust's numeric cast semantics]. Exact reads use these methods
/// only for type pairs whose complete stored range is representable by the
/// requested type.
///
/// [Rust's numeric cast semantics]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast
pub trait Sample: sealed::Sealed + Copy + EndianScalar {
    /// The runtime descriptor of `Self`.
    const TYPE: SampleType;

    /// Take the vector out of `buffer` when it holds `Self`, without copying;
    /// otherwise hand `buffer` back unchanged.
    ///
    /// # Errors
    ///
    /// Returns `buffer` itself when its variant is not `Self`.
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer>;

    /// Convert an unsigned stored sample directly into `Self`.
    ///
    /// Integer targets wrap at their width. Floating-point targets round to
    /// nearest, ties to even.
    fn from_unsigned_sample(value: u64) -> Self;

    /// Convert a signed stored sample directly into `Self`.
    ///
    /// Integer targets wrap at their width. Floating-point targets round to
    /// nearest, ties to even.
    fn from_signed_sample(value: i64) -> Self;

    /// Convert a real-valued stored sample directly into `Self`.
    ///
    /// Integer targets truncate toward zero and saturate; NaN maps to zero.
    /// Floating-point targets preserve signed zero and round narrowing
    /// conversion to nearest, ties to even.
    fn from_real_sample(value: f64) -> Self;
}

impl Sample for u8 {
    const TYPE: SampleType = SampleType::U8;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::U8(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}

impl Sample for i8 {
    const TYPE: SampleType = SampleType::I8;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::I8(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}

impl Sample for u16 {
    const TYPE: SampleType = SampleType::U16;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::U16(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}

impl Sample for i16 {
    const TYPE: SampleType = SampleType::I16;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::I16(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}

impl Sample for u32 {
    const TYPE: SampleType = SampleType::U32;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::U32(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}

impl Sample for i32 {
    const TYPE: SampleType = SampleType::I32;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::I32(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}

impl Sample for u64 {
    const TYPE: SampleType = SampleType::U64;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::U64(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}

impl Sample for i64 {
    const TYPE: SampleType = SampleType::I64;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::I64(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}

impl Sample for f32 {
    const TYPE: SampleType = SampleType::F32;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::F32(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}

impl Sample for f64 {
    const TYPE: SampleType = SampleType::F64;
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer> {
        match buffer {
            SampleBuffer::F64(values) => Ok(values),
            other => Err(other),
        }
    }
    fn from_unsigned_sample(value: u64) -> Self {
        <Self as PrimitiveSample>::from_unsigned_sample(value)
    }
    fn from_signed_sample(value: i64) -> Self {
        <Self as PrimitiveSample>::from_signed_sample(value)
    }
    fn from_real_sample(value: f64) -> Self {
        <Self as PrimitiveSample>::from_real_sample(value)
    }
}
