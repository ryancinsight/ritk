//! Compile-time sample types and the conversions between them.

use coeus_core::Scalar;
use consus_core::EndianScalar;

use super::numeric::{
    float_to_signed, float_to_unsigned, integer_to_float, signed_to_signed, signed_to_unsigned,
    unsigned_to_signed, unsigned_to_unsigned,
};
use super::{SampleBuffer, SampleType};

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
/// `Scalar` makes every sample type an [`Image`] element, and `EndianScalar`
/// gives it consus-core's byte-order codec.
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
/// [`Image`]: https://docs.rs/ritk-image
/// [Rust's numeric cast semantics]: https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast
pub trait Sample: sealed::Sealed + Scalar + EndianScalar {
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
        u8::try_from(unsigned_to_unsigned(value, 8))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_signed_sample(value: i64) -> Self {
        u8::try_from(signed_to_unsigned(value, 8))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_real_sample(value: f64) -> Self {
        u8::try_from(float_to_unsigned(value, 8))
            .expect("invariant: saturated sample fits the target width")
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
        i8::try_from(unsigned_to_signed(value, 8))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_signed_sample(value: i64) -> Self {
        i8::try_from(signed_to_signed(value, 8))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_real_sample(value: f64) -> Self {
        i8::try_from(float_to_signed(value, 8))
            .expect("invariant: saturated sample fits the target width")
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
        u16::try_from(unsigned_to_unsigned(value, 16))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_signed_sample(value: i64) -> Self {
        u16::try_from(signed_to_unsigned(value, 16))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_real_sample(value: f64) -> Self {
        u16::try_from(float_to_unsigned(value, 16))
            .expect("invariant: saturated sample fits the target width")
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
        i16::try_from(unsigned_to_signed(value, 16))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_signed_sample(value: i64) -> Self {
        i16::try_from(signed_to_signed(value, 16))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_real_sample(value: f64) -> Self {
        i16::try_from(float_to_signed(value, 16))
            .expect("invariant: saturated sample fits the target width")
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
        u32::try_from(unsigned_to_unsigned(value, 32))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_signed_sample(value: i64) -> Self {
        u32::try_from(signed_to_unsigned(value, 32))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_real_sample(value: f64) -> Self {
        u32::try_from(float_to_unsigned(value, 32))
            .expect("invariant: saturated sample fits the target width")
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
        i32::try_from(unsigned_to_signed(value, 32))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_signed_sample(value: i64) -> Self {
        i32::try_from(signed_to_signed(value, 32))
            .expect("invariant: wrapped sample fits the target width")
    }
    fn from_real_sample(value: f64) -> Self {
        i32::try_from(float_to_signed(value, 32))
            .expect("invariant: saturated sample fits the target width")
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
        value
    }
    fn from_signed_sample(value: i64) -> Self {
        signed_to_unsigned(value, 64)
    }
    fn from_real_sample(value: f64) -> Self {
        float_to_unsigned(value, 64)
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
        unsigned_to_signed(value, 64)
    }
    fn from_signed_sample(value: i64) -> Self {
        value
    }
    fn from_real_sample(value: f64) -> Self {
        float_to_signed(value, 64)
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
        let bits = integer_to_float(value, false, 23, 127, 31);
        Self::from_bits(u32::try_from(bits).expect("invariant: binary32 bits fit u32"))
    }
    fn from_signed_sample(value: i64) -> Self {
        let bits = integer_to_float(value.unsigned_abs(), value.is_negative(), 23, 127, 31);
        Self::from_bits(u32::try_from(bits).expect("invariant: binary32 bits fit u32"))
    }
    fn from_real_sample(value: f64) -> Self {
        super::numeric::real_to_float(value, 23, 127, 31)
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
        Self::from_bits(integer_to_float(value, false, 52, 1023, 63))
    }
    fn from_signed_sample(value: i64) -> Self {
        Self::from_bits(integer_to_float(
            value.unsigned_abs(),
            value.is_negative(),
            52,
            1023,
            63,
        ))
    }
    fn from_real_sample(value: f64) -> Self {
        value
    }
}
