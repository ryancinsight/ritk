//! Compile-time types supported by volume sample buffers.

use coeus_core::Scalar;
use consus_core::EndianScalar;
use eunomia::CastFrom;

use super::{SampleBuffer, SampleType};

mod sealed {
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

/// A fixed-width numeric type supported by volume formats.
///
/// The implementations are the ten primitive integer and floating-point
/// types named by [`SampleType`]. `CastFrom` is used only by the explicit
/// conversion operations; extracting the stored type moves its original
/// vector without a cast.
pub trait Sample:
    Scalar
    + EndianScalar
    + sealed::Sealed
    + CastFrom<u8>
    + CastFrom<i8>
    + CastFrom<u16>
    + CastFrom<i16>
    + CastFrom<u32>
    + CastFrom<i32>
    + CastFrom<u64>
    + CastFrom<i64>
    + CastFrom<f32>
    + CastFrom<f64>
{
    /// The runtime descriptor of `Self`.
    const TYPE: SampleType;

    /// Wrap a vector of `Self` as its [`SampleBuffer`] variant without copying.
    fn into_buffer(values: Vec<Self>) -> SampleBuffer;

    /// Take the vector out of `buffer` when it stores `Self`.
    ///
    /// # Errors
    ///
    /// Returns the unchanged buffer when its stored type differs from `Self`.
    ///
    /// # Errors
    ///
    /// Returns the unchanged [`SampleBuffer`] when its type is not `Self`.
    fn try_from_buffer(buffer: SampleBuffer) -> Result<Vec<Self>, SampleBuffer>;

    /// Compare the complete primitive representation of two samples.
    ///
    /// Floating-point samples compare by bits, preserving distinctions such as
    /// signed zero and NaN payloads. Integer samples compare by value, whose
    /// fixed-width representation is unique.
    fn same_representation(self, other: Self) -> bool;

    /// The exact integer represented by this sample, if it is integral.
    ///
    /// Integer sample types always return a value. Floating-point types return
    /// one only for finite, integral values within `i128` range.
    fn exact_integer_value(self) -> Option<i128>;
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
    fn same_representation(self, other: Self) -> bool {
        self == other
    }
    fn exact_integer_value(self) -> Option<i128> {
        Some(i128::from(self))
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
    fn same_representation(self, other: Self) -> bool {
        self == other
    }
    fn exact_integer_value(self) -> Option<i128> {
        Some(i128::from(self))
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
    fn same_representation(self, other: Self) -> bool {
        self == other
    }
    fn exact_integer_value(self) -> Option<i128> {
        Some(i128::from(self))
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
    fn same_representation(self, other: Self) -> bool {
        self == other
    }
    fn exact_integer_value(self) -> Option<i128> {
        Some(i128::from(self))
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
    fn same_representation(self, other: Self) -> bool {
        self == other
    }
    fn exact_integer_value(self) -> Option<i128> {
        Some(i128::from(self))
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
    fn same_representation(self, other: Self) -> bool {
        self == other
    }
    fn exact_integer_value(self) -> Option<i128> {
        Some(i128::from(self))
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
    fn same_representation(self, other: Self) -> bool {
        self == other
    }
    fn exact_integer_value(self) -> Option<i128> {
        i128::try_from(self).ok()
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
    fn same_representation(self, other: Self) -> bool {
        self == other
    }
    fn exact_integer_value(self) -> Option<i128> {
        Some(i128::from(self))
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
    fn same_representation(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
    fn exact_integer_value(self) -> Option<i128> {
        exact_integer_value(f64::from(self))
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
    fn same_representation(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
    fn exact_integer_value(self) -> Option<i128> {
        exact_integer_value(self)
    }
}

fn exact_integer_value(value: f64) -> Option<i128> {
    let bits = value.to_bits();
    let exponent_bits = u16::try_from((bits >> 52) & 0x7FF).ok()?;
    if exponent_bits == 0x7FF {
        return None;
    }

    let negative = bits & (1_u64 << 63) != 0;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0 {
        return (fraction == 0).then_some(0);
    }

    let exponent = i32::from(exponent_bits) - 1023;
    if exponent < 0 {
        return None;
    }

    let significand = (1_u64 << 52) | fraction;
    let magnitude = if exponent >= 52 {
        let shift = u32::try_from(exponent - 52).ok()?;
        let significand = u128::from(significand);
        if shift >= u128::BITS || significand > (u128::MAX >> shift) {
            return None;
        }
        significand << shift
    } else {
        let discarded_bits = u32::try_from(52 - exponent).ok()?;
        let mask = (1_u64 << discarded_bits) - 1;
        if significand & mask != 0 {
            return None;
        }
        u128::from(significand >> discarded_bits)
    };

    if negative {
        let minimum_magnitude = 1_u128 << 127;
        if magnitude > minimum_magnitude {
            return None;
        }
        if magnitude == minimum_magnitude {
            return Some(i128::MIN);
        }
        i128::try_from(magnitude).ok()?.checked_neg()
    } else {
        i128::try_from(magnitude).ok()
    }
}
