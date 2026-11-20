use std::str::FromStr;

/// A Rust primitive numeric target for converting a DICOM attribute.
///
/// A binary float converted to an integer truncates toward zero and must
/// fit the result type. Binary conversions to a float use the target's IEEE
/// representation. Text values use the target's numeric parser.
///
/// The supported targets are the primitive signed and unsigned integers,
/// including pointer-sized integers, and the two IEEE float primitives.
pub trait DicomNumber: FromStr + private::FromBinary {}

impl<T: FromStr + private::FromBinary> DicomNumber for T {}

mod private {
    use super::BinaryNumber;

    pub trait FromBinary: Sized {
        fn from_binary(value: BinaryNumber) -> Option<Self>;
    }
}

use private::FromBinary;

// Narrower integer VRs extend to these storage widths without loss. Each
// floating VR retains its own representation until the target conversion.
pub enum BinaryNumber {
    Signed(i64),
    Unsigned(u64),
    Single(f32),
    Double(f64),
}

macro_rules! extend_integer {
    ($variant:ident; $($scalar:ty),+) => {$(
        impl From<$scalar> for BinaryNumber {
            #[inline]
            fn from(value: $scalar) -> Self {
                Self::$variant(value.into())
            }
        }
    )+};
}
extend_integer!(Signed; i16, i32, i64);
extend_integer!(Unsigned; u8, u16, u32, u64);

impl From<f32> for BinaryNumber {
    #[inline]
    fn from(value: f32) -> Self {
        Self::Single(value)
    }
}

impl From<f64> for BinaryNumber {
    #[inline]
    fn from(value: f64) -> Self {
        Self::Double(value)
    }
}

// In float precision, MIN is zero or a signed power of two. MAX is one
// below the next power of two. The conversion either represents MAX
// exactly (then +1 is exact), or rounds it to that upper bound already.
// A strict upper comparison also rules out infinity, including for u128
// in single precision, where the bound itself is infinity.
macro_rules! integer_targets {
    ($($target:ty),+) => {$(
        impl FromBinary for $target {
            #[inline]
            fn from_binary(value: BinaryNumber) -> Option<Self> {
                match value {
                    BinaryNumber::Signed(value) => Self::try_from(value).ok(),
                    BinaryNumber::Unsigned(value) => Self::try_from(value).ok(),
                    BinaryNumber::Single(value) => in_bounds(
                        value.trunc(), Self::MIN as f32, Self::MAX as f32 + 1.0,
                    ).map(|value| value as Self),
                    BinaryNumber::Double(value) => in_bounds(
                        value.trunc(), Self::MIN as f64, Self::MAX as f64 + 1.0,
                    ).map(|value| value as Self),
                }
            }
        }
    )+};
}

#[inline]
fn in_bounds<T: PartialOrd>(truncated: T, lower: T, upper: T) -> Option<T> {
    (lower <= truncated && truncated < upper).then_some(truncated)
}

integer_targets!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl FromBinary for f32 {
    #[inline]
    fn from_binary(value: BinaryNumber) -> Option<Self> {
        Some(match value {
            BinaryNumber::Signed(value) => value as Self,
            BinaryNumber::Unsigned(value) => value as Self,
            BinaryNumber::Single(value) => value,
            BinaryNumber::Double(value) => value as Self,
        })
    }
}

impl FromBinary for f64 {
    #[inline]
    fn from_binary(value: BinaryNumber) -> Option<Self> {
        Some(match value {
            BinaryNumber::Signed(value) => value as Self,
            BinaryNumber::Unsigned(value) => value as Self,
            BinaryNumber::Single(value) => Self::from(value),
            BinaryNumber::Double(value) => value,
        })
    }
}
