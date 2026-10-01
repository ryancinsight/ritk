//! Runtime descriptor of a stored sample type.

use std::fmt;

/// Stored numeric type of one voxel sample.
///
/// The closed set of fixed-width numeric types that medical volume formats
/// store. Every on-disk datatype a reader parses maps onto exactly one variant,
/// and every [`Sample`](super::Sample) type names its variant in
/// [`Sample::TYPE`](super::Sample::TYPE), so a header decides the variant at
/// run time while typed code resolves it at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleType {
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
}

impl SampleType {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 10] = [
        Self::U8,
        Self::I8,
        Self::U16,
        Self::I16,
        Self::U32,
        Self::I32,
        Self::U64,
        Self::I64,
        Self::F32,
        Self::F64,
    ];

    /// Bytes one sample occupies in a packed buffer.
    #[must_use]
    pub const fn byte_width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Whether the type is an IEEE 754 floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Whether every value of this type is exactly a value of `target`.
    ///
    /// True for `target == self` and for the lossless widenings std implements
    /// `From` for: an integer into a wider integer that holds its sign, an
    /// integer of at most 16 bits into `f32` (24-bit significand), one of at
    /// most 32 bits into `f64` (53-bit significand), and `f32` into `f64`.
    #[must_use]
    pub const fn widens_to(self, target: Self) -> bool {
        match self {
            Self::U8 => !matches!(target, Self::I8),
            Self::I8 => matches!(
                target,
                Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F32 | Self::F64
            ),
            Self::U16 => matches!(
                target,
                Self::U16 | Self::U32 | Self::U64 | Self::I32 | Self::I64 | Self::F32 | Self::F64
            ),
            Self::I16 => matches!(
                target,
                Self::I16 | Self::I32 | Self::I64 | Self::F32 | Self::F64
            ),
            Self::U32 => matches!(target, Self::U32 | Self::U64 | Self::I64 | Self::F64),
            Self::I32 => matches!(target, Self::I32 | Self::I64 | Self::F64),
            Self::U64 => matches!(target, Self::U64),
            Self::I64 => matches!(target, Self::I64),
            Self::F32 => matches!(target, Self::F32 | Self::F64),
            Self::F64 => matches!(target, Self::F64),
        }
    }

    /// The Rust primitive name of the type (`"u8"`, …, `"f64"`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for SampleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
