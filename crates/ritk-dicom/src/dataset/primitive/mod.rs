use bytes::Bytes;
use smallvec::SmallVec;

use crate::DicomTag;

mod conversion;
mod representation;

/// Scalar values have inline storage for the common one- and two-value cases.
/// Raw values share their source buffer and retain their byte padding.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveValue {
    Empty,
    Bytes(Bytes),
    Str(String),
    Strs(SmallVec<[String; 2]>),
    I16(SmallVec<[i16; 2]>),
    U16(SmallVec<[u16; 2]>),
    I32(SmallVec<[i32; 2]>),
    U32(SmallVec<[u32; 2]>),
    I64(SmallVec<[i64; 2]>),
    U64(SmallVec<[u64; 2]>),
    F32(SmallVec<[f32; 2]>),
    F64(SmallVec<[f64; 2]>),
    Tags(SmallVec<[DicomTag; 2]>),
}

macro_rules! scalar_values {
    ($($kind:ident: $scalar:ty),+ $(,)?) => {
        $(
            impl From<$scalar> for PrimitiveValue {
                fn from(value: $scalar) -> Self {
                    Self::$kind(smallvec::smallvec![value])
                }
            }

            impl<const N: usize> From<[$scalar; N]> for PrimitiveValue {
                fn from(values: [$scalar; N]) -> Self {
                    Self::$kind(values.into_iter().collect())
                }
            }
        )+
    };
}

scalar_values!(I16: i16, U16: u16, I32: i32, U32: u32, I64: i64, U64: u64,
               F32: f32, F64: f64, Tags: DicomTag);

impl From<&str> for PrimitiveValue {
    fn from(value: &str) -> Self {
        Self::Str(value.to_owned())
    }
}

impl From<String> for PrimitiveValue {
    fn from(value: String) -> Self {
        Self::Str(value)
    }
}

impl From<Bytes> for PrimitiveValue {
    fn from(value: Bytes) -> Self {
        Self::Bytes(value)
    }
}

impl From<Vec<u8>> for PrimitiveValue {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value.into())
    }
}

impl<const N: usize> From<[u8; N]> for PrimitiveValue {
    fn from(value: [u8; N]) -> Self {
        Self::Bytes(Bytes::copy_from_slice(&value))
    }
}
