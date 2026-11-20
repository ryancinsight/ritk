use bytes::Bytes;

use super::PrimitiveValue;
use crate::DicomTag;

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
