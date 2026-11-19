use std::borrow::Cow;

use consus_core::ByteOrder;

use super::PrimitiveValue;

impl PrimitiveValue {
    pub fn cardinality(&self) -> usize {
        use PrimitiveValue::*;
        match self {
            Empty => 0,
            Bytes(v) => v.len(),
            Str(_) => 1,
            Strs(v) => v.len(),
            I16(v) => v.len(),
            U16(v) => v.len(),
            I32(v) => v.len(),
            U32(v) => v.len(),
            I64(v) => v.len(),
            U64(v) => v.len(),
            F32(v) => v.len(),
            F64(v) => v.len(),
            Tags(v) => v.len(),
        }
    }

    pub const fn value_type(&self) -> &'static str {
        use PrimitiveValue::*;
        match self {
            Empty => "Empty",
            Bytes(_) => "Bytes",
            Str(_) => "Str",
            Strs(_) => "Strs",
            I16(_) => "I16",
            U16(_) => "U16",
            I32(_) => "I32",
            U32(_) => "U32",
            I64(_) => "I64",
            U64(_) => "U64",
            F32(_) => "F32",
            F64(_) => "F64",
            Tags(_) => "Tags",
        }
    }

    pub fn to_bytes(&self) -> Cow<'_, [u8]> {
        self.encoded_bytes(ByteOrder::LittleEndian)
    }

    pub(crate) fn encoded_bytes(&self, order: ByteOrder) -> Cow<'_, [u8]> {
        use PrimitiveValue::*;
        macro_rules! numbers {
            ($values:expr) => {
                Cow::Owned(
                    $values
                        .iter()
                        .flat_map(|&value| match order {
                            ByteOrder::LittleEndian => value.to_le_bytes(),
                            ByteOrder::BigEndian => value.to_be_bytes(),
                        })
                        .collect(),
                )
            };
        }
        match self {
            Empty => Cow::Borrowed(&[]),
            Bytes(bytes) => Cow::Borrowed(bytes),
            Str(s) => Cow::Borrowed(s.as_bytes()),
            Strs(s) => Cow::Owned(s.join("\\").into_bytes()),
            I16(v) => numbers!(v),
            U16(v) => numbers!(v),
            I32(v) => numbers!(v),
            U32(v) => numbers!(v),
            I64(v) => numbers!(v),
            U64(v) => numbers!(v),
            F32(v) => numbers!(v),
            F64(v) => numbers!(v),
            Tags(v) => Cow::Owned(
                v.iter()
                    .flat_map(|tag| [tag.group, tag.element])
                    .flat_map(|value| match order {
                        ByteOrder::LittleEndian => value.to_le_bytes(),
                        ByteOrder::BigEndian => value.to_be_bytes(),
                    })
                    .collect(),
            ),
        }
    }
}
