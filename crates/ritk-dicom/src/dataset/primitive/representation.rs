use std::borrow::Cow;

use consus_core::ByteOrder;

use super::PrimitiveValue;

impl PrimitiveValue {
    pub fn cardinality(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Bytes(v) => v.len(),
            Self::Str(_) => 1,
            Self::Strs(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Tags(v) => v.len(),
        }
    }

    pub const fn value_type(&self) -> &'static str {
        match self {
            Self::Empty => "Empty",
            Self::Bytes(_) => "Bytes",
            Self::Str(_) => "Str",
            Self::Strs(_) => "Strs",
            Self::I16(_) => "I16",
            Self::U16(_) => "U16",
            Self::I32(_) => "I32",
            Self::U32(_) => "U32",
            Self::I64(_) => "I64",
            Self::U64(_) => "U64",
            Self::F32(_) => "F32",
            Self::F64(_) => "F64",
            Self::Tags(_) => "Tags",
        }
    }

    pub fn to_bytes(&self) -> Cow<'_, [u8]> {
        self.encoded_bytes(ByteOrder::LittleEndian)
    }

    pub(crate) fn encoded_bytes(&self, order: ByteOrder) -> Cow<'_, [u8]> {
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
            Self::Empty => Cow::Borrowed(&[]),
            Self::Bytes(bytes) => Cow::Borrowed(bytes),
            Self::Str(s) => Cow::Borrowed(s.as_bytes()),
            Self::Strs(s) => Cow::Owned(s.join("\\").into_bytes()),
            Self::I16(v) => numbers!(v),
            Self::U16(v) => numbers!(v),
            Self::I32(v) => numbers!(v),
            Self::U32(v) => numbers!(v),
            Self::I64(v) => numbers!(v),
            Self::U64(v) => numbers!(v),
            Self::F32(v) => numbers!(v),
            Self::F64(v) => numbers!(v),
            Self::Tags(v) => Cow::Owned(
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
