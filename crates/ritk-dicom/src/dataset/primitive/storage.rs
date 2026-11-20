use bytes::Bytes;
use smallvec::SmallVec;

use crate::DicomTag;

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
