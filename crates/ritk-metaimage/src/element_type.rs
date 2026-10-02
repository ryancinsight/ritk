//! The MetaImage `ElementType` names and the sample types they store.

use anyhow::{anyhow, Result};
use ritk_codecs::sample::SampleType;

/// The sample type a MetaImage `ElementType` names.
///
/// `MET_LONG` and `MET_ULONG` read as `i32` and `u32`: MetaIO stores them in
/// four bytes on every platform (`MET_ValueTypeSize` in MetaIO's
/// `src/metaTypes.h`), and the writer names those types `MET_INT` and
/// `MET_UINT`.
///
/// # Errors
///
/// Returns an error for a name outside `MET_CHAR`, `MET_UCHAR`, `MET_SHORT`,
/// `MET_USHORT`, `MET_INT`, `MET_UINT`, `MET_LONG`, `MET_ULONG`,
/// `MET_LONG_LONG`, `MET_ULONG_LONG`, `MET_FLOAT`, and `MET_DOUBLE`, the
/// fixed-width numeric types a RITK sample can hold.
pub(crate) fn sample_type_from_element_type(element_type: &str) -> Result<SampleType> {
    match element_type {
        "MET_LONG" => Ok(SampleType::I32),
        "MET_ULONG" => Ok(SampleType::U32),
        name => SampleType::ALL
            .into_iter()
            .find(|sample_type| element_type_name(*sample_type) == name)
            .ok_or_else(|| anyhow!("Unsupported MetaImage ElementType: '{name}'")),
    }
}

/// The `ElementType` name that stores `sample_type`.
pub(crate) const fn element_type_name(sample_type: SampleType) -> &'static str {
    match sample_type {
        SampleType::I8 => "MET_CHAR",
        SampleType::U8 => "MET_UCHAR",
        SampleType::I16 => "MET_SHORT",
        SampleType::U16 => "MET_USHORT",
        SampleType::I32 => "MET_INT",
        SampleType::U32 => "MET_UINT",
        SampleType::I64 => "MET_LONG_LONG",
        SampleType::U64 => "MET_ULONG_LONG",
        SampleType::F32 => "MET_FLOAT",
        SampleType::F64 => "MET_DOUBLE",
    }
}
