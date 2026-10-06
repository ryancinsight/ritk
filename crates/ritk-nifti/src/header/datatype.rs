//! NIfTI scalar datatype codes and their fixed-width stored representations.
//!
//! The datatype codes follow the official
//! [NIFTI definitions](https://github.com/NIFTI-Imaging/nifti_clib/blob/master/niftilib/nifti1.h#L441-L469).

use anyhow::{bail, Result};
use ritk_codecs::SampleType;

/// A fixed-width scalar datatype stored by a NIfTI-1 or NIfTI-2 file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NiftiDatatype {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Uint64,
    Int64,
    Float32,
    Float64,
}

impl NiftiDatatype {
    pub(crate) const fn code(self) -> i16 {
        self.header_values().0
    }

    pub(crate) const fn bitpix(self) -> i16 {
        self.header_values().1
    }

    pub(crate) const fn byte_width(self) -> usize {
        self.header_values().2
    }

    const fn header_values(self) -> (i16, i16, usize) {
        match self {
            Self::Uint8 => (2, 8, 1),
            Self::Int8 => (256, 8, 1),
            Self::Uint16 => (512, 16, 2),
            Self::Int16 => (4, 16, 2),
            Self::Uint32 => (768, 32, 4),
            Self::Int32 => (8, 32, 4),
            Self::Uint64 => (1280, 64, 8),
            Self::Int64 => (1024, 64, 8),
            Self::Float32 => (16, 32, 4),
            Self::Float64 => (64, 64, 8),
        }
    }

    pub(crate) fn from_code(code: i16) -> Result<Self> {
        match code {
            2 => Ok(Self::Uint8),
            4 => Ok(Self::Int16),
            8 => Ok(Self::Int32),
            16 => Ok(Self::Float32),
            64 => Ok(Self::Float64),
            256 => Ok(Self::Int8),
            512 => Ok(Self::Uint16),
            768 => Ok(Self::Uint32),
            1024 => Ok(Self::Int64),
            1280 => Ok(Self::Uint64),
            _ => bail!("Unsupported NIfTI datatype code {code}"),
        }
    }
}

impl TryFrom<SampleType> for NiftiDatatype {
    type Error = anyhow::Error;

    fn try_from(sample_type: SampleType) -> Result<Self, Self::Error> {
        match sample_type {
            SampleType::U8 => Ok(Self::Uint8),
            SampleType::I8 => Ok(Self::Int8),
            SampleType::U16 => Ok(Self::Uint16),
            SampleType::I16 => Ok(Self::Int16),
            SampleType::U32 => Ok(Self::Uint32),
            SampleType::I32 => Ok(Self::Int32),
            SampleType::U64 => Ok(Self::Uint64),
            SampleType::I64 => Ok(Self::Int64),
            SampleType::F32 => Ok(Self::Float32),
            SampleType::F64 => Ok(Self::Float64),
            _ => bail!("Unsupported RITK stored sample type {sample_type:?}"),
        }
    }
}

impl From<NiftiDatatype> for SampleType {
    fn from(datatype: NiftiDatatype) -> Self {
        match datatype {
            NiftiDatatype::Uint8 => Self::U8,
            NiftiDatatype::Int8 => Self::I8,
            NiftiDatatype::Uint16 => Self::U16,
            NiftiDatatype::Int16 => Self::I16,
            NiftiDatatype::Uint32 => Self::U32,
            NiftiDatatype::Int32 => Self::I32,
            NiftiDatatype::Uint64 => Self::U64,
            NiftiDatatype::Int64 => Self::I64,
            NiftiDatatype::Float32 => Self::F32,
            NiftiDatatype::Float64 => Self::F64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NiftiDatatype;
    use ritk_codecs::SampleType;

    #[test]
    fn stored_sample_types_round_trip_through_standard_nifti_codes() {
        let mappings = [
            (SampleType::U8, 2, 8, 1),
            (SampleType::I8, 256, 8, 1),
            (SampleType::U16, 512, 16, 2),
            (SampleType::I16, 4, 16, 2),
            (SampleType::U32, 768, 32, 4),
            (SampleType::I32, 8, 32, 4),
            (SampleType::U64, 1280, 64, 8),
            (SampleType::I64, 1024, 64, 8),
            (SampleType::F32, 16, 32, 4),
            (SampleType::F64, 64, 64, 8),
        ];

        for (sample_type, code, bitpix, byte_width) in mappings {
            let datatype = NiftiDatatype::try_from(sample_type).expect("supported sample type");
            assert_eq!(datatype.code(), code);
            assert_eq!(datatype.bitpix(), bitpix);
            assert_eq!(datatype.byte_width(), byte_width);
            assert_eq!(SampleType::from(datatype), sample_type);
            assert_eq!(
                NiftiDatatype::from_code(code).expect("standard code"),
                datatype
            );
        }
    }

    #[test]
    fn non_scalar_nifti_codes_remain_unsupported() {
        for code in [0, 1, 32, 128, 1536, 1792, 2048, 2304] {
            assert!(
                NiftiDatatype::from_code(code).is_err(),
                "unsupported NIfTI code {code} must be rejected"
            );
        }
    }
}
