use anyhow::{bail, Result};

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
        match self {
            Self::Uint8 => 2,
            Self::Int8 => 256,
            Self::Uint16 => 512,
            Self::Int16 => 4,
            Self::Uint32 => 768,
            Self::Int32 => 8,
            Self::Uint64 => 1280,
            Self::Int64 => 1024,
            Self::Float32 => 16,
            Self::Float64 => 64,
        }
    }

    pub(super) const fn bitpix(self) -> i16 {
        match self {
            Self::Uint8 | Self::Int8 => 8,
            Self::Uint16 | Self::Int16 => 16,
            Self::Uint32 | Self::Int32 | Self::Float32 => 32,
            Self::Uint64 | Self::Int64 | Self::Float64 => 64,
        }
    }

    pub(crate) const fn byte_width(self) -> usize {
        match self {
            Self::Uint8 | Self::Int8 => 1,
            Self::Uint16 | Self::Int16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
        }
    }

    pub(crate) fn from_code(code: i16) -> Result<Self> {
        match code {
            2 => Ok(Self::Uint8),
            256 => Ok(Self::Int8),
            512 => Ok(Self::Uint16),
            4 => Ok(Self::Int16),
            768 => Ok(Self::Uint32),
            8 => Ok(Self::Int32),
            1280 => Ok(Self::Uint64),
            1024 => Ok(Self::Int64),
            16 => Ok(Self::Float32),
            64 => Ok(Self::Float64),
            _ => bail!("Unsupported scalar NIfTI datatype code {code}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NiftiDatatype;

    #[test]
    fn scalar_datatype_codes_and_widths_match_the_header_contract() {
        let datatypes = [
            (NiftiDatatype::Uint8, 2, 1),
            (NiftiDatatype::Int8, 256, 1),
            (NiftiDatatype::Uint16, 512, 2),
            (NiftiDatatype::Int16, 4, 2),
            (NiftiDatatype::Uint32, 768, 4),
            (NiftiDatatype::Int32, 8, 4),
            (NiftiDatatype::Uint64, 1280, 8),
            (NiftiDatatype::Int64, 1024, 8),
            (NiftiDatatype::Float32, 16, 4),
            (NiftiDatatype::Float64, 64, 8),
        ];

        for (datatype, code, byte_width) in datatypes {
            assert_eq!(datatype.code(), code);
            assert_eq!(datatype.byte_width(), byte_width);
            assert_eq!(
                NiftiDatatype::from_code(code).expect("supported datatype code"),
                datatype
            );
        }
    }
}
