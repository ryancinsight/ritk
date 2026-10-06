use anyhow::{bail, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NiftiDatatype {
    Uint8,
    Int16,
    Int32,
    Float32,
    Uint32,
}

impl NiftiDatatype {
    pub(crate) const fn code(self) -> i16 {
        match self {
            Self::Uint8 => 2,
            Self::Int16 => 4,
            Self::Int32 => 8,
            Self::Float32 => 16,
            Self::Uint32 => 768,
        }
    }

    pub(super) const fn bitpix(self) -> i16 {
        match self {
            Self::Uint8 => 8,
            Self::Int16 => 16,
            Self::Int32 | Self::Float32 | Self::Uint32 => 32,
        }
    }

    pub(crate) const fn byte_width(self) -> usize {
        (self.bitpix() / 8) as usize
    }

    pub(crate) fn from_code(code: i16) -> Result<Self> {
        match code {
            2 => Ok(Self::Uint8),
            4 => Ok(Self::Int16),
            8 => Ok(Self::Int32),
            16 => Ok(Self::Float32),
            768 => Ok(Self::Uint32),
            _ => bail!("Unsupported NIfTI datatype code {code}"),
        }
    }
}
