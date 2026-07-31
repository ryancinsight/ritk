//! MGH scalar type metadata.

use crate::{MRI_FLOAT, MRI_INT, MRI_SHORT, MRI_UCHAR};
use anyhow::{bail, Result};

/// Validated MGH scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VoxelType {
    /// Unsigned 8-bit integer.
    UnsignedByte,
    /// Signed 16-bit integer.
    SignedShort,
    /// Signed 32-bit integer.
    SignedInteger,
    /// IEEE 754 32-bit float.
    Float,
}

impl VoxelType {
    /// Encoded bytes occupied by one voxel.
    pub(crate) const fn bytes_per_voxel(self) -> usize {
        match self {
            Self::UnsignedByte => 1,
            Self::SignedShort => 2,
            Self::SignedInteger | Self::Float => 4,
        }
    }
}

impl TryFrom<i32> for VoxelType {
    type Error = anyhow::Error;

    fn try_from(value: i32) -> Result<Self> {
        match value {
            MRI_UCHAR => Ok(Self::UnsignedByte),
            MRI_SHORT => Ok(Self::SignedShort),
            MRI_INT => Ok(Self::SignedInteger),
            MRI_FLOAT => Ok(Self::Float),
            other => bail!("Unsupported MGH data type code: {}", other),
        }
    }
}
