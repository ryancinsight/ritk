//! MGH scalar type metadata.

use crate::{MghError, MRI_FLOAT, MRI_INT, MRI_SHORT, MRI_UCHAR};

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

impl TryFrom<i32> for VoxelType {
    type Error = MghError;

    fn try_from(value: i32) -> Result<Self, MghError> {
        match value {
            MRI_UCHAR => Ok(Self::UnsignedByte),
            MRI_SHORT => Ok(Self::SignedShort),
            MRI_INT => Ok(Self::SignedInteger),
            MRI_FLOAT => Ok(Self::Float),
            code => Err(MghError::UnsupportedDataType { code }),
        }
    }
}
