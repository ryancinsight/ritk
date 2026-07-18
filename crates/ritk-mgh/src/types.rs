//! MGH scalar type metadata.

use crate::{MRI_FLOAT, MRI_INT, MRI_SHORT, MRI_UCHAR};
use anyhow::{bail, Result};

pub(crate) fn bytes_per_voxel(mri_type: i32) -> Result<usize> {
    match mri_type {
        MRI_UCHAR => Ok(1),
        MRI_SHORT => Ok(2),
        MRI_INT | MRI_FLOAT => Ok(4),
        other => bail!("Unsupported MGH data type code: {}", other) }
}
