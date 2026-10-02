//! The MGH `type` codes and the sample types they store.

use crate::{MRI_FLOAT, MRI_INT, MRI_SHORT, MRI_UCHAR};
use anyhow::{bail, Result};
use ritk_codecs::sample::SampleType;

/// The sample type an MGH `type` header code stores.
///
/// # Errors
///
/// Returns an error for a code outside `MRI_UCHAR`, `MRI_SHORT`, `MRI_INT`,
/// and `MRI_FLOAT`.
///
/// # Source
///
/// FreeSurfer `include/mri.h` (dev, commit `f4ae227c`) defines
/// `MRI_UCHAR` 0, `MRI_INT` 1, `MRI_FLOAT` 3, and `MRI_SHORT` 4 at lines
/// 53-57. `utils/mriio.cpp` reads them in `mghRead` (line 10882; the
/// per-voxel `switch (type)` around line 11107) and writes them in `mghWrite`
/// (line 11255; the `switch (mri->type)` around line 11433). The same files
/// also define `MRI_USHRT` 10 (`mri.h:63`, `mriio.cpp:11123` and `:11440`),
/// which this crate does not yet read.
pub(crate) fn sample_type_from_code(code: i32) -> Result<SampleType> {
    Ok(match code {
        MRI_UCHAR => SampleType::U8,
        MRI_SHORT => SampleType::I16,
        MRI_INT => SampleType::I32,
        MRI_FLOAT => SampleType::F32,
        other => bail!(
            "Unsupported MGH data type code {other}; MGH stores \
             MRI_UCHAR (0), MRI_INT (1), MRI_FLOAT (3), and MRI_SHORT (4)"
        ),
    })
}

/// The MGH `type` header code that stores `sample_type`.
///
/// # Errors
///
/// Returns an error for a sample type MGH has no code for: MGH stores only
/// `u8`, `i16`, `i32`, and `f32`.
pub(crate) fn code_for(sample_type: SampleType) -> Result<i32> {
    Ok(match sample_type {
        SampleType::U8 => MRI_UCHAR,
        SampleType::I16 => MRI_SHORT,
        SampleType::I32 => MRI_INT,
        SampleType::F32 => MRI_FLOAT,
        other => bail!(
            "MGH cannot store {other} samples; it stores u8, i16, i32, and f32 — \
             convert the image to one of them first"
        ),
    })
}
