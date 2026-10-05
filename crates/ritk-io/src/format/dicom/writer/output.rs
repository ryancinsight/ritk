//! Persist fully serialized DICOM slices after input preflight succeeds.

use anyhow::{bail, Context, Result};
use std::path::Path;

pub(crate) fn serialize_file(object: &dicom::object::DefaultDicomObject) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    object
        .write_all(&mut bytes)
        .context("DICOM serialization failed")?;
    Ok(bytes)
}

pub(crate) fn write_file(path: &Path, object: &dicom::object::DefaultDicomObject) -> Result<()> {
    let bytes = serialize_file(object)?;
    std::fs::write(path, bytes).context("DICOM output write failed")
}

pub(super) fn write_series_files(path: &Path, slices: &[Vec<u8>]) -> Result<()> {
    if path.exists() {
        if !path.is_dir() {
            bail!("DICOM output path is not a directory");
        }
    } else {
        std::fs::create_dir_all(path).context("failed to create DICOM series output directory")?;
    }
    for (z, bytes) in slices.iter().enumerate() {
        std::fs::write(path.join(format!("slice_{z:04}.dcm")), bytes)
            .with_context(|| format!("write slice {z} failed"))?;
    }
    Ok(())
}
