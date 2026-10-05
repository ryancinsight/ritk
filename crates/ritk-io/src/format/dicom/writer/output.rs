//! Persist fully serialized DICOM slices after input preflight succeeds.

use crate::format::dicom::writer::error::DicomWriteError;
use anyhow::{bail, Context, Result};
use dicom::core::{Tag, VR};
use std::path::Path;

pub(crate) fn serialize_file(object: &dicom::object::DefaultDicomObject) -> Result<Vec<u8>> {
    validate_pixel_module(object)?;
    let mut bytes = Vec::new();
    object
        .write_all(&mut bytes)
        .context("DICOM serialization failed")?;
    Ok(bytes)
}

fn unsigned_scalar(
    object: &dicom::object::DefaultDicomObject,
    tag: Tag,
    name: &'static str,
) -> Result<u16> {
    object
        .element(tag)
        .map_err(|_| DicomWriteError::MissingPixelAttribute { attribute: name })?
        .to_int::<u16>()
        .map_err(|_| DicomWriteError::InvalidSourcePixelDescription {
            bits_allocated: 0,
            bits_stored: 0,
            high_bit: 0,
        })
        .map_err(Into::into)
}

fn validate_pixel_module(object: &dicom::object::DefaultDicomObject) -> Result<()> {
    let Ok(pixel_data) = object.element(Tag(0x7FE0, 0x0010)) else {
        return Ok(());
    };
    let bits_allocated = unsigned_scalar(object, Tag(0x0028, 0x0100), "BitsAllocated")?;
    let bits_stored = unsigned_scalar(object, Tag(0x0028, 0x0101), "BitsStored")?;
    let high_bit = unsigned_scalar(object, Tag(0x0028, 0x0102), "HighBit")?;
    let pixel_representation = unsigned_scalar(object, Tag(0x0028, 0x0103), "PixelRepresentation")?;
    if !(bits_allocated == 1 || bits_allocated.is_multiple_of(8))
        || bits_stored == 0
        || bits_stored > bits_allocated
        || high_bit != bits_stored - 1
        || pixel_representation > 1
    {
        return Err(DicomWriteError::InvalidSourcePixelDescription {
            bits_allocated,
            bits_stored,
            high_bit,
        }
        .into());
    }
    if matches!(pixel_data.vr(), VR::OB | VR::OW) {
        let (Ok(rows), Ok(columns)) = (
            unsigned_scalar(object, Tag(0x0028, 0x0010), "Rows"),
            unsigned_scalar(object, Tag(0x0028, 0x0011), "Columns"),
        ) else {
            return Ok(());
        };
        let samples = usize::from(rows)
            .checked_mul(usize::from(columns))
            .ok_or(DicomWriteError::PixelCountOverflow)?;
        let expected = if bits_allocated == 1 {
            samples
                .checked_add(7)
                .ok_or(DicomWriteError::PixelCountOverflow)?
                / 8
        } else {
            samples
                .checked_mul(usize::from(bits_allocated / 8))
                .ok_or(DicomWriteError::PixelCountOverflow)?
        };
        let actual = pixel_data
            .to_bytes()
            .map_err(|_| DicomWriteError::PixelPayloadLengthMismatch {
                expected,
                actual: 0,
            })?
            .len();
        if actual != expected {
            return Err(DicomWriteError::PixelPayloadLengthMismatch { expected, actual }.into());
        }
    }
    Ok(())
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
