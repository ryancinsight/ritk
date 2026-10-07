//! Validate and persist native Explicit VR Little Endian DICOM objects.

use crate::format::dicom::writer::error::DicomWriteError;
use anyhow::{Context, Result, bail};
use dicom::core::value::Value;
use dicom::core::{Tag, VR};
use std::path::Path;

fn value_debug<T: std::fmt::Debug>(value: &T) -> String {
    format!("{value:?}")
}

fn unsigned_scalar(
    object: &dicom::object::DefaultDicomObject,
    tag: Tag,
    name: &'static str,
) -> Result<u16> {
    let element = object
        .element(tag)
        .map_err(|_| DicomWriteError::MissingPixelAttribute { attribute: name })?;
    element.to_int::<u16>().map_err(|_| {
        DicomWriteError::MalformedPixelAttribute {
            attribute: name,
            value: value_debug(element.value()),
        }
        .into()
    })
}

fn optional_frame_count(object: &dicom::object::DefaultDicomObject) -> Result<usize> {
    let Some(element) = object.get(Tag(0x0028, 0x0008)) else {
        return Ok(1);
    };
    element.to_int::<usize>().map_err(|_| {
        DicomWriteError::MalformedPixelAttribute {
            attribute: "NumberOfFrames",
            value: value_debug(element.value()),
        }
        .into()
    })
}

fn photometric(object: &dicom::object::DefaultDicomObject) -> Result<String> {
    let element = object.element(Tag(0x0028, 0x0004)).map_err(|_| {
        DicomWriteError::MissingPixelAttribute {
            attribute: "PhotometricInterpretation",
        }
    })?;
    element
        .to_str()
        .map(|value| value.trim().to_owned())
        .map_err(|_| {
            DicomWriteError::MalformedPixelAttribute {
                attribute: "PhotometricInterpretation",
                value: value_debug(element.value()),
            }
            .into()
        })
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
    {
        return Err(DicomWriteError::InvalidSourcePixelDescription {
            bits_allocated,
            bits_stored,
            high_bit,
        }
        .into());
    }
    if pixel_representation > 1 {
        return Err(DicomWriteError::MalformedPixelAttribute {
            attribute: "PixelRepresentation",
            value: pixel_representation.to_string(),
        }
        .into());
    }
    let rows = usize::from(unsigned_scalar(object, Tag(0x0028, 0x0010), "Rows")?);
    let columns = usize::from(unsigned_scalar(object, Tag(0x0028, 0x0011), "Columns")?);
    let samples_per_pixel = usize::from(unsigned_scalar(
        object,
        Tag(0x0028, 0x0002),
        "SamplesPerPixel",
    )?);
    for (attribute, value) in [
        ("Rows", rows),
        ("Columns", columns),
        ("SamplesPerPixel", samples_per_pixel),
    ] {
        if value == 0 {
            return Err(DicomWriteError::ZeroPixelAttribute { attribute, value }.into());
        }
    }
    let number_of_frames = optional_frame_count(object)?;
    if number_of_frames == 0 {
        return Err(DicomWriteError::ZeroPixelAttribute {
            attribute: "NumberOfFrames",
            value: number_of_frames,
        }
        .into());
    }

    let photo = photometric(object)?;
    let ybr_422 = photo == "YBR_FULL_422";
    if samples_per_pixel == 1 {
        if photo != "MONOCHROME1" && photo != "MONOCHROME2" {
            return Err(DicomWriteError::UnsupportedPhotometricInterpretationValue {
                value: photo,
            }
            .into());
        }
    } else if ybr_422 {
        if bits_allocated != 8 || pixel_representation != 0 {
            return Err(DicomWriteError::UnsupportedPhotometricInterpretationValue {
                value: photo,
            }
            .into());
        }
        if object.get(Tag(0x0028, 0x0006)).is_some() {
            return Err(DicomWriteError::InvalidPlanarConfiguration {
                value: "present for YBR_FULL_422".to_owned(),
            }
            .into());
        }
    } else {
        if photo != "RGB" && photo != "YBR_FULL" {
            return Err(DicomWriteError::UnsupportedPhotometricInterpretationValue {
                value: photo,
            }
            .into());
        }
        let planar = object.element(Tag(0x0028, 0x0006)).map_err(|_| {
            DicomWriteError::MissingPixelAttribute {
                attribute: "PlanarConfiguration",
            }
        })?;
        let planar_value =
            planar
                .to_int::<u16>()
                .map_err(|_| DicomWriteError::InvalidPlanarConfiguration {
                    value: value_debug(planar.value()),
                })?;
        if planar_value > 1 {
            return Err(DicomWriteError::InvalidPlanarConfiguration {
                value: planar_value.to_string(),
            }
            .into());
        }
    }

    if matches!(pixel_data.value(), Value::PixelSequence(_)) {
        return Ok(());
    }

    let vr = pixel_data.vr();
    if !matches!(vr, VR::OB | VR::OW) {
        return Err(DicomWriteError::InvalidPixelDataVr {
            value: vr.to_string().to_owned(),
        }
        .into());
    }
    if bits_allocated > 8 && !matches!(vr, VR::OW) {
        return Err(DicomWriteError::PixelDataVrMismatch {
            value: vr.to_string().to_owned(),
            bits_allocated,
        }
        .into());
    }

    let expected = if ybr_422 {
        let groups = columns
            .checked_add(1)
            .ok_or(DicomWriteError::PixelCountOverflow)?
            / 2;
        number_of_frames
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(groups))
            .and_then(|value| value.checked_mul(4))
            .ok_or(DicomWriteError::PixelCountOverflow)?
    } else {
        let samples = number_of_frames
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(columns))
            .and_then(|value| value.checked_mul(samples_per_pixel))
            .ok_or(DicomWriteError::PixelCountOverflow)?;
        if bits_allocated == 1 {
            samples
                .checked_add(7)
                .ok_or(DicomWriteError::PixelCountOverflow)?
                / 8
        } else {
            samples
                .checked_mul(usize::from(bits_allocated / 8))
                .ok_or(DicomWriteError::PixelCountOverflow)?
        }
    };
    let bytes = pixel_data
        .to_bytes()
        .map_err(|_| DicomWriteError::MalformedPixelAttribute {
            attribute: "PixelData",
            value: value_debug(pixel_data.value()),
        })?;
    let padded = expected
        .checked_add(expected % 2)
        .ok_or(DicomWriteError::PixelCountOverflow)?;
    let actual = bytes.len();
    if actual != expected && actual != padded {
        let error = if ybr_422 {
            DicomWriteError::YbrFull422PayloadLengthMismatch { expected, actual }
        } else {
            DicomWriteError::PixelPayloadLengthMismatch { expected, actual }
        };
        return Err(error.into());
    }
    Ok(())
}

pub(crate) fn serialize_file(object: &dicom::object::DefaultDicomObject) -> Result<Vec<u8>> {
    validate_pixel_module(object)?;
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
