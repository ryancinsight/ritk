use crate::format::dicom::geometry_validation::direction_cosines_are_orthonormal;
use dicom::core::{Tag, VR};
use dicom::object::DefaultDicomObject;
use ritk_codecs::{PixelLayout, PixelSignedness};
use ritk_dicom::TransferSyntaxKind;
use ritk_image_io::{LinearCalibration, ModalityLookupTable};

use super::lut::modality_lookup_table;
use super::parse::{
    optional_decimal, optional_text, required_decimal_values, required_long_text, required_text,
    required_u16, required_usize,
};
use super::StoredDicomError;

#[derive(Clone, Copy)]
pub(super) struct SliceGeometry {
    pub(super) position: [f64; 3],
    pub(super) orientation: [f64; 6],
    pub(super) pixel_spacing: [f64; 2],
    pub(super) depth_spacing: Option<f64>,
}

pub(super) enum SliceCalibration {
    Linear {
        value: LinearCalibration,
        unit: Option<String>,
    },
    ModalityLookup {
        value: ModalityLookupTable,
        unit: String,
    },
}

pub(super) fn validate_instance(
    object: &DefaultDicomObject,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(PixelLayout, SliceCalibration, SliceGeometry), StoredDicomError> {
    let uid = object.meta().transfer_syntax().to_owned();
    if !matches!(
        TransferSyntaxKind::from_uid(&uid),
        TransferSyntaxKind::ImplicitVrLittleEndian | TransferSyntaxKind::ExplicitVrLittleEndian
    ) {
        return Err(StoredDicomError::UnsupportedTransferSyntax { uid });
    }
    let rows = required_usize(object, Tag(0x0028, 0x0010), "Rows (0028,0010)")?;
    let columns = required_usize(object, Tag(0x0028, 0x0011), "Columns (0028,0011)")?;
    let samples = required_usize(object, Tag(0x0028, 0x0002), "SamplesPerPixel (0028,0002)")?;
    if samples != 1 {
        return Err(StoredDicomError::UnsupportedSamples { samples });
    }
    let photometric = required_text(object, Tag(0x0028, 0x0004), "PhotometricInterpretation")?;
    if !matches!(photometric.trim(), "MONOCHROME1" | "MONOCHROME2") {
        return Err(StoredDicomError::UnsupportedPhotometricInterpretation { value: photometric });
    }
    let frames = optional_text(object, Tag(0x0028, 0x0008))?
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .map_err(|_| StoredDicomError::InvalidTag {
                    tag: "NumberOfFrames (0028,0008)",
                })
        })
        .transpose()?
        .unwrap_or(1);
    if frames != 1 {
        return Err(StoredDicomError::UnsupportedFrames { frames });
    }
    let bits_allocated = required_u16(object, Tag(0x0028, 0x0100), "BitsAllocated (0028,0100)")?;
    let bits_stored = required_u16(object, Tag(0x0028, 0x0101), "BitsStored (0028,0101)")?;
    let high_bit = required_u16(object, Tag(0x0028, 0x0102), "HighBit (0028,0102)")?;
    if bits_stored.checked_sub(1) != Some(high_bit) {
        return Err(StoredDicomError::InvalidTag {
            tag: "HighBit (0028,0102)",
        });
    }
    let representation = required_u16(
        object,
        Tag(0x0028, 0x0103),
        "PixelRepresentation (0028,0103)",
    )?;
    let pixel_representation =
        PixelSignedness::try_from(representation).map_err(|_| StoredDicomError::InvalidTag {
            tag: "PixelRepresentation (0028,0103)",
        })?;
    let layout = PixelLayout {
        rows,
        cols: columns,
        samples_per_pixel: samples,
        bits_allocated,
        bits_stored,
        pixel_representation,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    };
    layout
        .bytes_per_frame()
        .map_err(|_| StoredDicomError::InvalidTag {
            tag: "Rows/Columns/BitsAllocated/BitsStored",
        })?;
    if rows != expected_rows || columns != expected_columns {
        return Err(StoredDicomError::InconsistentPixelEncoding);
    }
    let calibration = match modality_lookup_table(object, pixel_representation)? {
        Some((table, unit)) => {
            if object.element(Tag(0x0028, 0x1052)).is_ok()
                || object.element(Tag(0x0028, 0x1053)).is_ok()
            {
                return Err(StoredDicomError::ConflictingCalibrationForms);
            }
            SliceCalibration::ModalityLookup { value: table, unit }
        }
        None => {
            let has_slope = object.element(Tag(0x0028, 0x1053)).is_ok();
            let has_intercept = object.element(Tag(0x0028, 0x1052)).is_ok();
            if has_slope != has_intercept {
                return Err(StoredDicomError::InvalidCalibration);
            }
            let has_rescale_type = object.element(Tag(0x0028, 0x1054)).is_ok();
            let unit = if has_intercept || has_rescale_type {
                let value =
                    required_long_text(object, Tag(0x0028, 0x1054), "RescaleType (0028,1054)")?;
                if value.trim().is_empty() {
                    return Err(StoredDicomError::InvalidCalibration);
                }
                Some(value)
            } else {
                None
            };
            let slope = optional_decimal(object, Tag(0x0028, 0x1053), 1.0)?;
            let intercept = optional_decimal(object, Tag(0x0028, 0x1052), 0.0)?;
            SliceCalibration::Linear {
                value: LinearCalibration::new(slope, intercept)?,
                unit,
            }
        }
    };
    let geometry = validate_slice_geometry(object)?;
    Ok((layout, calibration, geometry))
}

fn validate_slice_geometry(object: &DefaultDicomObject) -> Result<SliceGeometry, StoredDicomError> {
    let position = required_decimal_values(
        object,
        Tag(0x0020, 0x0032),
        "ImagePositionPatient (0020,0032)",
    )?;
    let orientation = required_decimal_values(
        object,
        Tag(0x0020, 0x0037),
        "ImageOrientationPatient (0020,0037)",
    )?;
    let pixel_spacing =
        required_decimal_values(object, Tag(0x0028, 0x0030), "PixelSpacing (0028,0030)")?;
    if pixel_spacing.into_iter().any(|value| value <= 0.0) {
        return Err(StoredDicomError::InvalidGeometry {
            field: "PixelSpacing (0028,0030) is not positive",
        });
    }
    if !direction_cosines_are_orthonormal(&orientation) {
        return Err(StoredDicomError::InvalidGeometry {
            field: "ImageOrientationPatient (0020,0037) is not orthonormal",
        });
    }
    let spacing_between_slices = optional_positive_decimal(
        object,
        Tag(0x0018, 0x0088),
        "SpacingBetweenSlices (0018,0088) is not positive",
    )?;
    let slice_thickness = optional_positive_decimal(
        object,
        Tag(0x0018, 0x0050),
        "SliceThickness (0018,0050) is not positive",
    )?;
    Ok(SliceGeometry {
        position,
        orientation,
        pixel_spacing,
        depth_spacing: spacing_between_slices.or(slice_thickness),
    })
}

fn optional_positive_decimal(
    object: &DefaultDicomObject,
    tag: Tag,
    field: &'static str,
) -> Result<Option<f64>, StoredDicomError> {
    let Ok(element) = object.element(tag) else {
        return Ok(None);
    };
    if element.vr() != VR::DS {
        return Err(StoredDicomError::InvalidTag { tag: field });
    }
    let text = element
        .to_str()
        .map_err(|_| StoredDicomError::InvalidTag { tag: field })?;
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let value = text
        .parse::<f64>()
        .map_err(|_| StoredDicomError::InvalidTag { tag: field })?;
    if !value.is_finite() || value <= 0.0 {
        return Err(StoredDicomError::InvalidGeometry { field });
    }
    Ok(Some(value))
}
