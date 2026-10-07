//! Preflight native DICOM Pixel Data against its Image Pixel Module.

use super::error::DicomWriteError;
use anyhow::Result;
use dicom::core::value::{DicomValueType, PrimitiveValue};
use dicom::core::{Tag, VR};
use dicom::object::InMemDicomObject;

const PIXEL_DATA: Tag = Tag(0x7FE0, 0x0010);
const ROWS: Tag = Tag(0x0028, 0x0010);
const COLUMNS: Tag = Tag(0x0028, 0x0011);
const SAMPLES_PER_PIXEL: Tag = Tag(0x0028, 0x0002);
const PHOTOMETRIC_INTERPRETATION: Tag = Tag(0x0028, 0x0004);
const PLANAR_CONFIGURATION: Tag = Tag(0x0028, 0x0006);
const NUMBER_OF_FRAMES: Tag = Tag(0x0028, 0x0008);
const BITS_ALLOCATED: Tag = Tag(0x0028, 0x0100);
const BITS_STORED: Tag = Tag(0x0028, 0x0101);
const HIGH_BIT: Tag = Tag(0x0028, 0x0102);
const PIXEL_REPRESENTATION: Tag = Tag(0x0028, 0x0103);

// PS3.5 A.2 caps Explicit VR Little Endian value fields at 2^32 - 2 bytes.
const MAX_NATIVE_PIXEL_VALUE_LENGTH: usize = 0xFFFF_FFFE;

/// Whether the three Image Pixel Module bit attributes describe a valid layout.
///
/// DICOM PS3.3 C.7.6.3 requires BitsAllocated to be 1 or a multiple of 8,
/// BitsStored to lie in `1..=BitsAllocated`, and HighBit to equal
/// `BitsStored - 1`. Both the source-metadata preflight and the object
/// preflight enforce this rule, so it has a single definition here.
#[must_use]
pub(crate) fn pixel_bit_description_is_valid(
    bits_allocated: u16,
    bits_stored: u16,
    high_bit: u16,
) -> bool {
    (bits_allocated == 1 || bits_allocated.is_multiple_of(8))
        && bits_stored != 0
        && bits_stored <= bits_allocated
        && high_bit.checked_add(1) == Some(bits_stored)
}

/// Validate a native Pixel Data value against its Image Pixel Module.
///
/// The pixel count is derived from rows, columns, samples, and frames with
/// checked arithmetic. DICOM pads the complete value once to even length; it
/// does not add padding between native frames. YBR_FULL_422 and the retired
/// YBR_PARTIAL_422 use the same two luminance bytes followed by one Cb and one
/// Cr byte for each pair of pixels. The
/// attribute rules follow [PS3.3 C.7.6.3]; VR encoding follows [PS3.5 6.2],
/// and native pixel encoding follows [PS3.5 8.2] and [PS3.5 A.2]. The retired
/// interpretation's historical subsampled layout is specified in [PS3.3 2016d C.7.6.3].
///
/// [PS3.3 C.7.6.3]: https://dicom.nema.org/medical/DICOM/current/output/chtml/part03/sect_C.7.6.3.html
/// [PS3.3 2016d C.7.6.3]: https://dicom.nema.org/medical/dicom/2016d/output/chtml/part03/sect_C.7.6.3.html
/// [PS3.5 6.2]: https://dicom.nema.org/medical/DICOM/current/output/chtml/part05/sect_6.2.html
/// [PS3.5 8.2]: https://dicom.nema.org/medical/DICOM/current/output/chtml/part05/sect_8.2.html
/// [PS3.5 A.2]: https://dicom.nema.org/medical/DICOM/current/output/chtml/part05/sect_A.2.html
pub(crate) fn preflight_native_pixel_data(object: &InMemDicomObject) -> Result<()> {
    let Some(pixel_data) = object.get(PIXEL_DATA) else {
        return Ok(());
    };

    let rows = required_tag_unsigned(object, ROWS, "Rows")?;
    let columns = required_tag_unsigned(object, COLUMNS, "Columns")?;
    let samples_per_pixel = required_tag_unsigned(object, SAMPLES_PER_PIXEL, "SamplesPerPixel")?;
    let bits_allocated = required_tag_unsigned(object, BITS_ALLOCATED, "BitsAllocated")?;
    let bits_stored = required_tag_unsigned(object, BITS_STORED, "BitsStored")?;
    let high_bit = required_tag_unsigned(object, HIGH_BIT, "HighBit")?;
    let pixel_representation =
        required_tag_unsigned(object, PIXEL_REPRESENTATION, "PixelRepresentation")?;
    let frames = match object.get(NUMBER_OF_FRAMES) {
        Some(element) => required_number_of_frames(element)?,
        None => 1,
    };
    let photometric = required_single_cs(
        object,
        PHOTOMETRIC_INTERPRETATION,
        "PhotometricInterpretation",
    )?;

    if rows == 0 || columns == 0 || frames == 0 {
        return Err(DicomWriteError::InvalidDimensions {
            depth: frames,
            rows: usize::from(rows),
            columns: usize::from(columns),
        }
        .into());
    }
    if samples_per_pixel == 0 {
        return Err(malformed_pixel_attribute("SamplesPerPixel", "zero samples".to_owned()).into());
    }
    if !pixel_bit_description_is_valid(bits_allocated, bits_stored, high_bit) {
        return Err(DicomWriteError::InvalidPixelDescription {
            bits_allocated,
            bits_stored,
            high_bit,
        }
        .into());
    }
    if pixel_representation > 1 {
        return Err(malformed_pixel_attribute(
            "PixelRepresentation",
            pixel_representation.to_string(),
        )
        .into());
    }

    let expected_samples: Option<u16> = match photometric.as_str() {
        "MONOCHROME1" | "MONOCHROME2" | "PALETTE COLOR" => Some(1),
        "RGB" | "HSV" | "YBR_FULL" | "YBR_FULL_422" | "YBR_PARTIAL_422" => Some(3),
        // These retired interpretations remain readable in older instances.
        "ARGB" | "CMYK" => Some(4),
        // These encodings require encapsulated compression, not the native
        // Explicit VR Little Endian transfer syntax used by these writers.
        "YBR_RCT" | "YBR_ICT" | "YBR_PARTIAL_420" => {
            return Err(malformed_pixel_attribute(
                "PhotometricInterpretation",
                format!("{photometric} requires encapsulated pixel encoding"),
            )
            .into());
        }
        // PS3.3 permits other values when the transfer syntax supports them.
        // Their declared SamplesPerPixel remains the available length oracle.
        _ => None,
    };
    if let Some(expected_samples) = expected_samples
        && samples_per_pixel != expected_samples
    {
        return Err(malformed_pixel_attribute(
            "SamplesPerPixel",
            format!(
                "{} requires {expected_samples}, got {samples_per_pixel}",
                photometric
            ),
        )
        .into());
    }

    let planar_configuration = match object.get(PLANAR_CONFIGURATION) {
        Some(element) => Some(single_us(element, "PlanarConfiguration")?),
        None => None,
    };
    if samples_per_pixel == 1 {
        if planar_configuration.is_some() {
            return Err(malformed_pixel_attribute(
                "PlanarConfiguration",
                "must be absent when SamplesPerPixel is one".to_owned(),
            )
            .into());
        }
    } else if !matches!(planar_configuration, Some(0 | 1)) {
        return Err(malformed_pixel_attribute(
            "PlanarConfiguration",
            planar_configuration.map_or_else(|| "missing".to_owned(), |value| value.to_string()),
        )
        .into());
    }

    let is_ybr_422 = matches!(photometric.as_str(), "YBR_FULL_422" | "YBR_PARTIAL_422");
    if is_ybr_422
        && (bits_allocated != 8
            || pixel_representation != 0
            || planar_configuration != Some(0)
            || !columns.is_multiple_of(2))
    {
        return Err(malformed_pixel_attribute(
            "PhotometricInterpretation",
            "YBR 4:2:2 requires unsigned 8-bit samples, PlanarConfiguration 0, and an even column count".to_owned(),
        )
        .into());
    }

    let pixel_vr = pixel_data.vr();
    if !matches!(pixel_vr, VR::OB | VR::OW) || (bits_allocated > 8 && pixel_vr != VR::OW) {
        return Err(DicomWriteError::InvalidPixelDataVr {
            vr: format!("{pixel_vr}"),
            bits_allocated,
        }
        .into());
    }

    let primitive = pixel_data.value().primitive().ok_or_else(|| {
        DicomWriteError::UnsupportedPixelPayloadValue {
            value_type: format!("{:?}", pixel_data.value().value_type()),
        }
    })?;
    let actual_bytes = primitive_pixel_bytes(primitive)?;
    let rows = usize::from(rows);
    let columns = usize::from(columns);
    let samples_per_pixel = usize::from(samples_per_pixel);
    let pixels_per_frame = rows
        .checked_mul(columns)
        .ok_or(DicomWriteError::PixelCountOverflow)?;
    let total_pixels = pixels_per_frame
        .checked_mul(frames)
        .ok_or(DicomWriteError::PixelCountOverflow)?;
    let expected_bytes = if is_ybr_422 {
        rows.checked_mul(frames)
            .and_then(|row_frames| row_frames.checked_mul(columns / 2))
            .and_then(|pairs| pairs.checked_mul(4))
            .ok_or(DicomWriteError::PixelCountOverflow)?
    } else {
        let samples = total_pixels
            .checked_mul(samples_per_pixel)
            .ok_or(DicomWriteError::PixelCountOverflow)?;
        if bits_allocated == 1 {
            samples.div_ceil(8)
        } else {
            samples
                .checked_mul(usize::from(bits_allocated / 8))
                .ok_or(DicomWriteError::PixelCountOverflow)?
        }
    };
    let padded_bytes = expected_bytes
        .checked_add(expected_bytes % 2)
        .ok_or(DicomWriteError::PixelCountOverflow)?;
    if padded_bytes > MAX_NATIVE_PIXEL_VALUE_LENGTH || actual_bytes > MAX_NATIVE_PIXEL_VALUE_LENGTH
    {
        return Err(malformed_pixel_attribute(
            "PixelData length",
            format!("exceeds Explicit VR Little Endian maximum {MAX_NATIVE_PIXEL_VALUE_LENGTH}"),
        )
        .into());
    }
    if actual_bytes != expected_bytes && actual_bytes != padded_bytes {
        return Err(DicomWriteError::PixelPayloadLengthMismatch {
            expected: expected_bytes,
            actual: actual_bytes,
        }
        .into());
    }
    if actual_bytes == padded_bytes && padded_bytes != expected_bytes {
        let value = last_explicit_vr_le_byte(primitive).ok_or(
            DicomWriteError::PixelPayloadLengthMismatch {
                expected: expected_bytes,
                actual: actual_bytes,
            },
        )?;
        if value != 0 {
            return Err(DicomWriteError::InvalidPixelDataPadding { value }.into());
        }
    }

    Ok(())
}

fn required_tag_unsigned(
    object: &InMemDicomObject,
    tag: Tag,
    attribute: &'static str,
) -> Result<u16> {
    let element = object
        .get(tag)
        .ok_or(DicomWriteError::MissingPixelAttribute { attribute })?;
    single_us(element, attribute)
}

fn single_us(
    element: &dicom::core::DataElement<InMemDicomObject>,
    attribute: &'static str,
) -> Result<u16> {
    if element.vr() != VR::US {
        return Err(malformed_pixel_attribute(
            attribute,
            format!("expected VR US, found {}", element.vr()),
        )
        .into());
    }
    match element.value().primitive() {
        Some(PrimitiveValue::U16(values)) => match values.as_slice() {
            [value] => Ok(*value),
            values => Err(malformed_multiplicity(attribute, values.len()).into()),
        },
        Some(value) => Err(malformed_pixel_attribute(
            attribute,
            format!("expected one U16 value, found {:?}", value.value_type()),
        )
        .into()),
        None => Err(malformed_pixel_attribute(
            attribute,
            "expected a primitive US value".to_owned(),
        )
        .into()),
    }
}

fn required_single_cs(
    object: &InMemDicomObject,
    tag: Tag,
    attribute: &'static str,
) -> Result<String> {
    let element = object
        .get(tag)
        .ok_or(DicomWriteError::MissingPixelAttribute { attribute })?;
    if element.vr() != VR::CS {
        return Err(malformed_pixel_attribute(
            attribute,
            format!("expected VR CS, found {}", element.vr()),
        )
        .into());
    }
    let value = match element.value().primitive() {
        Some(PrimitiveValue::Str(value)) => value.as_str(),
        Some(PrimitiveValue::Strs(values)) => match values.as_slice() {
            [value] => value.as_str(),
            values => return Err(malformed_multiplicity(attribute, values.len()).into()),
        },
        Some(value) => {
            return Err(malformed_pixel_attribute(
                attribute,
                format!("expected one CS value, found {:?}", value.value_type()),
            )
            .into());
        }
        None => {
            return Err(malformed_pixel_attribute(
                attribute,
                "expected a primitive CS value".to_owned(),
            )
            .into());
        }
    };
    if value.len() > 16
        || value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_uppercase() || byte.is_ascii_digit() || matches!(byte, b' ' | b'_')
        })
    {
        return Err(
            malformed_pixel_attribute(attribute, format!("invalid CS value {value:?}")).into(),
        );
    }
    let value = value.trim_matches(' ');
    if value.is_empty() {
        return Err(malformed_pixel_attribute(attribute, "empty CS value".to_owned()).into());
    }
    Ok(value.to_owned())
}

fn required_number_of_frames(
    element: &dicom::core::DataElement<InMemDicomObject>,
) -> Result<usize> {
    if element.vr() != VR::IS {
        return Err(malformed_pixel_attribute(
            "NumberOfFrames",
            format!("expected VR IS, found {}", element.vr()),
        )
        .into());
    }
    let value = match element.value().primitive() {
        Some(PrimitiveValue::I32(values)) => match values.as_slice() {
            [value] => *value,
            values => return Err(malformed_multiplicity("NumberOfFrames", values.len()).into()),
        },
        Some(PrimitiveValue::Str(value)) => parse_frame_count(value)?,
        Some(PrimitiveValue::Strs(values)) => match values.as_slice() {
            [value] => parse_frame_count(value)?,
            values => return Err(malformed_multiplicity("NumberOfFrames", values.len()).into()),
        },
        Some(value) => {
            return Err(malformed_pixel_attribute(
                "NumberOfFrames",
                format!("expected one IS value, found {:?}", value.value_type()),
            )
            .into());
        }
        None => {
            return Err(malformed_pixel_attribute(
                "NumberOfFrames",
                "expected a primitive IS value".to_owned(),
            )
            .into());
        }
    };
    usize::try_from(value).map_err(|_| {
        malformed_pixel_attribute(
            "NumberOfFrames",
            format!("frame count must be positive, found {value}"),
        )
        .into()
    })
}

fn parse_frame_count(value: &str) -> Result<i32> {
    if value.len() > 12
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b' '))
    {
        return Err(malformed_pixel_attribute(
            "NumberOfFrames",
            "invalid IS characters or value exceeds 12 bytes".to_owned(),
        )
        .into());
    }
    value
        .trim_matches(' ')
        .parse::<i32>()
        .map_err(|error| malformed_pixel_attribute("NumberOfFrames", error.to_string()).into())
}

fn malformed_pixel_attribute(attribute: &'static str, value: String) -> DicomWriteError {
    DicomWriteError::MalformedPixelAttribute { attribute, value }
}

fn malformed_multiplicity(attribute: &'static str, actual: usize) -> DicomWriteError {
    malformed_pixel_attribute(attribute, format!("expected VM 1, found {actual} values"))
}

fn primitive_pixel_bytes(value: &PrimitiveValue) -> Result<usize> {
    let (count, width) = match value {
        PrimitiveValue::Empty => return Ok(0),
        PrimitiveValue::U8(values) => (values.len(), 1),
        PrimitiveValue::I16(values) => (values.len(), 2),
        PrimitiveValue::U16(values) => (values.len(), 2),
        PrimitiveValue::I32(values) => (values.len(), 4),
        PrimitiveValue::U32(values) => (values.len(), 4),
        PrimitiveValue::I64(values) => (values.len(), 8),
        PrimitiveValue::U64(values) => (values.len(), 8),
        _ => {
            return Err(DicomWriteError::UnsupportedPixelPayloadValue {
                value_type: format!("{:?}", value.value_type()),
            }
            .into());
        }
    };
    count
        .checked_mul(width)
        .ok_or_else(|| DicomWriteError::PixelCountOverflow.into())
}

fn last_explicit_vr_le_byte(value: &PrimitiveValue) -> Option<u8> {
    match value {
        PrimitiveValue::U8(values) => values.last().copied(),
        PrimitiveValue::I16(values) => values
            .last()
            .and_then(|value| value.to_le_bytes().last().copied()),
        PrimitiveValue::U16(values) => values
            .last()
            .and_then(|value| value.to_le_bytes().last().copied()),
        PrimitiveValue::I32(values) => values
            .last()
            .and_then(|value| value.to_le_bytes().last().copied()),
        PrimitiveValue::U32(values) => values
            .last()
            .and_then(|value| value.to_le_bytes().last().copied()),
        PrimitiveValue::I64(values) => values
            .last()
            .and_then(|value| value.to_le_bytes().last().copied()),
        PrimitiveValue::U64(values) => values
            .last()
            .and_then(|value| value.to_le_bytes().last().copied()),
        _ => None,
    }
}

#[cfg(test)]
#[path = "pixel_preflight_tests.rs"]
mod tests;
