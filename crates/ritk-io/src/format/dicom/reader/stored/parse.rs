use dicom::core::{Tag, VR};
use dicom::object::DefaultDicomObject;
use ritk_codecs::{PixelLayout, PixelSignedness, SampleType};
use ritk_dicom::{parse_bytes_with_budget, DicomRsBackend};

use super::super::types::DicomSliceMetadata;
use super::super::DicomReadBudget;
use super::StoredDicomError;

pub(super) fn parse_retained_instance(
    slice: &DicomSliceMetadata,
    budget: &DicomReadBudget,
) -> Result<DefaultDicomObject, StoredDicomError> {
    let bytes = slice
        .part10_bytes
        .as_deref()
        .ok_or(StoredDicomError::MissingRetainedBytes)?;
    parse_bytes_with_budget::<DicomRsBackend>(bytes, &budget.parser())
        .map_err(StoredDicomError::Parse)
}

pub(super) fn pixel_data<'a>(
    object: &'a DefaultDicomObject,
    layout: PixelLayout,
) -> Result<std::borrow::Cow<'a, [u8]>, StoredDicomError> {
    let element =
        object
            .element(Tag(0x7fe0, 0x0010))
            .map_err(|_| StoredDicomError::MissingTag {
                tag: "PixelData (7FE0,0010)",
            })?;
    let bytes = element
        .to_bytes()
        .map_err(|source| StoredDicomError::PixelValue(anyhow::Error::from(source)))?;
    let expected = layout
        .bytes_per_frame()
        .map_err(|_| StoredDicomError::InvalidTag {
            tag: "Rows/Columns/BitsAllocated/BitsStored",
        })?;
    validate_pixel_payload(&bytes, expected)?;
    Ok(bytes)
}

pub(super) fn validate_pixel_payload(
    bytes: &[u8],
    expected: usize,
) -> Result<(), StoredDicomError> {
    validate_pixel_length(bytes.len(), expected)?;
    if expected.checked_add(1) == Some(bytes.len()) && bytes.last() != Some(&0) {
        return Err(StoredDicomError::PixelDataLength {
            actual: bytes.len(),
            expected,
        });
    }
    Ok(())
}

pub(super) fn validate_pixel_length(
    actual: usize,
    expected: usize,
) -> Result<(), StoredDicomError> {
    if actual == expected
        || (expected % 2 == 1
            && expected
                .checked_add(1)
                .is_some_and(|padded| actual == padded))
    {
        return Ok(());
    }
    Err(StoredDicomError::PixelDataLength { actual, expected })
}

pub(super) fn trim_pixel_padding(
    bytes: &mut Vec<u8>,
    expected: usize,
) -> Result<(), StoredDicomError> {
    validate_pixel_length(bytes.len(), expected)?;
    if expected.checked_add(1) == Some(bytes.len()) && bytes.last() == Some(&0) {
        bytes.truncate(expected);
        return Ok(());
    }
    if bytes.len() != expected {
        return Err(StoredDicomError::PixelDataLength {
            actual: bytes.len(),
            expected,
        });
    }
    Ok(())
}

pub(super) fn required_u16(
    object: &DefaultDicomObject,
    tag: Tag,
    name: &'static str,
) -> Result<u16, StoredDicomError> {
    object
        .element(tag)
        .map_err(|_| StoredDicomError::MissingTag { tag: name })?
        .to_int::<u16>()
        .map_err(|_| StoredDicomError::InvalidTag { tag: name })
}

pub(super) fn required_usize(
    object: &DefaultDicomObject,
    tag: Tag,
    name: &'static str,
) -> Result<usize, StoredDicomError> {
    object
        .element(tag)
        .map_err(|_| StoredDicomError::MissingTag { tag: name })?
        .to_int::<usize>()
        .map_err(|_| StoredDicomError::InvalidTag { tag: name })
}

pub(super) fn required_text(
    object: &DefaultDicomObject,
    tag: Tag,
    name: &'static str,
) -> Result<String, StoredDicomError> {
    object
        .element(tag)
        .map_err(|_| StoredDicomError::MissingTag { tag: name })?
        .to_str()
        .map(|value| value.into_owned())
        .map_err(|_| StoredDicomError::InvalidTag { tag: name })
}

pub(super) fn required_long_text(
    object: &DefaultDicomObject,
    tag: Tag,
    name: &'static str,
) -> Result<String, StoredDicomError> {
    let element = object
        .element(tag)
        .map_err(|_| StoredDicomError::MissingTag { tag: name })?;
    if element.vr() != VR::LO {
        return Err(StoredDicomError::InvalidTag { tag: name });
    }
    element
        .to_str()
        .map(|value| value.into_owned())
        .map_err(|_| StoredDicomError::InvalidTag { tag: name })
}

pub(super) fn required_decimal_values<const N: usize>(
    object: &DefaultDicomObject,
    tag: Tag,
    name: &'static str,
) -> Result<[f64; N], StoredDicomError> {
    let element = object
        .element(tag)
        .map_err(|_| StoredDicomError::InvalidGeometry { field: name })?;
    if element.vr() != VR::DS {
        return Err(StoredDicomError::InvalidTag { tag: name });
    }
    let value = element
        .to_str()
        .map_err(|_| StoredDicomError::InvalidTag { tag: name })?;
    let mut components = value.split('\\');
    let mut values = [0.0; N];
    for component in &mut values {
        let text = components
            .next()
            .ok_or(StoredDicomError::InvalidTag { tag: name })?;
        *component = text
            .trim()
            .parse::<f64>()
            .map_err(|_| StoredDicomError::InvalidTag { tag: name })?;
        if !component.is_finite() {
            return Err(StoredDicomError::InvalidGeometry { field: name });
        }
    }
    if components.next().is_some() {
        return Err(StoredDicomError::InvalidTag { tag: name });
    }
    Ok(values)
}

pub(super) fn optional_text(
    object: &DefaultDicomObject,
    tag: Tag,
) -> Result<Option<String>, StoredDicomError> {
    match object.element(tag) {
        Ok(element) => element
            .to_str()
            .map(|value| Some(value.into_owned()))
            .map_err(|_| StoredDicomError::InvalidTag {
                tag: "NumberOfFrames (0028,0008)",
            }),
        Err(_) => Ok(None),
    }
}

pub(super) fn optional_decimal(
    object: &DefaultDicomObject,
    tag: Tag,
    default: f64,
) -> Result<f64, StoredDicomError> {
    let Some(value) = optional_text_for_tag(object, tag)? else {
        return Ok(default);
    };
    let parsed = value
        .trim()
        .parse::<f64>()
        .map_err(|_| StoredDicomError::InvalidCalibration)?;
    if !parsed.is_finite() {
        return Err(StoredDicomError::InvalidCalibration);
    }
    Ok(parsed)
}

pub(super) fn optional_text_for_tag(
    object: &DefaultDicomObject,
    tag: Tag,
) -> Result<Option<String>, StoredDicomError> {
    match object.element(tag) {
        Ok(element) => element
            .to_str()
            .map(|value| Some(value.into_owned()))
            .map_err(|_| StoredDicomError::InvalidCalibration),
        Err(_) => Ok(None),
    }
}

pub(super) fn same_encoding(left: PixelLayout, right: PixelLayout) -> bool {
    left.rows == right.rows
        && left.cols == right.cols
        && left.samples_per_pixel == right.samples_per_pixel
        && left.bits_allocated == right.bits_allocated
        && left.bits_stored == right.bits_stored
        && left.pixel_representation == right.pixel_representation
}

pub(super) fn sample_type(layout: PixelLayout) -> Result<SampleType, StoredDicomError> {
    match (layout.bits_allocated, layout.pixel_representation) {
        (8, PixelSignedness::Unsigned) => Ok(SampleType::U8),
        (8, PixelSignedness::Signed) => Ok(SampleType::I8),
        (16, PixelSignedness::Unsigned) => Ok(SampleType::U16),
        (16, PixelSignedness::Signed) => Ok(SampleType::I16),
        (24 | 32, PixelSignedness::Unsigned) => Ok(SampleType::U32),
        (24 | 32, PixelSignedness::Signed) => Ok(SampleType::I32),
        _ => Err(StoredDicomError::InvalidTag {
            tag: "BitsAllocated/PixelRepresentation",
        }),
    }
}
