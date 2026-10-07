//! Lossless stored-sample import for a selected DICOM image series.

use std::path::Path;

use ritk_codecs::{
    decode_stored_pixel_frame, ByteOrder, PixelLayout, Sample, SampleBuffer, SampleType,
};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    IntensityCalibration, IntensityUnit, LinearCalibration, ModalityLookupTable, SeriesAxis,
    StoredSeries, StoredVolume,
};
use ritk_spatial::{CoordinateMap, Direction, Point, SliceSeries, SliceTransform, Spacing, Vector};

use super::scan::scan_dicom_path_with_budget;
use super::types::{DicomReadMetadata, DicomSeriesInfo, DicomSliceMetadata};
use super::DicomReadBudget;

/// Read one selected DICOM image series without changing its stored samples.
///
/// The result contains one [`StoredVolume`] and the DICOM metadata associated
/// with the selected series. Stored integer pixels remain in their declared
/// signed or unsigned type; modality rescale values and their unit label travel
/// beside the samples. Native implicit- and explicit-VR little-endian monochrome
/// scalar images are supported. Encapsulated data, color, and multi-frame
/// instances return typed errors. Modality lookup tables remain typed
/// calibration instead of passing through the calibrated `f32` reader.
///
/// # Errors
///
/// Returns [`StoredDicomError`] when scanning, pixel metadata, geometry,
/// calibration, resource limits, or stored-pixel decoding fails.
pub fn read_dicom_stored_series<P: AsRef<Path>>(
    path: P,
) -> Result<(StoredSeries, DicomReadMetadata), StoredDicomError> {
    read_dicom_stored_series_with_budget(path, &DicomReadBudget::DEFAULT)
}

/// Read a selected DICOM image series with explicit parser and memory budgets.
///
/// # Errors
///
/// Returns [`StoredDicomError`] for scan failures, unsupported encodings,
/// invalid metadata, or resource-budget violations.
pub fn read_dicom_stored_series_with_budget<P: AsRef<Path>>(
    path: P,
    budget: &DicomReadBudget,
) -> Result<(StoredSeries, DicomReadMetadata), StoredDicomError> {
    let series = scan_dicom_path_with_budget(path, budget).map_err(StoredDicomError::Scan)?;
    load_dicom_stored_series_with_budget(series, budget)
}

/// Load stored samples from a previously scanned DICOM series.
///
/// The descriptor must retain the validated Part 10 bytes produced by the
/// scanner. The returned metadata releases those bytes after decoding.
///
/// # Errors
///
/// Returns [`StoredDicomError`] when a descriptor lacks retained bytes, its
/// instances disagree on encoding, metadata is invalid, or decoding fails.
pub fn load_dicom_stored_series(
    series: DicomSeriesInfo,
) -> Result<(StoredSeries, DicomReadMetadata), StoredDicomError> {
    load_dicom_stored_series_with_budget(series, &DicomReadBudget::DEFAULT)
}

/// Load a previously scanned DICOM series with explicit parser and memory budgets.
///
/// # Errors
///
/// Returns [`StoredDicomError`] for unsupported encodings, invalid metadata,
/// missing retained instance bytes, or resource-budget violations.
pub fn load_dicom_stored_series_with_budget(
    mut series: DicomSeriesInfo,
    budget: &DicomReadBudget,
) -> Result<(StoredSeries, DicomReadMetadata), StoredDicomError> {
    let validated = preflight(&series, budget)?;
    let metadata = image_metadata(&validated)?;
    let samples = decode_volume_samples(&series.metadata.slices, &validated, budget)?;
    let ValidatedSeries {
        shape,
        calibration,
        intensity_unit,
        transforms,
        ..
    } = validated;
    let coordinate_map = slice_coordinate_map(transforms)?;
    let volume = StoredVolume::new(shape, samples, metadata, coordinate_map, calibration)?;
    let volume = match intensity_unit {
        Some(unit) => volume.with_intensity_unit(unit),
        None => volume,
    };
    for slice in &mut series.metadata.slices {
        slice.part10_bytes = None;
    }
    let stored_series = StoredSeries::new(vec![volume], SeriesAxis::SingleVolume)?;
    Ok((stored_series, series.metadata))
}

#[path = "stored/error.rs"]
mod error;
#[path = "stored/lut.rs"]
mod lut;
#[path = "stored/parse.rs"]
mod parse;
#[path = "stored/validation.rs"]
mod validation;

pub use error::StoredDicomError;
use parse::{
    parse_retained_instance, pixel_data, same_encoding, sample_type, trim_pixel_padding,
    validate_pixel_payload,
};
use validation::{validate_instance, SliceCalibration, SliceGeometry};

struct ValidatedSeries {
    shape: [usize; 3],
    layout: PixelLayout,
    calibration: IntensityCalibration,
    intensity_unit: Option<IntensityUnit>,
    transforms: Vec<SliceTransform>,
    first_geometry: SliceGeometry,
    sample_type: SampleType,
    frame_pixels: usize,
}

enum SeriesCalibration {
    Linear(Vec<LinearCalibration>),
    ModalityLookup(ModalityLookupTable),
}

fn preflight(
    series: &DicomSeriesInfo,
    budget: &DicomReadBudget,
) -> Result<ValidatedSeries, StoredDicomError> {
    let metadata = &series.metadata;
    let [rows, columns, depth] = metadata.dimensions;
    if rows == 0 || columns == 0 || depth == 0 || metadata.slices.len() != depth {
        return Err(StoredDicomError::InvalidGeometry {
            field: "series dimensions and slice count disagree",
        });
    }
    let first_slice = metadata
        .slices
        .first()
        .ok_or(StoredDicomError::InvalidGeometry {
            field: "series has no slices",
        })?;
    let first_object = parse_retained_instance(first_slice, budget)?;
    let (layout, first_calibration, first_geometry) =
        validate_instance(&first_object, rows, columns)?;
    let sample_type = sample_type(layout)?;
    let expected_frame_bytes =
        layout
            .bytes_per_frame()
            .map_err(|_| StoredDicomError::InvalidTag {
                tag: "Rows/Columns/BitsAllocated/BitsStored",
            })?;
    let frame_pixels = rows
        .checked_mul(columns)
        .ok_or(StoredDicomError::ShapeOverflow)?;
    let total_samples = frame_pixels
        .checked_mul(depth)
        .ok_or(StoredDicomError::ShapeOverflow)?;
    let output_bytes = total_samples
        .checked_mul(sample_type.byte_width())
        .ok_or(StoredDicomError::ShapeOverflow)?;
    let encoded_frame_bytes = frame_pixels
        .checked_mul(usize::from(layout.bits_allocated / 8))
        .ok_or(StoredDicomError::ShapeOverflow)?;
    let decoded_frame_bytes = frame_pixels
        .checked_mul(sample_type.byte_width())
        .ok_or(StoredDicomError::ShapeOverflow)?;
    let maximum_instance_bytes = metadata.slices.iter().fold(0, |maximum, slice| {
        maximum.max(slice.part10_bytes.as_ref().map_or(0, std::vec::Vec::len))
    });
    let transform_bytes = depth
        .checked_mul(std::mem::size_of::<SliceTransform>())
        .ok_or(StoredDicomError::ShapeOverflow)?;
    let calibration_bytes = match &first_calibration {
        SliceCalibration::Linear { .. } => depth
            .checked_mul(std::mem::size_of::<LinearCalibration>())
            .ok_or(StoredDicomError::ShapeOverflow)?,
        SliceCalibration::ModalityLookup { value, .. } => value
            .entries()
            .len()
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(StoredDicomError::ShapeOverflow)?,
    };
    let unit_bytes = first_calibration.unit_text().map_or(0, str::len);
    let peak_bytes = [
        output_bytes,
        maximum_instance_bytes,
        encoded_frame_bytes,
        decoded_frame_bytes,
        transform_bytes,
        calibration_bytes,
        std::mem::size_of::<PixelLayout>(),
        std::mem::size_of::<SliceGeometry>(),
        std::mem::size_of::<IntensityCalibration>(),
        std::mem::size_of::<Option<IntensityUnit>>(),
        unit_bytes,
    ]
    .into_iter()
    .try_fold(0_usize, usize::checked_add)
    .ok_or(StoredDicomError::ShapeOverflow)?;
    budget
        .checked_decoded_bytes(peak_bytes)
        .map_err(StoredDicomError::Budget)?;

    let first_pixels = pixel_data(&first_object, layout)?;
    validate_pixel_payload(&first_pixels, expected_frame_bytes)?;
    drop(first_object);

    let (mut calibration, unit_text) = match first_calibration {
        SliceCalibration::Linear { value, unit } => {
            let mut values = Vec::new();
            values
                .try_reserve_exact(depth)
                .map_err(StoredDicomError::Allocation)?;
            values.push(value);
            (SeriesCalibration::Linear(values), unit)
        }
        SliceCalibration::ModalityLookup { value, unit } => {
            (SeriesCalibration::ModalityLookup(value), Some(unit))
        }
    };
    let mut transforms = Vec::new();
    transforms
        .try_reserve_exact(depth)
        .map_err(StoredDicomError::Allocation)?;
    transforms.push(slice_transform(first_geometry)?);

    for slice in metadata.slices.iter().skip(1) {
        let object = parse_retained_instance(slice, budget)?;
        let (slice_layout, slice_calibration, slice_geometry) =
            validate_instance(&object, rows, columns)?;
        if !same_encoding(layout, slice_layout) {
            return Err(StoredDicomError::InconsistentPixelEncoding);
        }
        let pixels = pixel_data(&object, slice_layout)?;
        validate_pixel_payload(&pixels, expected_frame_bytes)?;
        if slice_calibration.unit_text() != unit_text.as_deref() {
            return Err(StoredDicomError::InconsistentCalibration);
        }
        match (&mut calibration, slice_calibration) {
            (SeriesCalibration::Linear(values), SliceCalibration::Linear { value, .. }) => {
                values.push(value)
            }
            (
                SeriesCalibration::ModalityLookup(existing),
                SliceCalibration::ModalityLookup { value, .. },
            ) if *existing == value => {}
            _ => return Err(StoredDicomError::InconsistentCalibration),
        }
        transforms.push(slice_transform(slice_geometry)?);
    }

    let calibration = match calibration {
        SeriesCalibration::Linear(values) => {
            IntensityCalibration::PerFrameLinear(values.into_boxed_slice())
        }
        SeriesCalibration::ModalityLookup(value) => IntensityCalibration::ModalityLookup(value),
    };
    let intensity_unit = unit_text
        .map(IntensityUnit::new)
        .transpose()
        .map_err(|_| StoredDicomError::InvalidCalibration)?;

    Ok(ValidatedSeries {
        shape: [depth, rows, columns],
        layout,
        calibration,
        intensity_unit,
        transforms,
        first_geometry,
        sample_type,
        frame_pixels,
    })
}

impl SliceCalibration {
    fn unit_text(&self) -> Option<&str> {
        match self {
            Self::Linear { unit, .. } => unit.as_deref(),
            Self::ModalityLookup { unit, .. } => Some(unit),
        }
    }
}

fn slice_transform(slice: SliceGeometry) -> Result<SliceTransform, StoredDicomError> {
    let row_direction = Vector::new([
        slice.orientation[0],
        slice.orientation[1],
        slice.orientation[2],
    ]);
    let column_direction = Vector::new([
        slice.orientation[3],
        slice.orientation[4],
        slice.orientation[5],
    ]);
    let normal = row_direction.cross(&column_direction).normalized().ok_or(
        StoredDicomError::InvalidGeometry {
            field: "ImageOrientationPatient axes are parallel or invalid",
        },
    )?;
    let pixel_column = row_direction * slice.pixel_spacing[1];
    let pixel_row = column_direction * slice.pixel_spacing[0];
    let direction = Direction::from_columns([pixel_column, pixel_row, normal]);
    Ok(SliceTransform::new(direction, slice.position))
}

fn decode_volume_samples(
    slices: &[DicomSliceMetadata],
    validated: &ValidatedSeries,
    budget: &DicomReadBudget,
) -> Result<SampleBuffer, StoredDicomError> {
    match validated.sample_type {
        SampleType::U8 => decode_typed::<u8>(slices, validated, budget),
        SampleType::I8 => decode_typed::<i8>(slices, validated, budget),
        SampleType::U16 => decode_typed::<u16>(slices, validated, budget),
        SampleType::I16 => decode_typed::<i16>(slices, validated, budget),
        SampleType::U32 => decode_typed::<u32>(slices, validated, budget),
        SampleType::I32 => decode_typed::<i32>(slices, validated, budget),
        SampleType::U64 | SampleType::I64 | SampleType::F32 | SampleType::F64 => {
            Err(StoredDicomError::InvalidTag {
                tag: "BitsAllocated/PixelRepresentation",
            })
        }
        _ => Err(StoredDicomError::InvalidTag {
            tag: "BitsAllocated/PixelRepresentation",
        }),
    }
}

fn decode_typed<T: Sample>(
    slices: &[DicomSliceMetadata],
    validated: &ValidatedSeries,
    budget: &DicomReadBudget,
) -> Result<SampleBuffer, StoredDicomError> {
    let sample_count = validated.shape[0]
        .checked_mul(validated.frame_pixels)
        .ok_or(StoredDicomError::ShapeOverflow)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(sample_count)
        .map_err(StoredDicomError::Allocation)?;
    for slice in slices {
        let object = parse_retained_instance(slice, budget)?;
        let bytes = pixel_data(&object, validated.layout)?;
        let expected =
            validated
                .layout
                .bytes_per_frame()
                .map_err(|_| StoredDicomError::InvalidTag {
                    tag: "Rows/Columns/BitsAllocated/BitsStored",
                })?;
        let mut frame_bytes = bytes.into_owned();
        trim_pixel_padding(&mut frame_bytes, expected)?;
        let frame = decode_stored_pixel_frame(
            &frame_bytes,
            validated.layout,
            ByteOrder::LeastSignificantByteFirst,
        )?;
        let frame_samples =
            frame
                .try_into_samples::<T>()
                .map_err(|_| StoredDicomError::InvalidTag {
                    tag: "BitsAllocated/PixelRepresentation",
                })?;
        samples.extend(frame_samples);
    }
    Ok(SampleBuffer::from_samples(samples))
}

fn slice_coordinate_map(
    transforms: Vec<SliceTransform>,
) -> Result<CoordinateMap, StoredDicomError> {
    let series = SliceSeries::try_new(transforms).map_err(ritk_image_io::VolumeError::from)?;
    Ok(CoordinateMap::SliceSeries(series))
}

fn image_metadata(validated: &ValidatedSeries) -> Result<ImageMetadata<3>, StoredDicomError> {
    let geometry = validated.first_geometry;
    let slice_spacing = match validated.transforms.as_slice() {
        [first, second, ..] => {
            let [_, _, normal] = first.rotation().axis_directions_array();
            let first_position = first.translation();
            let second_position = second.translation();
            let displacement = Vector::new([
                second_position[0] - first_position[0],
                second_position[1] - first_position[1],
                second_position[2] - first_position[2],
            ]);
            let spacing = displacement.dot(&normal).abs();
            if !spacing.is_finite() || spacing <= 0.0 {
                return Err(StoredDicomError::InvalidGeometry {
                    field: "raw slice positions do not define positive normal spacing",
                });
            }
            spacing
        }
        [_] => geometry
            .depth_spacing
            .ok_or(StoredDicomError::InvalidGeometry {
                field: "single-slice series requires raw SpacingBetweenSlices or SliceThickness",
            })?,
        [] => {
            return Err(StoredDicomError::InvalidGeometry {
                field: "series has no slice transforms",
            });
        }
    };
    let spacing = Spacing::try_new([
        slice_spacing,
        geometry.pixel_spacing[0],
        geometry.pixel_spacing[1],
    ])
    .map_err(|_| StoredDicomError::InvalidGeometry {
        field: "series spacing is invalid",
    })?;
    let row_direction = Vector::new([
        geometry.orientation[0],
        geometry.orientation[1],
        geometry.orientation[2],
    ]);
    let column_direction = Vector::new([
        geometry.orientation[3],
        geometry.orientation[4],
        geometry.orientation[5],
    ]);
    let normal = row_direction.cross(&column_direction).normalized().ok_or(
        StoredDicomError::InvalidGeometry {
            field: "ImageOrientationPatient axes are parallel or invalid",
        },
    )?;
    let direction = Direction::from_columns([normal, column_direction, row_direction]);
    Ok(ImageMetadata::new(
        Point::new(geometry.position),
        spacing,
        direction,
    ))
}
