//! Exact stored-sample reads through the shared RITK image-I/O contract.

mod volumes;

use ritk_codecs::SampleError;
use ritk_image_io::{
    ImageReadBudget, ImageReadBudgetError, SeriesAxis, StoredSeries, StoredSeriesError,
    StoredVolume, VolumeError,
};
use std::collections::TryReserveError;
use std::io;
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use thiserror::Error;

use super::diffusion::scheme_from_header;
use super::header::{NrrdHeader, NrrdHeaderError};
use super::volume::{parse_nrrd_raw, NrrdReadPurpose};
use crate::axes::AcquisitionAxis;

/// A NRRD stored-sample read failed at a format, payload, allocation, or value boundary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NrrdStoredReadError {
    /// A format payload or decoded image exceeded its configured resource limit.
    #[error(transparent)]
    ReadBudget {
        /// Resource bound rejected before payload allocation.
        #[from]
        source: ImageReadBudgetError,
    },
    /// The header could not be parsed as a NRRD header.
    #[error(transparent)]
    HeaderParse {
        /// Header syntax, input, or resource-limit failure.
        #[from]
        source: NrrdHeaderError,
    },
    /// The header file could not be opened.
    #[error("cannot open NRRD file {path:?}: {source}")]
    OpenHeader {
        /// Header path supplied by the caller.
        path: PathBuf,
        /// Filesystem failure.
        #[source]
        source: io::Error,
    },
    /// A required header field is absent.
    #[error("NRRD header is missing required field {field:?}")]
    MissingHeaderField {
        /// Header field name.
        field: &'static str,
    },
    /// The array-rank field is not an integer.
    #[error("NRRD dimension {value:?} is not a valid integer: {source}")]
    InvalidDimension {
        /// Text supplied in the dimension field.
        value: String,
        /// Integer parse failure.
        #[source]
        source: ParseIntError,
    },
    /// A line-skip or byte-skip field is not an integer.
    #[error("NRRD {field} value {value:?} is not a valid integer: {source}")]
    InvalidSkipField {
        /// Field name from the header.
        field: &'static str,
        /// Text supplied in the field.
        value: String,
        /// Integer parse failure.
        #[source]
        source: ParseIntError,
    },
    /// The line-skip field must be nonnegative.
    #[error("NRRD line skip cannot be negative: {value}")]
    NegativeLineSkip {
        /// Parsed line count.
        value: i32,
    },
    /// The byte-skip field is invalid for its declared encoding.
    #[error("NRRD byte skip {value} is invalid: {reason}")]
    InvalidByteSkip {
        /// Parsed byte count.
        value: i32,
        /// Constraint violated by the value.
        reason: &'static str,
    },
    /// The NRRD array rank is outside the implemented range.
    #[error("NRRD array dimension {dimension} is unsupported; expected 2, 3, or 4")]
    UnsupportedDimension {
        /// Array rank from the header.
        dimension: usize,
    },
    /// The sizes field is malformed.
    #[error("NRRD sizes field {value:?} is invalid: {source}")]
    InvalidSizes {
        /// Text supplied in the sizes field.
        value: String,
        /// Parsing failure from the NRRD field parser.
        #[source]
        source: anyhow::Error,
    },
    /// One array axis has no elements.
    #[error("NRRD sizes entry for array axis {axis} is zero")]
    EmptyAxis {
        /// Zero-based array axis.
        axis: usize,
    },
    /// The header does not identify one supported acquisition axis.
    #[error("NRRD acquisition axis is invalid: {source}")]
    InvalidAcquisitionAxis {
        /// Axis parser failure.
        #[source]
        source: anyhow::Error,
    },
    /// The requested encoding is not implemented.
    #[error("NRRD encoding {encoding:?} is unsupported")]
    UnsupportedEncoding {
        /// Encoding field value.
        encoding: String,
    },
    /// The declared sample type has no exact stored codec.
    #[error("NRRD element type {element_type:?} is unsupported")]
    UnsupportedElementType {
        /// Element type from the header.
        element_type: String,
    },
    /// The byte-order field is present but invalid.
    #[error("NRRD endian marker {endian:?} is invalid")]
    InvalidByteOrder {
        /// Endian field value.
        endian: String,
    },
    /// A multi-byte payload omits its required byte order.
    #[error("NRRD endian field is required for {sample_width}-byte samples")]
    MissingByteOrder {
        /// Width of the declared sample type.
        sample_width: usize,
    },
    /// The declared coordinate system or spatial metadata is unsupported or invalid.
    #[error("NRRD {field} metadata is invalid or unsupported: {source}")]
    SpatialMetadata {
        /// NRRD spatial field whose value could not be represented.
        field: NrrdSpatialMetadataField,
        /// Parsing or geometry failure.
        #[source]
        source: anyhow::Error,
    },
    /// Both NRRD spatial direction and scalar-spacing representations are present.
    #[error("NRRD header contains both `space directions` and `spacings`")]
    ConflictingSpatialFields,
    /// Sample-value units are not represented by the stored-volume contract.
    #[error("NRRD sample units {units:?} are not represented by StoredVolume")]
    UnsupportedSampleUnits {
        /// Declared units for each stored scalar value.
        units: String,
    },
    /// A measurement frame is not represented outside a diffusion scheme.
    #[error("NRRD measurement frame {measurement_frame:?} is not represented by the stored-volume contract")]
    UnsupportedMeasurementFrame {
        /// Declared frame vectors from the NRRD header.
        measurement_frame: String,
    },
    /// The decoded voxel count overflows `usize`.
    #[error("NRRD voxel count overflows for sizes {sizes:?}")]
    VoxelCountOverflow {
        /// Spatial sizes in NRRD X, Y, Z order.
        sizes: [usize; 3],
    },
    /// The acquisition element count overflows `usize`.
    #[error("NRRD acquisition element count overflows usize")]
    SeriesCountOverflow,
    /// The declared payload byte count overflows `usize`.
    #[error("NRRD payload byte count overflows for {voxel_count} voxels of {sample_width} bytes")]
    PayloadByteCountOverflow {
        /// Number of declared voxels including acquisition volumes.
        voxel_count: usize,
        /// Width of one stored sample.
        sample_width: usize,
    },
    /// The decoded output byte count overflows `usize`.
    #[error("NRRD decoded byte count overflows for {voxel_count} voxels of {sample_width} bytes")]
    DecodedByteCountOverflow {
        /// Number of voxels including acquisition volumes.
        voxel_count: usize,
        /// Width of one output sample.
        sample_width: usize,
    },
    /// The decoded byte count cannot be compared with the shared u64 budget.
    #[error("NRRD decoded byte count {decoded_bytes} cannot be represented by the reader budget")]
    DecodedByteCountNotRepresentable {
        /// Decoded sample bytes required by the output representation.
        decoded_bytes: usize,
    },
    /// A gzip skip plus decoded payload exceeds the byte-count representation.
    #[error(
        "NRRD expanded payload byte count overflows for {payload_bytes} payload bytes and {skipped_bytes} skipped bytes"
    )]
    ExpandedPayloadByteCountOverflow {
        /// Bytes declared for the decoded payload.
        payload_bytes: u64,
        /// Bytes expanded and discarded before the declared payload.
        skipped_bytes: u64,
    },
    /// The coordinate-map field cannot be represented.
    #[error("NRRD coordinate map is invalid: {source}")]
    CoordinateMap {
        /// Coordinate-map parsing failure.
        #[source]
        source: anyhow::Error,
    },
    /// A detached-data path is absolute or traverses to a parent directory.
    #[error("NRRD detached data path must be relative and cannot traverse parents: {data_file:?}")]
    InvalidDetachedPath {
        /// Path text from the header.
        data_file: String,
    },
    /// A detached file sequence or pattern is not supported by the single-file reader.
    #[error("NRRD detached data source {data_file:?} describes a file set, not one file")]
    UnsupportedDetachedFileSet {
        /// `data file` field value that describes a file set.
        data_file: String,
    },
    /// A detached data file could not be opened.
    #[error("cannot open NRRD data file {path:?}: {source}")]
    OpenDetachedData {
        /// Resolved data path.
        path: PathBuf,
        /// Filesystem failure.
        #[source]
        source: io::Error,
    },
    /// Payload bytes could not be read or decompressed.
    #[error("NRRD payload read failed: {source}")]
    PayloadIo {
        /// Read or decompression failure.
        #[source]
        source: io::Error,
    },
    /// Fewer payload bytes are available than the header declares.
    #[error("NRRD payload contains {actual_bytes} bytes but requires {expected_bytes}")]
    TruncatedPayload {
        /// Required payload length.
        expected_bytes: usize,
        /// Bytes read from the payload.
        actual_bytes: usize,
    },
    /// The payload ended before a declared line or byte skip was consumed.
    #[error("NRRD payload ended while applying {field}: requested {requested}, consumed {actual}")]
    InsufficientPayloadSkip {
        /// Header field whose offset could not be applied.
        field: &'static str,
        /// Requested line or byte count.
        requested: u64,
        /// Count actually consumed before EOF.
        actual: u64,
    },
    /// An ASCII-encoded sample is not representable as its declared type.
    #[error("NRRD ASCII sample {sample_index} value {value:?} is invalid for {sample_type}")]
    InvalidAsciiSample {
        /// Zero-based sample position in acquisition order.
        sample_index: usize,
        /// Text supplied for the sample.
        value: String,
        /// Declared scalar type.
        sample_type: String,
    },
    /// ASCII payload contains fewer samples than the NRRD sizes field declares.
    #[error(
        "NRRD ASCII payload contains {actual_samples} samples but requires {expected_samples}"
    )]
    TruncatedAsciiPayload {
        /// Required sample count.
        expected_samples: usize,
        /// Number of parsed samples.
        actual_samples: usize,
    },
    /// An ASCII sample token exceeds the bounded numeric-token size.
    #[error("NRRD ASCII sample {sample_index} exceeds the {maximum_bytes}-byte token limit")]
    AsciiTokenTooLong {
        /// Zero-based sample position in acquisition order.
        sample_index: usize,
        /// Maximum accepted token length.
        maximum_bytes: usize,
    },
    /// The requested byte count disagrees with the parsed sample count and width.
    #[error("NRRD ASCII sample count requires {actual_bytes} bytes but the header declares {expected_bytes}")]
    PayloadSampleCountMismatch {
        /// Number of bytes declared by the format type and dimensions.
        expected_bytes: usize,
        /// Number of bytes implied by the sample count and codec width.
        actual_bytes: usize,
    },
    /// The declared payload length does not fit the reader's byte limit type.
    #[error("NRRD payload length {expected_bytes} cannot be represented by the reader")]
    PayloadLengthNotRepresentable {
        /// Required payload length.
        expected_bytes: usize,
    },
    /// The payload or one sample volume could not be reserved.
    #[error("cannot reserve memory for NRRD {operation}: {source}")]
    Allocation {
        /// Allocation being requested.
        operation: &'static str,
        /// Reservation failure.
        #[source]
        source: TryReserveError,
    },
    /// The input declares an acquisition axis and must use the series API.
    #[error("NRRD acquisition axis {axis} ({kind:?}) requires the series reader")]
    AcquisitionAxisRequiresSeries {
        /// Zero-based file-axis position.
        axis: usize,
        /// Declared `kinds` value, if present.
        kind: Option<String>,
    },
    /// The stored-series contract cannot represent the declared axis kind.
    #[error("NRRD acquisition kind {kind:?} is not supported by stored-series I/O")]
    UnsupportedAcquisitionKind {
        /// Axis kind from the NRRD `kinds` field.
        kind: String,
    },
    /// Diffusion metadata could not be represented as a validated RITK scheme.
    #[error("NRRD diffusion acquisition metadata is invalid: {source}")]
    DiffusionScheme {
        /// Gradient-table parse or validation failure.
        #[source]
        source: anyhow::Error,
    },
    /// DWMRI metadata requires an explicit non-spatial acquisition axis.
    #[error("NRRD diffusion metadata requires a declared acquisition axis")]
    DiffusionRequiresAcquisitionAxis,
    /// A stored volume's byte length overflows `usize`.
    #[error("NRRD stored-volume byte count overflows usize")]
    VolumeByteCountOverflow,
    /// A selected volume range does not fit within the parsed payload.
    #[error("NRRD payload does not contain volume {volume_index}")]
    InvalidVolumeRange {
        /// Acquisition-order volume index.
        volume_index: usize,
    },
    /// Exact sample decoding rejected the payload.
    #[error("NRRD stored sample decoding failed: {source}")]
    SampleDecoding {
        /// Exact sample-codec failure.
        #[source]
        source: SampleError,
    },
    /// Shared stored-volume validation rejected the decoded samples or geometry.
    #[error("NRRD stored volume is invalid: {source}")]
    StoredVolume {
        /// Shared volume-contract failure.
        #[source]
        source: VolumeError,
    },
    /// The shared stored-series contract rejected axis semantics or extent.
    #[error("NRRD stored series is invalid: {source}")]
    StoredSeries {
        /// Shared series-contract failure.
        #[source]
        source: StoredSeriesError,
    },
    /// The stored-series parser did not retain its preflighted axis meaning.
    #[error("NRRD stored-series axis semantics were not retained")]
    MissingSeriesAxis,
}

/// The NRRD spatial field whose value failed parsing or conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NrrdSpatialMetadataField {
    /// The named world space, coordinate dimension, or coordinate units.
    CoordinateSystem,
    /// The per-axis space-direction vectors.
    SpaceDirections,
    /// The per-axis scalar spacings.
    Spacings,
    /// The world-space origin.
    SpaceOrigin,
    /// Per-axis units the stored-volume geometry cannot model.
    AxisUnits,
    /// Per-axis bounds the stored-volume geometry cannot model.
    AxisBounds,
    /// Per-axis sample centering the stored-volume geometry cannot model.
    Centering,
}

impl std::fmt::Display for NrrdSpatialMetadataField {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let field = match self {
            Self::CoordinateSystem => "coordinate system",
            Self::SpaceDirections => "space directions",
            Self::Spacings => "spacings",
            Self::SpaceOrigin => "space origin",
            Self::AxisUnits => "axis units",
            Self::AxisBounds => "axis bounds",
            Self::Centering => "centering",
        };
        formatter.write_str(field)
    }
}

/// Read one NRRD volume without changing its stored sample type or bits.
///
/// A 2-D file is represented as a one-slice volume. Every 4-D file has an
/// acquisition axis and must use [`read_nrrd_stored_series`], including files
/// whose axis contains one entry.
///
/// # Errors
///
/// Returns a [`NrrdStoredReadError`] that identifies an invalid header field,
/// unsupported encoding or geometry, truncated payload, sample decoding
/// failure, allocation failure, or a violation of the shared stored-volume
/// contract. `budget` bounds encoded payload bytes, decoded sample bytes,
/// gzip-expanded payload bytes (including a declared byte skip), and series
/// volume count before payload allocation. A multi-volume acquisition returns
/// [`NrrdStoredReadError::AcquisitionAxisRequiresSeries`].
pub fn read_nrrd_stored<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
) -> Result<StoredVolume, NrrdStoredReadError> {
    let parsed = parse_nrrd_raw(path, budget, NrrdReadPurpose::StoredVolume)?;
    volumes::decode_stored_volumes(parsed)?
        .into_iter()
        .next()
        .ok_or(NrrdStoredReadError::InvalidVolumeRange { volume_index: 0 })
}

/// Read a NRRD volume sequence without discarding acquisition-axis meaning.
///
/// Both a leading, interleaved acquisition axis and a trailing, contiguous
/// acquisition axis are preserved in acquisition order. The returned series
/// carries its declared `list` or diffusion meaning; an undeclared axis is
/// marked unspecified. Unsupported axis kinds fail rather than becoming lists.
/// `budget` bounds encoded and decoded payload bytes and the returned volume
/// count before sample storage is allocated.
///
/// # Errors
///
/// Returns a [`NrrdStoredReadError`] that identifies an invalid header field,
/// unsupported encoding or geometry, truncated payload, sample decoding
/// failure, allocation failure, or a violation of the shared stored-volume
/// contract. `budget` bounds encoded bytes, decoded samples, gzip-expanded
/// bytes (including a declared byte skip), and series volume count before
/// payload allocation.
pub fn read_nrrd_stored_series<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
) -> Result<StoredSeries, NrrdStoredReadError> {
    let mut parsed = parse_nrrd_raw(path, budget, NrrdReadPurpose::StoredSeries)?;
    let axis = parsed
        .series_axis
        .take()
        .ok_or(NrrdStoredReadError::MissingSeriesAxis)?;
    let volumes = volumes::decode_stored_volumes(parsed)?;
    StoredSeries::new(volumes, axis).map_err(|source| NrrdStoredReadError::StoredSeries { source })
}

pub(super) fn stored_series_axis(
    header: &NrrdHeader,
    acquisition: AcquisitionAxis,
) -> Result<SeriesAxis, NrrdStoredReadError> {
    let has_diffusion = header.key_values.iter().any(|(key, value)| {
        (key.eq_ignore_ascii_case("modality") && value.eq_ignore_ascii_case("DWMRI"))
            || key.to_ascii_uppercase().starts_with("DWMRI_")
    });
    if !has_diffusion && let Some(measurement_frame) = header.fields.get("measurement frame") {
        return Err(NrrdStoredReadError::UnsupportedMeasurementFrame {
            measurement_frame: measurement_frame.clone(),
        });
    }
    if has_diffusion {
        if acquisition == AcquisitionAxis::Absent {
            return Err(NrrdStoredReadError::DiffusionRequiresAcquisitionAxis);
        }
        if let Some(kind) = acquisition_kind(header, acquisition)
            && !kind.eq_ignore_ascii_case("list")
        {
            return Err(NrrdStoredReadError::UnsupportedAcquisitionKind {
                kind: kind.to_owned(),
            });
        }
        let scheme = scheme_from_header(header)
            .map_err(|source| NrrdStoredReadError::DiffusionScheme { source })?;
        return Ok(SeriesAxis::Diffusion(scheme));
    }
    if acquisition == AcquisitionAxis::Absent {
        return Ok(SeriesAxis::SingleVolume);
    }
    match acquisition_kind(header, acquisition) {
        Some(kind) if kind.eq_ignore_ascii_case("list") => Ok(SeriesAxis::List),
        Some(kind) => Err(NrrdStoredReadError::UnsupportedAcquisitionKind {
            kind: kind.to_owned(),
        }),
        None => Ok(SeriesAxis::Unspecified),
    }
}

pub(super) fn acquisition_axis_index(axis: AcquisitionAxis) -> usize {
    match axis {
        AcquisitionAxis::Absent | AcquisitionAxis::Slowest => 3,
        AcquisitionAxis::Fastest => 0,
    }
}

pub(super) fn acquisition_kind(header: &NrrdHeader, axis: AcquisitionAxis) -> Option<&str> {
    let index = acquisition_axis_index(axis);
    header.fields.get("kinds")?.split_whitespace().nth(index)
}
