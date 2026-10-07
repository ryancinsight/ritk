use thiserror::Error;

/// A DICOM stored-sample reader rejected the input before exposing a volume.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoredDicomError {
    /// The series scanner rejected the selected DICOM input.
    #[error("DICOM series scan failed: {0}")]
    Scan(#[source] anyhow::Error),
    /// A pre-scanned series did not retain its validated Part 10 bytes.
    #[error("stored DICOM import requires scanner-retained Part 10 bytes")]
    MissingRetainedBytes,
    /// A retained Part 10 instance could not be parsed.
    #[error("retained DICOM instance could not be parsed: {0}")]
    Parse(#[source] anyhow::Error),
    /// A required DICOM image-pixel attribute was absent.
    #[error("required DICOM attribute {tag} is absent")]
    MissingTag {
        /// Attribute name and tag number.
        tag: &'static str,
    },
    /// A DICOM attribute could not be represented by its required value type.
    #[error("DICOM attribute {tag} has an invalid value")]
    InvalidTag {
        /// Attribute name and tag number.
        tag: &'static str,
    },
    /// The transfer syntax does not expose native little-endian stored pixels.
    #[error("stored DICOM import does not support transfer syntax {uid}")]
    UnsupportedTransferSyntax {
        /// Transfer Syntax UID from the Part 10 file meta information.
        uid: String,
    },
    /// The pixel representation is not a single monochrome sample per pixel.
    #[error("stored DICOM import requires one monochrome sample per pixel, got {samples}")]
    UnsupportedSamples {
        /// Samples declared for each pixel.
        samples: usize,
    },
    /// The monochrome photometric interpretation cannot be represented.
    #[error("stored DICOM import does not support photometric interpretation {value}")]
    UnsupportedPhotometricInterpretation {
        /// Photometric interpretation from the DICOM image-pixel module.
        value: String,
    },
    /// One DICOM instance contains more than one frame.
    #[error("stored DICOM import requires one frame per instance, got {frames}")]
    UnsupportedFrames {
        /// Number of frames declared by the instance.
        frames: usize,
    },
    /// Pixel encoding differs between instances in a series.
    #[error("DICOM series instances use inconsistent stored pixel encodings")]
    InconsistentPixelEncoding,
    /// Calibration differs between instances in a series.
    #[error("DICOM series instances use inconsistent modality calibration")]
    InconsistentCalibration,
    /// The dataset specifies both a modality LUT and a linear rescale.
    #[error("DICOM specifies both Modality LUT and rescale calibration")]
    ConflictingCalibrationForms,
    /// A Modality LUT sequence or its descriptor/data are malformed.
    #[error("DICOM Modality LUT is invalid: {field}")]
    InvalidModalityLookupTable {
        /// LUT sequence component rejected by the reader.
        field: &'static str,
    },
    /// Pixel Value Field length is inconsistent with declared dimensions.
    #[error(
        "DICOM PixelData has {actual} bytes; expected {expected} bytes including valid padding"
    )]
    PixelDataLength {
        /// Observed Pixel Data length.
        actual: usize,
        /// Required unpadded Pixel Data length.
        expected: usize,
    },
    /// Pixel Data could not be converted to byte values.
    #[error("DICOM PixelData is not a supported byte value: {0}")]
    PixelValue(#[source] anyhow::Error),
    /// Stored pixel encoding was invalid or could not be decoded.
    #[error(transparent)]
    PixelDecode(#[from] ritk_codecs::StoredPixelError),
    /// A scanner-retained calibration value was non-finite or malformed.
    #[error("DICOM modality calibration is invalid")]
    InvalidCalibration,
    /// Image geometry is absent, invalid, or unrepresentable.
    #[error("DICOM image geometry is invalid: {field}")]
    InvalidGeometry {
        /// Geometry component rejected by the reader.
        field: &'static str,
    },
    /// The output dimensions overflowed the host or allocation budget.
    #[error("DICOM stored volume dimensions overflow")]
    ShapeOverflow,
    /// Allocating bounded stored-sample storage failed.
    #[error("DICOM stored-sample allocation failed: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    /// The peak decoded workspace exceeds the configured read budget.
    #[error("DICOM stored import exceeds the decoded-byte budget: {0}")]
    Budget(#[source] anyhow::Error),
    /// Stored-volume metadata validation failed.
    #[error(transparent)]
    Volume(#[from] ritk_image_io::VolumeError),
    /// Stored-series invariants were not satisfied.
    #[error(transparent)]
    Series(#[from] ritk_image_io::StoredSeriesError),
    /// A modality calibration value was not finite.
    #[error(transparent)]
    Calibration(#[from] ritk_image_io::CalibrationError),
}
