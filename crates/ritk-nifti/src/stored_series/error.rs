//! Errors returned while preparing and encoding a stored-series conversion.

use crate::document::NiftiDocumentError;
use ritk_codecs::{SampleError, SampleType};
use ritk_image_io::{
    ConversionCapabilityReport, ConversionLocation, ConversionPrepareError, ConversionRejection,
};
use thiserror::Error;

/// A stored-series conversion failed before a NIfTI document was produced.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NiftiStoredSeriesError {
    /// The capability report lists semantics that NIfTI cannot represent.
    #[error("NIfTI conversion is blocked by unsupported source semantics")]
    Capabilities(ConversionCapabilityReport),
    /// An unrecognized preflight result from the shared conversion contract.
    #[error(transparent)]
    Preparation(NiftiConversionPreparationError),
    /// An input value or combination violates the NIfTI target contract.
    #[error(transparent)]
    Rejected(#[from] NiftiStoredSeriesRejection),
    /// The complete document failed NIfTI validation.
    #[error(transparent)]
    Document(#[from] NiftiDocumentError),
    /// Stored samples could not be written to the output buffer.
    #[error(transparent)]
    SampleEncoding(NiftiSampleEncodingError),
    /// The bounded document allocation could not be reserved.
    #[error("could not reserve {bytes} bytes for a NIfTI document: {source}")]
    Allocation {
        /// Requested document capacity.
        bytes: usize,
        /// Allocation failure.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// The sample encoder produced a byte count different from the plan.
    #[error("NIfTI output has {actual} bytes; the prepared plan requires {expected}")]
    OutputSizeMismatch {
        /// Planned total document size.
        expected: usize,
        /// Actual total document size.
        actual: usize,
    },
}

/// An otherwise-unclassified error from shared conversion preparation.
///
/// The underlying error remains available through [`std::error::Error::source`].
#[derive(Debug, Error)]
#[error(transparent)]
pub struct NiftiConversionPreparationError {
    source: ConversionPrepareError<NiftiStoredSeriesRejection>,
}

impl NiftiConversionPreparationError {
    pub(super) fn new(source: ConversionPrepareError<NiftiStoredSeriesRejection>) -> Self {
        Self { source }
    }
}

/// An error while encoding stored samples into the NIfTI payload.
///
/// The underlying error remains available through [`std::error::Error::source`].
#[derive(Debug, Error)]
#[error(transparent)]
pub struct NiftiSampleEncodingError {
    source: SampleError,
}

impl NiftiSampleEncodingError {
    pub(super) fn new(source: SampleError) -> Self {
        Self { source }
    }
}

/// A target-specific reason and source location for rejecting a stored series.
#[derive(Debug, Error)]
#[error("NIfTI target rejects {location:?}: {issue}")]
#[non_exhaustive]
pub struct NiftiStoredSeriesRejection {
    /// The series, volume, or frame that cannot be represented.
    pub location: ConversionLocation,
    /// The violated NIfTI representation constraint.
    pub issue: NiftiStoredSeriesIssue,
}

impl ConversionRejection for NiftiStoredSeriesRejection {
    fn location(&self) -> ConversionLocation {
        self.location
    }
}

/// A value or combination that the NIfTI header cannot preserve.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum NiftiStoredSeriesIssue {
    /// Volumes in one NIfTI series must share one spatial shape.
    #[error("volume shape differs from the first volume")]
    ShapeMismatch {
        /// Required depth, row, column shape.
        expected: [usize; 3],
        /// Rejected depth, row, column shape.
        actual: [usize; 3],
    },
    /// Volumes in one NIfTI series must share one stored sample type.
    #[error("volume sample type differs from the first volume")]
    SampleTypeMismatch {
        /// Required stored sample type.
        expected: SampleType,
        /// Rejected stored sample type.
        actual: SampleType,
    },
    /// NIfTI stores one physical transform for all volumes.
    #[error("volume geometry differs from the first volume")]
    GeometryMismatch,
    /// The source sample type has no NIfTI scalar datatype mapping.
    #[error("NIfTI cannot encode stored sample type {sample_type:?}")]
    UnsupportedSampleType {
        /// Rejected stored sample type.
        sample_type: SampleType,
    },
    /// NIfTI stores one global calibration for all volumes.
    #[error("volume calibration differs from the first volume")]
    CalibrationMismatch,
    /// Per-frame calibration values differ within a volume.
    #[error("per-frame calibration differs within the volume")]
    PerFrameCalibrationMismatch,
    /// NIfTI's zero slope means scaling is disabled, not a constant map.
    #[error("NIfTI cannot encode a zero-slope calibration")]
    ZeroSlopeCalibration,
    /// NIfTI has no representation for a nonlinear modality lookup table.
    #[error("NIfTI cannot encode nonlinear modality lookup calibration")]
    UnsupportedCalibration,
    /// A dimension, geometry, scaling value, or header field is not representable.
    #[error("NIfTI header field is not representable: {detail}")]
    HeaderField {
        /// Format parser or field-validation detail.
        detail: Box<str>,
    },
    /// The checked voxel or payload byte count overflowed.
    #[error("NIfTI sample payload size overflows usize")]
    PayloadSizeOverflow,
    /// The complete uncompressed document exceeds the decoder's bound.
    #[error("NIfTI document needs {bytes} bytes; the limit is {limit}")]
    DocumentTooLarge {
        /// Required total document size.
        bytes: usize,
        /// Maximum total document size.
        limit: usize,
    },
}
