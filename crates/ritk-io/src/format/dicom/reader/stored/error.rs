//! Typed failures of the exact stored-sample DICOM import.

use ritk_codecs::{SampleExtractionError, SampleType, StoredPixelError};
use ritk_dicom::PixelSignedness;
use ritk_image_io::{StoredSeriesError, VolumeError};
use std::path::PathBuf;
use thiserror::Error;

/// A DICOM series the exact stored-sample import cannot represent.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DicomStoredImportError {
    /// The directory scan or an instance parse failed.
    #[error("DICOM stored import could not read the series: {0}")]
    Reader(#[source] anyhow::Error),
    /// The series declares no pixel geometry.
    #[error("DICOM stored import requires non-empty rows, columns, and slice depth")]
    EmptyGeometry,
    /// The scanned slice count disagrees with the series' declared depth.
    #[error("DICOM stored import found {actual} slices but the series declares {expected}")]
    SliceCountMismatch {
        /// Depth declared by the series metadata.
        expected: usize,
        /// Slice descriptors the scan produced.
        actual: usize,
    },
    /// A non-monochrome photometric interpretation has no scalar stored form.
    #[error("DICOM stored import requires MONOCHROME1 or MONOCHROME2; got {photometric}")]
    NonMonochrome {
        /// Photometric interpretation declared by the source.
        photometric: String,
    },
    /// A compressed transfer syntax has no stored payload before decoding.
    #[error("DICOM stored import requires an uncompressed transfer syntax; {uid} is compressed")]
    CompressedSyntax {
        /// Transfer syntax UID that requires a decode step.
        uid: String,
    },
    /// A big-endian transfer syntax has no little-endian stored payload.
    #[error("DICOM stored import requires a little-endian transfer syntax; {uid} is big-endian")]
    BigEndianSyntax {
        /// Big-endian transfer syntax UID.
        uid: String,
    },
    /// The modality transform would change the stored values.
    #[error(
        "DICOM stored import requires identity rescale; {path:?} declares slope {slope} \
         intercept {intercept}"
    )]
    NonIdentityCalibration {
        /// Slice declaring the non-identity transform.
        path: PathBuf,
        /// RescaleSlope (0028,1053).
        slope: f32,
        /// RescaleIntercept (0028,1054).
        intercept: f32,
    },
    /// Slice positions would require resampling stored samples.
    #[error(
        "DICOM stored import requires uniformly spaced slices; the series would need resampling"
    )]
    NonUniformGeometry,
    /// HighBit is not BitsStored-1, so the stored bits are not the low-order ones.
    #[error("DICOM stored import requires HighBit = BitsStored-1; got {high_bit} for BitsStored {bits_stored}")]
    UnexpectedHighBit {
        /// BitsStored (0028,0101).
        bits_stored: u16,
        /// HighBit (0028,0102).
        high_bit: u16,
    },
    /// BitsAllocated has no fixed-width stored sample type.
    #[error("DICOM stored import cannot represent BitsAllocated={bits_allocated} with {representation:?}")]
    UnsupportedSampleWidth {
        /// BitsAllocated (0028,0100).
        bits_allocated: u16,
        /// PixelRepresentation (0028,0103).
        representation: PixelSignedness,
    },
    /// The series uses a stored sample representation the shared model lacks.
    #[error("DICOM stored import cannot represent stored sample type {sample_type:?}")]
    UnsupportedSampleType {
        /// Sample representation without a shared fixed-width type.
        sample_type: SampleType,
    },
    /// The series mixes stored sample types across slices.
    #[error("DICOM stored import requires one sample type; {path:?} uses {actual:?}, expected {expected:?}")]
    MixedSampleTypes {
        /// Slice whose sample type differs.
        path: PathBuf,
        /// Sample type the first slice established.
        expected: SampleType,
        /// Sample type this slice actually decoded to.
        actual: SampleType,
    },
    /// The pixel payload is shorter than the declared frame.
    #[error(
        "DICOM stored import found {actual} pixel bytes in {path:?}; one frame needs {expected}"
    )]
    ShortPixelFrame {
        /// Slice with the short payload.
        path: PathBuf,
        /// Bytes one frame requires.
        expected: usize,
        /// Bytes the Pixel Data element supplied.
        actual: usize,
    },
    /// A stored pixel frame could not be decoded exactly.
    #[error("DICOM stored pixel decode failed for {path:?}: {source}")]
    Pixel {
        /// Slice whose frame failed to decode.
        path: PathBuf,
        /// Exact sample-codec failure.
        #[source]
        source: StoredPixelError,
    },
    /// A decoded frame did not yield the series' sample type.
    #[error("DICOM stored import produced inconsistent samples: {0}")]
    InconsistentSamples(#[source] SampleExtractionError),
    /// The decoded volume failed the shared stored-volume contract.
    #[error(transparent)]
    Volume(#[from] VolumeError),
    /// The assembled series failed the shared stored-series contract.
    #[error(transparent)]
    Series(#[from] StoredSeriesError),
}
