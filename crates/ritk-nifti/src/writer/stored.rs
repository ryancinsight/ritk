//! Typed NIfTI-2 output for the shared stored-volume contract.

use std::io;
use std::path::Path;

use ritk_codecs::{ByteOrder, SampleError, SampleType};
use ritk_image_io::{IntensityCalibration, LinearCalibration, StoredVolume};
use thiserror::Error;

use crate::header::{HeaderDims, HeaderVersion, NiftiDatatype, NiftiHeaderError};

use super::{direction_row_major, header_from_spatial, write_single_file_with};

/// Writes a stored volume to an uncompressed or gzip-compressed NIfTI-2 file.
///
/// The writer preserves all ten fixed-width scalar representations, their
/// stored bits, Cartesian LPS-millimeter geometry, and one volume-wide linear
/// calibration. It emits little-endian samples and uses `.nii.gz` to select
/// gzip compression. NIfTI-2 stores the affine and calibration coefficients as
/// `f64`.
///
/// # Examples
///
/// ```no_run
/// use ritk_codecs::SampleBuffer;
/// use ritk_image::ImageMetadata;
/// use ritk_image_io::{IntensityCalibration, StoredVolume};
/// use ritk_nifti::write_nifti_stored;
/// use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
///
/// fn main() -> anyhow::Result<()> {
///     let volume = StoredVolume::new(
///         [1, 1, 1],
///         SampleBuffer::from_samples(vec![42_u16]),
///         ImageMetadata::new(
///             Point::new([0.0; 3]),
///             Spacing::new([1.0; 3]),
///             Direction::identity(),
///         ),
///         CoordinateMap::Cartesian,
///         IntensityCalibration::Identity,
///     )?;
///     write_nifti_stored("scan.nii.gz", &volume)?;
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// Returns a typed capability error before creating or truncating the output
/// when the coordinate map, sample type, or calibration cannot be represented.
/// Header, sample-codec, and filesystem failures are reported separately.
pub fn write_nifti_stored<P: AsRef<Path>>(
    path: P,
    volume: &StoredVolume,
) -> Result<(), NiftiStoredWriteError> {
    if !volume.coordinate_map().is_cartesian() {
        return Err(NiftiStoredWriteError::UnsupportedCoordinateMap);
    }
    let datatype = nifti_datatype(volume.samples().sample_type())?;
    let (slope, intercept) = nifti_scaling(volume.calibration())?;
    let [nz, ny, nx] = volume.shape();
    let header = header_from_spatial(
        HeaderVersion::Two,
        HeaderDims { nx, ny, nz },
        datatype,
        volume.metadata().origin().to_array(),
        volume.metadata().spacing().to_array(),
        direction_row_major(volume.metadata().direction()),
    )
    .map_err(NiftiHeaderError::from)?;
    let mut header = header;
    header.scl_slope = slope;
    header.scl_inter = intercept;

    write_single_file_with(path, &header, |writer| {
        volume
            .samples()
            .write_to(writer, ByteOrder::LeastSignificantByteFirst)
            .map_err(NiftiStoredWriteError::SampleEncoding)
    })
}

fn nifti_datatype(sample_type: SampleType) -> Result<NiftiDatatype, NiftiStoredWriteError> {
    match sample_type {
        SampleType::U8 => Ok(NiftiDatatype::Uint8),
        SampleType::I8 => Ok(NiftiDatatype::Int8),
        SampleType::U16 => Ok(NiftiDatatype::Uint16),
        SampleType::I16 => Ok(NiftiDatatype::Int16),
        SampleType::U32 => Ok(NiftiDatatype::Uint32),
        SampleType::I32 => Ok(NiftiDatatype::Int32),
        SampleType::U64 => Ok(NiftiDatatype::Uint64),
        SampleType::I64 => Ok(NiftiDatatype::Int64),
        SampleType::F32 => Ok(NiftiDatatype::Float32),
        SampleType::F64 => Ok(NiftiDatatype::Float64),
        _ => Err(NiftiStoredWriteError::UnsupportedSampleType { sample_type }),
    }
}

fn nifti_scaling(calibration: &IntensityCalibration) -> Result<(f64, f64), NiftiStoredWriteError> {
    match calibration {
        IntensityCalibration::Identity => Ok((0.0, 0.0)),
        IntensityCalibration::Linear(calibration) => scaling_coefficients(*calibration),
        IntensityCalibration::PerFrameLinear(calibrations) => {
            let first = calibrations
                .first()
                .copied()
                .ok_or(NiftiStoredWriteError::EmptyFrameCalibration)?;
            if calibrations.iter().any(|calibration| *calibration != first) {
                return Err(NiftiStoredWriteError::VaryingFrameCalibration);
            }
            scaling_coefficients(first)
        }
        IntensityCalibration::ModalityLookup(_) => {
            Err(NiftiStoredWriteError::ModalityLookupCalibration)
        }
    }
}

fn scaling_coefficients(
    calibration: LinearCalibration,
) -> Result<(f64, f64), NiftiStoredWriteError> {
    if calibration.slope() == 0.0 {
        return Err(NiftiStoredWriteError::ZeroSlopeCalibration);
    }
    if calibration.is_identity() {
        Ok((0.0, 0.0))
    } else {
        Ok((calibration.slope(), calibration.intercept()))
    }
}

/// A stored volume cannot be represented by the NIfTI-2 scalar-volume writer.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NiftiStoredWriteError {
    /// NIfTI's affine cannot represent a non-Cartesian coordinate map.
    #[error("NIfTI cannot preserve this non-Cartesian coordinate map")]
    UnsupportedCoordinateMap,
    /// A future codec sample type has no NIfTI scalar datatype code.
    #[error("NIfTI cannot represent stored sample type {sample_type:?}")]
    UnsupportedSampleType {
        /// Stored representation without a NIfTI scalar datatype code.
        sample_type: SampleType,
    },
    /// NIfTI scaling fields cannot represent modality lookup calibration.
    #[error("NIfTI scl_slope and scl_inter cannot preserve a modality lookup table")]
    ModalityLookupCalibration,
    /// NIfTI has one global scaling transform, not varying frame transforms.
    #[error("NIfTI cannot preserve calibration that varies between frames")]
    VaryingFrameCalibration,
    /// A zero NIfTI slope disables scaling, so a zero-slope transform is lost.
    #[error("NIfTI scl_slope=0 disables scaling and cannot preserve this calibration")]
    ZeroSlopeCalibration,
    /// A per-frame calibration must contain at least one entry.
    #[error("NIfTI cannot encode an empty per-frame calibration")]
    EmptyFrameCalibration,
    /// The NIfTI header could not represent the volume dimensions or geometry.
    #[error(transparent)]
    Header(#[from] NiftiHeaderError),
    /// The stored-sample codec could not write a complete payload.
    #[error("NIfTI stored sample encoding failed: {0}")]
    SampleEncoding(#[source] SampleError),
    /// File creation, write, or flush failed.
    #[error("NIfTI output failed: {0}")]
    Io(#[from] io::Error),
}
