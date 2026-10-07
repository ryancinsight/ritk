//! Decoding validated NIfTI documents into exact stored samples.
//!
//! This is the read half of the NIfTI ↔ [`StoredSeries`] conversion contract
//! described in `docs/architecture.md` (Theorem 21.1). The write half lives in
//! [`crate::stored_series`]. Both share [`crate::spatial`] for axis/affine
//! mapping, so the file-axis permutation has exactly one definition.

use crate::header::{NiftiHeader, RANK_SERIES};
use crate::spatial::metadata_from_nifti_ras_affine_precise;
use crate::NiftiDocument;
use consus_core::ByteOrder as ConsusByteOrder;
use ritk_codecs::{ByteOrder, SampleBuffer, SampleError, SampleType};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    CalibrationError, IntensityCalibration, LinearCalibration, SeriesAxis, StoredSeries,
    StoredSeriesError, StoredVolume, VolumeError,
};
use ritk_spatial::CoordinateMap;
use std::collections::TryReserveError;
use thiserror::Error;

/// Spatial-unit field mask within the packed NIfTI `xyzt_units` value.
const SPATIAL_UNIT_MASK: i32 = 0x07;
/// NIfTI spatial-unit code for millimetres.
const SPATIAL_UNIT_MILLIMETER: i32 = 2;

impl NiftiDocument {
    /// Decode this validated document into exact stored samples and typed series metadata.
    ///
    /// The payload is decoded in the file's declared byte order and retains its
    /// stored sample type; no rescaling is applied. Physical geometry is
    /// reconstructed at `f64` precision from the active spatial transform, and
    /// the NIfTI scaling fields become the volume's [`IntensityCalibration`].
    ///
    /// A rank-3 document yields one [`SeriesAxis::SingleVolume`]; a rank-4
    /// document yields an ordered [`SeriesAxis::List`], including when it
    /// declares a single volume.
    ///
    /// # Errors
    ///
    /// Returns a scoped typed error when the header cannot be re-parsed, the
    /// spatial transform is not representable as LPS-millimetre metadata, the
    /// spatial units are not millimetres, the scaling fields are not a finite
    /// linear calibration, the payload is truncated, a decoded volume violates
    /// the stored-volume contract, or a sample cannot be decoded.
    pub fn to_stored_series(&self) -> Result<StoredSeries, NiftiStoredReadError> {
        let header =
            NiftiHeader::parse(self.uncompressed_bytes()).map_err(NiftiStoredReadError::Header)?;
        reject_non_millimeter_units(header.xyzt_units)?;
        let sample_type = SampleType::from(header.datatype);
        let byte_order = codec_byte_order(header.byte_order());
        let affine = header
            .affine_precise()
            .map_err(NiftiStoredReadError::Spatial)?;
        let spatial = metadata_from_nifti_ras_affine_precise(affine)
            .map_err(NiftiStoredReadError::Spatial)?;
        let metadata = ImageMetadata::new(spatial.origin, spatial.spacing, spatial.direction);
        let calibration = calibration_from_scaling(header.scl_slope, header.scl_inter)?;

        // NIfTI `dim[1..=3]` is `[nx, ny, nz]`; the stored volume is `[nz, ny, nx]`.
        let shape = [header.dim[3], header.dim[2], header.dim[1]];
        let voxels = header
            .voxels_per_volume()
            .map_err(NiftiStoredReadError::Payload)?;
        let volume_count = header.volume_count();
        let bytes_per_volume = voxels
            .checked_mul(sample_type.byte_width())
            .ok_or(NiftiStoredReadError::PayloadSizeOverflow)?;

        let payload = self.sample_bytes();
        let mut volumes = Vec::new();
        volumes
            .try_reserve_exact(volume_count)
            .map_err(NiftiStoredReadError::Allocation)?;
        for volume_index in 0..volume_count {
            let start = volume_index
                .checked_mul(bytes_per_volume)
                .ok_or(NiftiStoredReadError::PayloadSizeOverflow)?;
            let end = start
                .checked_add(bytes_per_volume)
                .ok_or(NiftiStoredReadError::PayloadSizeOverflow)?;
            let encoded = payload
                .get(start..end)
                .ok_or(NiftiStoredReadError::TruncatedPayload { volume_index })?;
            let samples = SampleBuffer::decode(sample_type, encoded, byte_order)
                .map_err(NiftiStoredReadError::Sample)?;
            let volume = StoredVolume::new(
                shape,
                samples,
                metadata.clone(),
                CoordinateMap::Cartesian,
                calibration.clone(),
            )
            .map_err(|source| NiftiStoredReadError::Volume {
                volume_index,
                source,
            })?;
            volumes.push(volume);
        }

        let axis = if header.dim[0] == RANK_SERIES {
            SeriesAxis::List
        } else {
            SeriesAxis::SingleVolume
        };
        StoredSeries::new(volumes, axis).map_err(NiftiStoredReadError::Series)
    }
}

/// Map the header's consus byte order onto the codec byte order.
const fn codec_byte_order(order: ConsusByteOrder) -> ByteOrder {
    match order {
        ConsusByteOrder::LittleEndian => ByteOrder::LeastSignificantByteFirst,
        ConsusByteOrder::BigEndian => ByteOrder::MostSignificantByteFirst,
    }
}

/// Reject spatial units the stored model cannot represent as millimetres.
///
/// Code `0` (unknown) keeps the LPS-millimetre interpretation; code `2` is
/// millimetres. Metres (`1`) and microns (`3`) would require conversion, which
/// this read path reports as typed loss instead of applying silently.
fn reject_non_millimeter_units(xyzt_units: i32) -> Result<(), NiftiStoredReadError> {
    let units = xyzt_units & SPATIAL_UNIT_MASK;
    if units == 0 || units == SPATIAL_UNIT_MILLIMETER {
        Ok(())
    } else {
        Err(NiftiStoredReadError::UnsupportedSpatialUnits { code: units })
    }
}

/// Turn NIfTI scaling fields into an intensity calibration.
///
/// NIfTI treats `scl_slope == 0` as "scaling disabled", so it maps to
/// [`IntensityCalibration::Identity`]. A nonzero slope with a non-finite
/// intercept, or a non-finite slope, is rejected rather than narrowed.
fn calibration_from_scaling(
    slope: f64,
    intercept: f64,
) -> Result<IntensityCalibration, NiftiStoredReadError> {
    if slope == 0.0 {
        return Ok(IntensityCalibration::Identity);
    }
    LinearCalibration::new(slope, intercept)
        .map(IntensityCalibration::Linear)
        .map_err(NiftiStoredReadError::Calibration)
}

/// Failure to decode a validated NIfTI document into stored image values.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NiftiStoredReadError {
    /// The stored document bytes could not be re-parsed as a NIfTI header.
    #[error("NIfTI stored-read header is invalid: {0}")]
    Header(#[source] anyhow::Error),
    /// The active spatial transform is not representable as LPS-millimetre geometry.
    #[error("NIfTI spatial transform is not representable: {0}")]
    Spatial(#[source] anyhow::Error),
    /// The declared payload shape is invalid.
    #[error("NIfTI payload shape is invalid: {0}")]
    Payload(#[source] anyhow::Error),
    /// The document declares spatial units other than millimetres.
    #[error("NIfTI spatial units code {code} is not millimetres")]
    UnsupportedSpatialUnits {
        /// The rejected `xyzt_units` spatial-unit code.
        code: i32,
    },
    /// The scaling fields are not a finite linear calibration.
    #[error("NIfTI scaling is not a finite linear calibration: {0}")]
    Calibration(#[source] CalibrationError),
    /// The payload byte count overflows `usize`.
    #[error("NIfTI payload byte count overflows usize")]
    PayloadSizeOverflow,
    /// The stored payload ends before a declared volume.
    #[error("NIfTI payload is truncated before volume {volume_index}")]
    TruncatedPayload {
        /// The zero-based volume that could not be read.
        volume_index: usize,
    },
    /// The volume allocation could not be reserved.
    #[error("cannot reserve NIfTI stored-read output: {0}")]
    Allocation(#[source] TryReserveError),
    /// A stored sample could not be decoded.
    #[error("cannot decode stored NIfTI samples: {0}")]
    Sample(#[source] SampleError),
    /// A decoded volume violates the stored-volume contract.
    #[error("volume {volume_index} violates the stored-volume contract: {source}")]
    Volume {
        /// The zero-based volume that violates the contract.
        volume_index: usize,
        /// The stored-volume contract violation.
        #[source]
        source: VolumeError,
    },
    /// The decoded volumes do not form a stored series.
    #[error("decoded NIfTI volumes do not form a stored series: {0}")]
    Series(#[source] StoredSeriesError),
}

#[cfg(test)]
#[path = "stored_read_tests.rs"]
mod tests;
