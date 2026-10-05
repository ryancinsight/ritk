//! Stored-volume assembly from a parsed NRRD payload.
//!
//! The payload arrives bounded and decoded to raw bytes by
//! [`super::volume::parse_nrrd_raw`](crate::reader::volume::parse_nrrd_raw);
//! this module deinterleaves acquisition layouts, decodes exact stored
//! samples, and validates each volume against the shared stored-volume
//! contract. Rejections carry the offending counts, never a truncated volume.

use ritk_codecs::SampleBuffer;
use ritk_image::ImageMetadata;
use ritk_image_io::{IntensityCalibration, StoredVolume};

use super::super::decode::sample_type;
use super::super::volume::RawNrrd;
use super::NrrdStoredReadError;
use crate::axes::AcquisitionAxis;

pub(super) fn decode_stored_volumes(
    parsed: RawNrrd,
) -> Result<Vec<StoredVolume>, NrrdStoredReadError> {
    let RawNrrd {
        series_axis: _,
        raw_bytes,
        element_type,
        byte_order,
        acquisition,
        volumes,
        voxels_per_volume,
        dims,
        origin,
        spacing,
        direction,
        coordinate_map,
    } = parsed;
    let stored_type =
        sample_type(&element_type).map_err(|_| NrrdStoredReadError::UnsupportedElementType {
            element_type: element_type.clone(),
        })?;
    let bytes_per_sample = stored_type.byte_width();
    let bytes_per_volume = voxels_per_volume
        .checked_mul(bytes_per_sample)
        .ok_or(NrrdStoredReadError::VolumeByteCountOverflow)?;
    let expected_bytes = bytes_per_volume
        .checked_mul(volumes)
        .ok_or(NrrdStoredReadError::VolumeByteCountOverflow)?;
    if raw_bytes.len() < expected_bytes {
        return Err(NrrdStoredReadError::TruncatedPayload {
            expected_bytes,
            actual_bytes: raw_bytes.len(),
        });
    }

    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(volumes)
        .map_err(|source| NrrdStoredReadError::Allocation {
            operation: "stored-volume table",
            source,
        })?;
    for volume_index in 0..volumes {
        let mut volume_bytes = Vec::new();
        volume_bytes
            .try_reserve_exact(bytes_per_volume)
            .map_err(|source| NrrdStoredReadError::Allocation {
                operation: "stored-volume payload",
                source,
            })?;
        match acquisition {
            AcquisitionAxis::Fastest => {
                for sample_bytes in raw_bytes
                    .chunks_exact(bytes_per_sample)
                    .skip(volume_index)
                    .step_by(volumes)
                {
                    volume_bytes.extend_from_slice(sample_bytes);
                }
            }
            AcquisitionAxis::Absent | AcquisitionAxis::Slowest => {
                let start = volume_index
                    .checked_mul(bytes_per_volume)
                    .ok_or(NrrdStoredReadError::VolumeByteCountOverflow)?;
                let end = start
                    .checked_add(bytes_per_volume)
                    .ok_or(NrrdStoredReadError::VolumeByteCountOverflow)?;
                let payload = raw_bytes
                    .get(start..end)
                    .ok_or(NrrdStoredReadError::InvalidVolumeRange { volume_index })?;
                volume_bytes.extend_from_slice(payload);
            }
        }
        let samples = SampleBuffer::decode(stored_type, &volume_bytes, byte_order)
            .map_err(|source| NrrdStoredReadError::SampleDecoding { source })?;
        let metadata = ImageMetadata::new(origin, spacing, direction);
        let volume = StoredVolume::new(
            dims,
            samples,
            metadata,
            coordinate_map.clone(),
            IntensityCalibration::Identity,
        )
        .map_err(|source| NrrdStoredReadError::StoredVolume { source })?;
        decoded.push(volume);
    }
    Ok(decoded)
}
