use ritk_image_io::IntensityUnit;

use super::super::super::stored::NrrdStoredReadError;
use super::super::NrrdReadPurpose;

pub(super) fn value<'a>(
    units: Option<&'a String>,
    read_purpose: &NrrdReadPurpose,
) -> Result<Option<&'a str>, NrrdStoredReadError> {
    let Some(units) = units.filter(|units| !units.is_empty()) else {
        return Ok(None);
    };
    if matches!(read_purpose, NrrdReadPurpose::ComputeF32) {
        return Err(NrrdStoredReadError::UnsupportedSampleUnits {
            units: units.clone(),
        });
    }
    Ok(Some(units))
}

pub(super) fn storage_bytes(
    units: Option<&str>,
    volumes: usize,
) -> Result<usize, NrrdStoredReadError> {
    let Some(units) = units else {
        return Ok(0);
    };
    let bytes_per_volume = std::mem::size_of::<Option<IntensityUnit>>()
        .checked_add(units.len())
        .ok_or(NrrdStoredReadError::DecodedMetadataByteCountOverflow {
            volume_count: 1,
            bytes_per_volume: usize::MAX,
        })?;
    bytes_per_volume.checked_mul(volumes).ok_or(
        NrrdStoredReadError::DecodedMetadataByteCountOverflow {
            volume_count: volumes,
            bytes_per_volume,
        },
    )
}

pub(super) fn retain(units: Option<&str>) -> Result<Option<IntensityUnit>, NrrdStoredReadError> {
    units
        .map(|units| {
            IntensityUnit::new(units.to_owned()).map_err(|source| {
                NrrdStoredReadError::InvalidSampleUnits {
                    units: units.to_owned(),
                    source,
                }
            })
        })
        .transpose()
}
