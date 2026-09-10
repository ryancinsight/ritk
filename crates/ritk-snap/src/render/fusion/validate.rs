//! Preconditions that must hold before any secondary voxel is sampled.
//!
//! Fusion presents one volume's anatomy through another's grid, so a blend
//! computed across mismatched frames of reference or non-parallel planes
//! would look registered while being nothing of the kind. These checks are
//! what stands between the two.

use super::FusionError;
use crate::geometry::affine::AffineTransform;
use crate::LoadedVolume;

pub(super) fn validate_volume(
    volume: &LoadedVolume,
    name: &'static str,
) -> Result<AffineTransform, FusionError> {
    if volume.shape.contains(&0) {
        return Err(FusionError::EmptyVolume { volume: name });
    }
    let channels = usize::from(volume.channels);
    if channels == 0 {
        return Err(FusionError::InvalidChannelCount { volume: name });
    }
    if channels != 1 {
        return Err(FusionError::UnsupportedChannelCount {
            volume: name,
            channels: volume.channels,
        });
    }
    let voxel_count = volume
        .shape
        .iter()
        .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
        .ok_or(FusionError::InvalidVolumeLayout { volume: name })?;
    let expected = voxel_count
        .checked_mul(channels)
        .ok_or(FusionError::InvalidVolumeLayout { volume: name })?;
    if volume.data.len() != expected {
        return Err(FusionError::InvalidSampleCount {
            volume: name,
            actual: volume.data.len(),
            expected,
        });
    }
    AffineTransform::from_parts(volume.origin, volume.direction, volume.spacing).map_err(|source| {
        FusionError::InvalidGeometry {
            volume: name,
            source,
        }
    })
}

pub(super) fn validate_axis(axis: usize) -> Result<(), FusionError> {
    (axis < 3)
        .then_some(())
        .ok_or(FusionError::InvalidAxis { axis })
}

pub(super) fn validate_selection(
    volume: &LoadedVolume,
    name: &'static str,
    axis: usize,
    slice: usize,
) -> Result<(), FusionError> {
    validate_axis(axis)?;
    let extent = volume.shape[axis];
    if slice >= extent {
        return Err(FusionError::SliceOutOfRange {
            volume: name,
            axis,
            slice,
            extent,
        });
    }
    Ok(())
}

pub(super) fn validate_frame_of_reference(
    primary: &LoadedVolume,
    secondary: &LoadedVolume,
) -> Result<(), FusionError> {
    let primary_uid = primary
        .metadata
        .as_deref()
        .and_then(|metadata| metadata.frame_of_reference_uid.as_ref());
    let secondary_uid = secondary
        .metadata
        .as_deref()
        .and_then(|metadata| metadata.frame_of_reference_uid.as_ref());
    match (primary_uid, secondary_uid) {
        (Some(primary_uid), Some(secondary_uid)) if primary_uid == secondary_uid => Ok(()),
        (Some(_), Some(_)) => Err(FusionError::IncompatibleFrameOfReference),
        (None, None)
            if primary.origin == secondary.origin
                && primary.spacing == secondary.spacing
                && primary.direction == secondary.direction =>
        {
            Ok(())
        }
        _ => Err(FusionError::MissingFrameOfReference),
    }
}

pub(super) fn validate_parallel_planes(
    primary: &AffineTransform,
    primary_axis: usize,
    secondary: &AffineTransform,
    secondary_axis: usize,
) -> Result<(), FusionError> {
    let Some(primary_normal) = primary.axis_direction(primary_axis) else {
        return Err(FusionError::NonParallelPlanes);
    };
    let Some(secondary_normal) = secondary.axis_direction(secondary_axis) else {
        return Err(FusionError::NonParallelPlanes);
    };
    let dot = primary_normal[0] * secondary_normal[0]
        + primary_normal[1] * secondary_normal[1]
        + primary_normal[2] * secondary_normal[2];
    if dot.abs() + plane_cosine_tolerance() < 1.0 {
        return Err(FusionError::NonParallelPlanes);
    }
    Ok(())
}

pub(super) fn plane_cosine_tolerance() -> f64 {
    // Normalized direction cosines parsed from DICOM carry roughly one unit
    // round-off per component. A square-root-epsilon bound with a small
    // dimensional guard accepts that representation error without accepting a
    // visibly oblique plane.
    64.0 * f64::EPSILON.sqrt()
}

pub(super) fn index_tolerance(first: f64, second: f64) -> f64 {
    plane_cosine_tolerance() * (1.0 + first.abs().max(second.abs()))
}

pub(super) fn validate_secondary_plane_coordinate(
    coordinate: f64,
    selected_slice: usize,
    extent: usize,
) -> Result<(), FusionError> {
    if !coordinate.is_finite()
        || coordinate < -0.5
        || coordinate > extent.saturating_sub(1) as f64 + 0.5
    {
        return Err(FusionError::NoPhysicalOverlap);
    }
    if (coordinate - selected_slice as f64).abs()
        > index_tolerance(coordinate, selected_slice as f64)
    {
        return Err(FusionError::PlaneMismatch);
    }
    Ok(())
}
