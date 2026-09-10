//! Primary/secondary fused slice rendering in patient coordinates.
//!
//! This module is the single implementation for compare blending used by the
//! application shell. A fused pixel is sampled from the secondary volume only
//! after the two validated affine grids have been shown to describe parallel
//! patient-space planes in the same frame of reference.
//!
//! # Theorem (convex bounded blend)
//!
//! Let `a in [0, 1]` be the secondary blend weight and let channel values be
//! `p, s in [0, 255]`. Define `b = (1-a) * p + a * s`. Then `b in [0, 255]`.
//! The renderer uses this convex combination after each source has been
//! windowed and colour-mapped.

use egui::ColorImage;
use iris::color::{ColorMap, Normalized};
use thiserror::Error;

use crate::dicom::pet::PetAcquisitionParams;
use crate::geometry::affine::{AffineError, AffineTransform};
use crate::render::{NamedColorMap, WindowLevel};
use crate::LoadedVolume;

/// Failure while validating or rendering a fused compare slice.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum FusionError {
    /// A volume has a zero spatial dimension.
    #[error("{volume} volume has no voxels")]
    EmptyVolume {
        /// The side of the comparison with invalid geometry.
        volume: &'static str,
    },
    /// A volume declares no interleaved channels.
    #[error("{volume} volume declares zero channels")]
    InvalidChannelCount {
        /// The side of the comparison with invalid geometry.
        volume: &'static str,
    },
    /// Fused compare currently operates on scalar presentation values only.
    #[error("{volume} fused rendering requires a scalar volume; received {channels} channels")]
    UnsupportedChannelCount {
        /// The side of the comparison with an unsupported channel layout.
        volume: &'static str,
        /// Number of interleaved channels declared by the volume.
        channels: u8,
    },
    /// The volume's shape and channel count overflow the sample index space.
    #[error("{volume} volume has an overflowing sample layout")]
    InvalidVolumeLayout {
        /// The side of the comparison with invalid geometry.
        volume: &'static str,
    },
    /// The volume data length does not match its declared shape and channels.
    #[error(
        "{volume} volume contains {actual} samples; the declared geometry requires {expected}"
    )]
    InvalidSampleCount {
        /// The side of the comparison with invalid data.
        volume: &'static str,
        /// Number of samples present in the buffer.
        actual: usize,
        /// Number of samples required by the declared layout.
        expected: usize,
    },
    /// A slice axis is outside the three spatial dimensions.
    #[error("{axis} is not a valid slice axis; expected 0, 1, or 2")]
    InvalidAxis {
        /// The invalid axis value.
        axis: usize,
    },
    /// A selected slice is outside its volume's extent.
    #[error("{volume} slice {slice} is outside axis {axis} extent {extent}")]
    SliceOutOfRange {
        /// The side of the comparison with the invalid selection.
        volume: &'static str,
        /// The slice axis.
        axis: usize,
        /// The requested index.
        slice: usize,
        /// The valid axis extent.
        extent: usize,
    },
    /// The blend weight is not a finite value.
    #[error("secondary blend weight must be finite")]
    InvalidBlendWeight,
    /// A volume's physical geometry cannot be inverted or contains invalid values.
    #[error("{volume} volume geometry is invalid: {source}")]
    InvalidGeometry {
        /// The side of the comparison with invalid geometry.
        volume: &'static str,
        /// The affine validation failure.
        #[source]
        source: AffineError,
    },
    /// The two volumes have different explicit DICOM frame identifiers.
    #[error("primary and secondary frame of reference identifiers differ")]
    IncompatibleFrameOfReference,
    /// A differing grid cannot be trusted without two explicit frame identifiers.
    #[error("a differing grid requires frame of reference identifiers on both volumes")]
    MissingFrameOfReference,
    /// The selected primary and secondary planes are not parallel in patient space.
    #[error("primary and secondary slice planes are not parallel")]
    NonParallelPlanes,
    /// The selected secondary plane is not the patient-space plane requested by the primary.
    #[error("secondary slice does not coincide with the selected primary patient-space plane")]
    PlaneMismatch,
    /// The selected planes do not intersect the secondary volume along its normal.
    #[error("primary and secondary volumes do not overlap along the selected slice normal")]
    NoPhysicalOverlap,
}

/// Slice selection and display parameters for one volume in a fused render.
pub struct FusedSliceParams<'a> {
    /// Source volume with validated spatial geometry.
    pub volume: &'a LoadedVolume,
    /// Fixed voxel axis: 0 = depth, 1 = row, 2 = column.
    pub axis: usize,
    /// Fixed voxel index along [`Self::axis`].
    pub slice: usize,
    /// Window and level applied to the source values.
    pub wl: WindowLevel,
    /// Colour map applied after windowing.
    pub colormap: NamedColorMap,
}

/// Render a fused compare slice where the output geometry follows `primary`.
///
/// Primary voxel centres are transformed into patient coordinates and then
/// into the secondary continuous voxel grid. Nearest-neighbour sampling is
/// used for the secondary in-plane coordinates. A primary pixel remains
/// unchanged when its mapped secondary coordinate is outside the secondary
/// field of view. The function rejects mismatched frames and non-parallel or
/// non-coincident planes rather than presenting a normalized-coordinate blend
/// as registered anatomy.
///
/// # Errors
/// Returns [`FusionError`] when a volume, slice selection, frame of reference,
/// or physical plane relationship is invalid.
pub fn render_fused_slice(
    primary: FusedSliceParams<'_>,
    secondary: FusedSliceParams<'_>,
    secondary_alpha: f32,
) -> Result<ColorImage, FusionError> {
    if !secondary_alpha.is_finite() {
        return Err(FusionError::InvalidBlendWeight);
    }
    let primary_transform = validate_volume(primary.volume, "primary")?;
    let secondary_transform = validate_volume(secondary.volume, "secondary")?;
    validate_selection(primary.volume, "primary", primary.axis, primary.slice)?;
    validate_selection(
        secondary.volume,
        "secondary",
        secondary.axis,
        secondary.slice,
    )?;
    validate_frame_of_reference(primary.volume, secondary.volume)?;
    validate_parallel_planes(
        &primary_transform,
        primary.axis,
        &secondary_transform,
        secondary.axis,
    )?;

    let primary_center = slice_center_voxel(primary.axis, primary.slice, primary.volume.shape);
    let secondary_center =
        secondary_transform.patient_to_voxel(primary_transform.voxel_to_patient(primary_center));
    validate_secondary_plane_coordinate(
        secondary_center[secondary.axis],
        secondary.slice,
        secondary.volume.shape[secondary.axis],
    )?;

    let (primary_pixels, width, height) = primary.volume.extract_slice(primary.axis, primary.slice);
    if width == 0 || height == 0 {
        return Err(FusionError::EmptyVolume { volume: "primary" });
    }

    let alpha = secondary_alpha.clamp(0.0, 1.0);
    let inverse_alpha = 1.0 - alpha;
    let mut rgb = vec![0_u8; width * height * 3];
    let primary_value_transform = DisplayValueTransform::for_volume(primary.volume);
    let secondary_value_transform = DisplayValueTransform::for_volume(secondary.volume);

    for row in 0..height {
        for col in 0..width {
            let primary_voxel = voxel_for_slice(primary.axis, primary.slice, row, col);
            let patient = primary_transform.voxel_to_patient(primary_voxel);
            let secondary_voxel = secondary_transform.patient_to_voxel(patient);
            if (secondary_voxel[secondary.axis] - secondary.slice as f64).abs()
                > index_tolerance(secondary_voxel[secondary.axis], secondary.slice as f64)
            {
                return Err(FusionError::PlaneMismatch);
            }

            let p = primary_value_transform.apply(primary_pixels[row * width + col]);
            let s = nearest_secondary_pixel(
                secondary.volume,
                secondary_voxel,
                secondary.axis,
                secondary.slice,
            )
            .map(|value| secondary_value_transform.apply(value));
            let p_rgb = primary
                .colormap
                .sample(Normalized::from_u8(primary.wl.apply(p)))
                .to_rgba8();
            let s_rgb = s.map_or([0, 0, 0, 0], |value| {
                secondary
                    .colormap
                    .sample(Normalized::from_u8(secondary.wl.apply(value)))
                    .to_rgba8()
            });
            let out_idx = (row * width + col) * 3;
            for channel in 0..3 {
                rgb[out_idx + channel] = if s.is_some() {
                    (inverse_alpha * f32::from(p_rgb[channel]) + alpha * f32::from(s_rgb[channel]))
                        .round() as u8
                } else {
                    p_rgb[channel]
                };
            }
        }
    }

    Ok(ColorImage::from_rgb([width, height], &rgb))
}

/// Map the centre of a primary slice into the secondary slice index.
///
/// This helper is used by compare viewports to derive a physically matching
/// secondary plane before calling [`render_fused_slice`].
///
/// # Errors
/// Returns [`FusionError`] when either volume is malformed, frame identities
/// are not compatible, the planes are not parallel, or the primary plane lies
/// outside the secondary normal extent.
pub fn secondary_slice_for_primary(
    primary: &LoadedVolume,
    primary_axis: usize,
    primary_slice: usize,
    secondary: &LoadedVolume,
    secondary_axis: usize,
) -> Result<usize, FusionError> {
    let primary_transform = validate_volume(primary, "primary")?;
    let secondary_transform = validate_volume(secondary, "secondary")?;
    validate_selection(primary, "primary", primary_axis, primary_slice)?;
    validate_axis(secondary_axis)?;
    validate_frame_of_reference(primary, secondary)?;
    validate_parallel_planes(
        &primary_transform,
        primary_axis,
        &secondary_transform,
        secondary_axis,
    )?;

    let primary_center = slice_center_voxel(primary_axis, primary_slice, primary.shape);
    let patient = primary_transform.voxel_to_patient(primary_center);
    let secondary_voxel = secondary_transform.patient_to_voxel(patient);
    let extent = secondary.shape[secondary_axis];
    let coordinate = secondary_voxel[secondary_axis];
    if !coordinate.is_finite()
        || coordinate < -0.5
        || coordinate > extent.saturating_sub(1) as f64 + 0.5
    {
        return Err(FusionError::NoPhysicalOverlap);
    }
    let selected = coordinate.round();
    if selected < 0.0 || selected >= extent as f64 {
        return Err(FusionError::NoPhysicalOverlap);
    }
    Ok(selected as usize)
}

#[derive(Clone, Copy)]
struct DisplayValueTransform {
    pet: Option<PetAcquisitionParams>,
    delta_t_s: f64,
}

impl DisplayValueTransform {
    fn for_volume(volume: &LoadedVolume) -> Self {
        let pet = if is_pet_modality(volume.modality.as_deref()) {
            PetAcquisitionParams::from_loaded_volume(volume)
        } else {
            None
        };
        Self {
            pet,
            delta_t_s: PetAcquisitionParams::delta_t_s_from_vol(volume),
        }
    }

    #[inline]
    fn apply(self, pixel: f32) -> f64 {
        let raw = f64::from(pixel);
        self.pet
            .map(|pet| pet.pixel_to_suvbw(raw, self.delta_t_s))
            .unwrap_or(raw)
    }
}

fn is_pet_modality(modality: Option<&str>) -> bool {
    modality
        .and_then(|m| m.trim().get(..2).map(str::to_ascii_uppercase))
        .is_some_and(|prefix| prefix == "PT")
}

fn validate_volume(
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

fn validate_axis(axis: usize) -> Result<(), FusionError> {
    (axis < 3)
        .then_some(())
        .ok_or(FusionError::InvalidAxis { axis })
}

fn validate_selection(
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

fn validate_frame_of_reference(
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

fn validate_parallel_planes(
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

fn plane_cosine_tolerance() -> f64 {
    // Normalized direction cosines parsed from DICOM carry roughly one unit
    // round-off per component. A square-root-epsilon bound with a small
    // dimensional guard accepts that representation error without accepting a
    // visibly oblique plane.
    64.0 * f64::EPSILON.sqrt()
}

fn index_tolerance(first: f64, second: f64) -> f64 {
    plane_cosine_tolerance() * (1.0 + first.abs().max(second.abs()))
}

fn validate_secondary_plane_coordinate(
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

fn slice_center_voxel(axis: usize, slice: usize, shape: [usize; 3]) -> [f64; 3] {
    let mut voxel = [0.0; 3];
    for (index, extent) in shape.into_iter().enumerate() {
        voxel[index] = (extent.saturating_sub(1) as f64) * 0.5;
    }
    voxel[axis] = slice as f64;
    voxel
}

fn voxel_for_slice(axis: usize, slice: usize, row: usize, col: usize) -> [f64; 3] {
    match axis {
        0 => [slice as f64, row as f64, col as f64],
        1 => [row as f64, slice as f64, col as f64],
        _ => [row as f64, col as f64, slice as f64],
    }
}

fn nearest_secondary_pixel(
    volume: &LoadedVolume,
    continuous_voxel: [f64; 3],
    axis: usize,
    slice: usize,
) -> Option<f32> {
    let mut indices = [0_usize; 3];
    indices[axis] = slice;
    for (index, extent) in volume.shape.into_iter().enumerate() {
        if index == axis {
            continue;
        }
        let coordinate = continuous_voxel[index];
        if !coordinate.is_finite()
            || coordinate < -0.5
            || coordinate > extent.saturating_sub(1) as f64 + 0.5
        {
            return None;
        }
        let nearest = coordinate.round();
        if nearest < 0.0 || nearest >= extent as f64 {
            return None;
        }
        indices[index] = nearest as usize;
    }
    Some(volume.pixel_at(indices[0], indices[1], indices[2]))
}

#[cfg(test)]
#[path = "tests_fusion.rs"]
mod tests;
