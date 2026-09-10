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

use crate::dicom::pet::PetAcquisitionParams;
use crate::render::{GrayscalePresentation, NamedColorMap, WindowLevel};
use crate::LoadedVolume;

mod errors;
mod geometry;
mod validate;

pub use errors::FusionError;
use geometry::{nearest_secondary_pixel, slice_center_voxel, voxel_for_slice};
use validate::{
    index_tolerance, validate_axis, validate_frame_of_reference, validate_parallel_planes,
    validate_secondary_plane_coordinate, validate_selection, validate_volume,
};

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

    let primary_presentation =
        GrayscalePresentation::for_volume(primary.volume).map_err(|source| {
            FusionError::InvalidPresentation {
                volume: "primary",
                source,
            }
        })?;
    let secondary_presentation =
        GrayscalePresentation::for_volume(secondary.volume).map_err(|source| {
            FusionError::InvalidPresentation {
                volume: "secondary",
                source,
            }
        })?;

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
                .sample(Normalized::from_u8(
                    primary_presentation.apply(primary.wl, p),
                ))
                .to_rgba8();
            let s_rgb = s.map_or([0, 0, 0, 0], |value| {
                secondary
                    .colormap
                    .sample(Normalized::from_u8(
                        secondary_presentation.apply(secondary.wl, value),
                    ))
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

#[cfg(test)]
#[path = "tests_fusion.rs"]
mod tests;
