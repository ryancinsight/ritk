//! Primary/secondary fused slice rendering.
//!
//! This module is the SSOT for 2-volume compare blending used by the app shell.
//!
//! # Theorem (convex bounded blend)
//!
//! Let `a in [0, 1]` be the secondary blend weight and let channel values be
//! `p, s in [0, 255]`. Define:
//! `b = (1-a) * p + a * s`.
//!
//! Then `b in [0, 255]`.
//!
//! Proof sketch:
//! `b` is a convex combination of two values in a closed interval, so it
//! remains in the same interval.

use egui::ColorImage;

use crate::dicom::pet::PetAcquisitionParams;
use crate::render::{Colormap, WindowLevel};
use crate::LoadedVolume;

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

/// Slice selection and display parameters for one volume in a fused render.
pub struct FusedSliceParams<'a> {
    pub volume: &'a LoadedVolume,
    pub axis: usize,
    pub slice: usize,
    pub wl: WindowLevel,
    pub colormap: Colormap,
}

/// Render a fused compare slice where the output geometry follows `primary`.
///
/// Secondary pixels are sampled with nearest-neighbor interpolation in
/// normalized slice coordinates so mismatched slice dimensions can still be
/// blended into the primary output frame.
pub fn render_fused_slice(
    primary: FusedSliceParams<'_>,
    secondary: FusedSliceParams<'_>,
    secondary_alpha: f32,
) -> ColorImage {
    let primary_axis = primary.axis;
    let primary_slice = primary.slice;
    let primary_wl = primary.wl;
    let primary_colormap = primary.colormap;
    let secondary_axis = secondary.axis;
    let secondary_slice = secondary.slice;
    let secondary_wl = secondary.wl;
    let secondary_colormap = secondary.colormap;
    let (primary, secondary) = (primary.volume, secondary.volume);
    let (primary_pixels, width, height) = primary.extract_slice(primary_axis, primary_slice);
    if width == 0 || height == 0 {
        return ColorImage::from_rgb([1, 1], &[0, 0, 0]);
    }

    let (secondary_pixels, secondary_width, secondary_height) =
        secondary.extract_slice(secondary_axis, secondary_slice);
    if secondary_width == 0 || secondary_height == 0 {
        return ColorImage::from_rgb([width, height], &vec![0; width * height * 3]);
    }

    let alpha = secondary_alpha.clamp(0.0, 1.0);
    let inv_alpha = 1.0 - alpha;
    let mut rgb = vec![0u8; width * height * 3];
    let primary_transform = DisplayValueTransform::for_volume(primary);
    let secondary_transform = DisplayValueTransform::for_volume(secondary);

    for row in 0..height {
        let sy = ((row as f32 + 0.5) * secondary_height as f32 / height as f32)
            .floor()
            .clamp(0.0, (secondary_height - 1) as f32) as usize;
        for col in 0..width {
            let sx = ((col as f32 + 0.5) * secondary_width as f32 / width as f32)
                .floor()
                .clamp(0.0, (secondary_width - 1) as f32) as usize;

            let p = primary_transform.apply(primary_pixels[row * width + col]);
            let s = secondary_transform.apply(secondary_pixels[sy * secondary_width + sx]);
            let p_rgb = primary_colormap.map(primary_wl.apply(p) as f32 / super::U8_MAX_F32);
            let s_rgb = secondary_colormap.map(secondary_wl.apply(s) as f32 / super::U8_MAX_F32);

            let out_idx = (row * width + col) * 3;
            rgb[out_idx] = (inv_alpha * p_rgb[0] as f32 + alpha * s_rgb[0] as f32).round() as u8;
            rgb[out_idx + 1] =
                (inv_alpha * p_rgb[1] as f32 + alpha * s_rgb[1] as f32).round() as u8;
            rgb[out_idx + 2] =
                (inv_alpha * p_rgb[2] as f32 + alpha * s_rgb[2] as f32).round() as u8;
        }
    }

    ColorImage::from_rgb([width, height], &rgb)
}

#[cfg(test)]
#[path = "tests_fusion.rs"]
mod tests;
