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

use crate::render::{Colormap, WindowLevel};
use crate::LoadedVolume;

/// Render a fused compare slice where the output geometry follows `primary`.
///
/// Secondary pixels are sampled with nearest-neighbor interpolation in
/// normalized slice coordinates so mismatched slice dimensions can still be
/// blended into the primary output frame.
pub fn render_fused_slice(
    primary: &LoadedVolume,
    primary_axis: usize,
    primary_slice: usize,
    primary_wl: WindowLevel,
    primary_colormap: Colormap,
    secondary: &LoadedVolume,
    secondary_axis: usize,
    secondary_slice: usize,
    secondary_wl: WindowLevel,
    secondary_colormap: Colormap,
    secondary_alpha: f32,
) -> ColorImage {
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

    for row in 0..height {
        let sy = ((row as f32 + 0.5) * secondary_height as f32 / height as f32)
            .floor()
            .clamp(0.0, (secondary_height - 1) as f32) as usize;
        for col in 0..width {
            let sx = ((col as f32 + 0.5) * secondary_width as f32 / width as f32)
                .floor()
                .clamp(0.0, (secondary_width - 1) as f32) as usize;

            let p = primary_pixels[row * width + col] as f64;
            let s = secondary_pixels[sy * secondary_width + sx] as f64;
            let p_rgb = primary_colormap.map(primary_wl.apply(p) as f32 / 255.0);
            let s_rgb = secondary_colormap.map(secondary_wl.apply(s) as f32 / 255.0);

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
mod tests {
    use super::*;
    use crate::render::SliceRenderer;

    fn test_volume(shape: [usize; 3], scale: f32) -> LoadedVolume {
        let [d, r, c] = shape;
        let mut data = Vec::with_capacity(d * r * c);
        for z in 0..d {
            for y in 0..r {
                for x in 0..c {
                    data.push(scale * (z * r * c + y * c + x) as f32);
                }
            }
        }
        LoadedVolume {
            data: std::sync::Arc::new(data),
            shape,
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            metadata: None,
            source: None,
            modality: None,
            patient_name: None,
            patient_id: None,
            study_date: None,
            series_description: None,
            patient_weight_kg: None,
            injected_dose_bq: None,
            radionuclide_half_life_s: None,
            radiopharmaceutical_start_time: None,
            decay_correction: None,
        }
    }

    #[test]
    fn alpha_zero_equals_primary_render() {
        let p = test_volume([2, 3, 4], 1.0);
        let s = test_volume([2, 3, 4], 2.0);
        let wl = WindowLevel::new(128.0, 256.0);
        let fused = render_fused_slice(
            &p,
            0,
            1,
            wl,
            Colormap::Grayscale,
            &s,
            0,
            1,
            wl,
            Colormap::Hot,
            0.0,
        );
        let primary = SliceRenderer::render(&p, 0, 1, wl, Colormap::Grayscale);
        assert_eq!(fused.size, primary.size);
        assert_eq!(fused.pixels, primary.pixels);
    }

    #[test]
    fn output_size_matches_primary_slice_geometry() {
        let p = test_volume([5, 7, 9], 1.0);
        let s = test_volume([3, 4, 6], 1.0);
        let wl = WindowLevel::new(64.0, 128.0);
        let fused = render_fused_slice(
            &p,
            1,
            2,
            wl,
            Colormap::Grayscale,
            &s,
            2,
            3,
            wl,
            Colormap::Jet,
            0.5,
        );
        // Axis 1 slice is [depth, cols] => [5, 9] in [rows, cols], egui [width, height].
        assert_eq!(fused.size, [9, 5]);
    }
}
