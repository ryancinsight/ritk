//! RT-DOSE overlay colorization helpers.
//!
//! Converts projected RT-DOSE scalar slices into straight-alpha RGBA images.

use crate::render::slice_render::RgbaImage;
use crate::ui::rtdose_overlay::dose_to_rgba;

/// Convert opacity in `[0, 1]` to the effective alpha byte used by dose overlay mapping.
pub fn overlay_alpha(opacity: f32) -> u8 {
    (opacity.clamp(0.0, 1.0) * 200.0) as u8
}

/// Return `(min, max)` over strictly positive finite dose values.
///
/// Returns `None` when no positive finite values are present, or when the
/// resulting range is degenerate (`min >= max`).
pub fn positive_finite_dose_range(dose_map: &[f32]) -> Option<(f32, f32)> {
    let (min_dose, max_dose) = dose_map
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .fold((f32::MAX, f32::MIN), |(mn, mx), v| (mn.min(v), mx.max(v)));
    (min_dose < max_dose).then_some((min_dose, max_dose))
}

/// Build a colorized RT-DOSE overlay image in row-major order.
///
/// `dose_map` must have length `rows * cols`.
pub fn build_overlay_image(
    dose_map: &[f32],
    rows: usize,
    cols: usize,
    min_dose: f32,
    max_dose: f32,
    opacity: f32,
) -> Option<RgbaImage> {
    if rows == 0 || cols == 0 || dose_map.len() != rows.saturating_mul(cols) {
        return None;
    }
    let pixels = dose_map
        .iter()
        .flat_map(|&dose| dose_to_rgba(dose, min_dose, max_dose, opacity))
        .collect();
    Some(RgbaImage::new([cols, rows], pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_range_ignores_nan_and_zero() {
        let map = [f32::NAN, -1.0, 0.0, 2.0, 5.0, 3.0];
        let range = positive_finite_dose_range(&map).expect("positive finite range must exist");
        assert_eq!(range.0, 2.0);
        assert_eq!(range.1, 5.0);
    }

    #[test]
    fn positive_range_returns_none_for_degenerate_values() {
        let map = [0.0, f32::NAN, -2.0, 4.0, 4.0];
        assert!(positive_finite_dose_range(&map).is_none());
    }

    #[test]
    fn build_overlay_image_produces_expected_shape_and_alpha() {
        let map = [0.0, f32::NAN, 2.5, 5.0];
        let img = build_overlay_image(&map, 2, 2, 2.5, 5.0, 0.5).expect("valid image");
        assert_eq!(img.size(), [2, 2]);
        let pixels: Vec<[u8; 4]> = img.rgba_pixels().collect();
        assert_eq!(pixels.len(), 4);

        // Zero and NaN dose are fully transparent by definition.
        assert_eq!(pixels[0][3], 0);
        assert_eq!(pixels[1][3], 0);

        // Positive in-range dose must be visible.
        assert!(pixels[2][3] > 0);
        assert!(pixels[3][3] > 0);
    }

    #[test]
    fn overlay_alpha_clamps_to_byte_range() {
        assert_eq!(overlay_alpha(-2.0), 0);
        assert_eq!(overlay_alpha(0.0), 0);
        assert_eq!(overlay_alpha(1.0), 200);
        assert_eq!(overlay_alpha(2.0), 200);
    }
}
