//! Viewport zoom policy and scroll-to-zoom mapping SSOT.
//!
//! This module owns the deterministic mapping from scroll-wheel delta to
//! viewport zoom factor. The policy is intentionally pure and side-effect free
//! so app integration points can be tested independently from UI event wiring.

use eunomia::FloatElement;

/// Minimum permitted viewport zoom multiplier.
pub const MIN_ZOOM: f32 = 0.05;

/// Maximum permitted viewport zoom multiplier.
pub const MAX_ZOOM: f32 = 32.0;

/// Canonical zoom multiplier for fit-to-panel rendering.
pub const FIT_ZOOM: f32 = 1.0;

/// Canonical pan offset for fit-to-panel rendering.
pub const FIT_PAN_OFFSET: [f32; 2] = [0.0, 0.0];

/// Scale mapping from wheel-delta units to multiplicative zoom change.
const SCROLL_SENSITIVITY: f32 = 0.0015;

/// Lower bound for one scroll-step multiplicative change.
const MIN_STEP_FACTOR: f32 = 0.25;

/// Upper bound for one scroll-step multiplicative change.
const MAX_STEP_FACTOR: f32 = 4.0;

/// Pixel-to-zoom sensitivity for drag-based zoom.
const DRAG_SENSITIVITY: f32 = 0.01;

/// Lower bound for one drag update multiplicative change.
const MIN_DRAG_STEP_FACTOR: f32 = 0.25;

/// Upper bound for one drag update multiplicative change.
const MAX_DRAG_STEP_FACTOR: f32 = 4.0;

/// Return whether wheel input should drive zoom instead of slice stepping.
///
/// On Windows/Linux this is `Ctrl`; on macOS `Command` can be mapped by the
/// caller into `ctrl_or_cmd_pressed`.
pub fn should_zoom_with_scroll(ctrl_or_cmd_pressed: bool) -> bool {
    ctrl_or_cmd_pressed
}

/// Compute the next zoom multiplier from current zoom and wheel delta.
///
/// Positive `scroll_y` zooms in. Negative `scroll_y` zooms out.
pub fn zoom_from_scroll(current_zoom: f32, scroll_y: f32) -> f32 {
    if scroll_y == 0.0 {
        return current_zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    }

    let step_factor = (1.0 + scroll_y * SCROLL_SENSITIVITY).clamp(MIN_STEP_FACTOR, MAX_STEP_FACTOR);
    (current_zoom * step_factor).clamp(MIN_ZOOM, MAX_ZOOM)
}

/// Return the canonical fit-to-panel pan/zoom transform.
pub fn fit_view_transform() -> ([f32; 2], f32) {
    (FIT_PAN_OFFSET, FIT_ZOOM)
}

/// Zoom multiplier that renders one frame pixel as one screen pixel.
///
/// This is the exact reciprocal of the native placement fit scale:
/// with `zoom = 1 / min(panel / physical)` the texel factors reduce to the
/// spacing-relative unit, so square-pixel data maps 1:1.
///
/// Returns `None` when the panel or frame extent is degenerate (zero or
/// non-finite physical size) or the reciprocal leaves the `f32` range.
pub fn one_to_one_zoom(panel: [u32; 2], frame: [u32; 2], display_spacing: [f64; 2]) -> Option<f32> {
    let physical = [
        f64::from(frame[0]) * display_spacing[0],
        f64::from(frame[1]) * display_spacing[1],
    ];
    if !physical
        .iter()
        .all(|extent| extent.is_finite() && *extent > 0.0)
    {
        return None;
    }
    let fit = (f64::from(panel[0]) / physical[0]).min(f64::from(panel[1]) / physical[1]);
    if !fit.is_finite() || fit <= 0.0 {
        return None;
    }
    let zoom = 1.0 / fit;
    if !zoom.is_finite() || zoom < f64::from(f32::MIN) || zoom > f64::from(f32::MAX) {
        return None;
    }
    // The zoom multiplier is an f32 viewport factor; the finite f64
    // reciprocal is range-checked against that contract above, so narrowing
    // matches the cast bit-for-bit.
    Some(f32::from_f64(zoom))
}

/// Compute zoom multiplier from a vertical drag delta in screen pixels.
///
/// Upward drag (`drag_delta_y < 0`) zooms in. Downward drag zooms out.
pub fn zoom_from_drag_delta(original_zoom: f32, drag_delta_y: f32) -> f32 {
    if drag_delta_y == 0.0 {
        return original_zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    }

    let step_factor =
        (1.0 - drag_delta_y * DRAG_SENSITIVITY).clamp(MIN_DRAG_STEP_FACTOR, MAX_DRAG_STEP_FACTOR);
    (original_zoom * step_factor).clamp(MIN_ZOOM, MAX_ZOOM)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_modifier_policy_is_identity() {
        assert!(should_zoom_with_scroll(true));
        assert!(!should_zoom_with_scroll(false));
    }

    #[test]
    fn zoom_increases_on_positive_scroll() {
        let z = zoom_from_scroll(1.0, 120.0);
        assert!(z > 1.0, "expected zoom-in, got {z}");
    }

    #[test]
    fn zoom_decreases_on_negative_scroll() {
        let z = zoom_from_scroll(1.0, -120.0);
        assert!(z < 1.0, "expected zoom-out, got {z}");
    }

    #[test]
    fn zoom_clamps_to_supported_range() {
        let z_min = zoom_from_scroll(0.01, -10_000.0);
        let z_max = zoom_from_scroll(99.0, 10_000.0);
        assert_eq!(z_min, MIN_ZOOM);
        assert_eq!(z_max, MAX_ZOOM);
    }

    #[test]
    fn zero_scroll_preserves_zoom() {
        let z = zoom_from_scroll(2.75, 0.0);
        assert!((z - 2.75).abs() < 1e-6, "unexpected zoom {z}");
    }

    #[test]
    fn one_to_one_zoom_is_the_reciprocal_fit_scale() {
        let zoom = one_to_one_zoom([200, 100], [100, 50], [1.0, 1.0]).expect("finite extent");
        assert!((zoom - 0.5).abs() < 1e-6, "unexpected 1:1 zoom {zoom}");
        let anisotropic = one_to_one_zoom([200, 100], [50, 50], [2.0, 1.0]).expect("finite extent");
        assert!(
            (anisotropic - 0.5).abs() < 1e-6,
            "unexpected anisotropic 1:1 zoom {anisotropic}"
        );
    }

    #[test]
    fn one_to_one_zoom_rejects_degenerate_extents() {
        assert_eq!(one_to_one_zoom([200, 100], [0, 50], [1.0, 1.0]), None);
        assert_eq!(one_to_one_zoom([0, 0], [100, 50], [1.0, 1.0]), None);
        assert_eq!(one_to_one_zoom([200, 100], [100, 50], [0.0, 1.0]), None);
    }

    #[test]
    fn fit_view_transform_returns_canonical_fit_state() {
        let (pan, zoom) = fit_view_transform();
        assert_eq!(pan, [0.0, 0.0]);
        assert_eq!(zoom, 1.0);
    }

    #[test]
    fn zoom_drag_up_increases_zoom() {
        let z = zoom_from_drag_delta(1.0, -20.0);
        assert!(z > 1.0, "expected zoom-in, got {z}");
    }

    #[test]
    fn zoom_drag_down_decreases_zoom() {
        let z = zoom_from_drag_delta(1.0, 20.0);
        assert!(z < 1.0, "expected zoom-out, got {z}");
    }

    #[test]
    fn zoom_drag_clamps_to_supported_range() {
        let z_min = zoom_from_drag_delta(0.01, 10_000.0);
        let z_max = zoom_from_drag_delta(99.0, -10_000.0);
        assert_eq!(z_min, MIN_ZOOM);
        assert_eq!(z_max, MAX_ZOOM);
    }
}
