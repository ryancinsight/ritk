//! Viewport zoom policy and scroll-to-zoom mapping SSOT.
//!
//! This module owns the deterministic mapping from scroll-wheel delta to
//! viewport zoom factor. The policy is intentionally pure and side-effect free
//! so app integration points can be tested independently from UI event wiring.

/// Minimum permitted viewport zoom multiplier.
pub const MIN_ZOOM: f32 = 0.05;

/// Maximum permitted viewport zoom multiplier.
pub const MAX_ZOOM: f32 = 32.0;

/// Scale mapping from wheel-delta units to multiplicative zoom change.
const SCROLL_SENSITIVITY: f32 = 0.0015;

/// Lower bound for one scroll-step multiplicative change.
const MIN_STEP_FACTOR: f32 = 0.25;

/// Upper bound for one scroll-step multiplicative change.
const MAX_STEP_FACTOR: f32 = 4.0;

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
}
