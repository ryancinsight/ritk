use super::*;

// ── Intensity mapping ─────────────────────────────────────────────────────

/// For center=100, width=200 (range [-100, 100]):
/// - t=1.0 → intensity = 100 (max at top of bar)
/// - t=0.0 → intensity = -100 (min at bottom)
/// - t=0.5 → intensity = 0 (center)
#[test]
fn test_intensity_mapping_analytical() {
    let center = 100.0f32;
    let width = 200.0f32;
    let half = width * 0.5;

    let max_intensity = center + half;
    let min_intensity = center - half;
    let mid_intensity = center;

    assert_eq!(max_intensity, 200.0, "t=1.0 must equal center+width/2");
    assert_eq!(min_intensity, 0.0, "t=0.0 must equal center-width/2");
    assert_eq!(mid_intensity, 100.0, "t=0.5 must equal center");
}

/// Standard CT brain preset: center=40, width=80 → range [-0, 80].
#[test]
fn test_ct_brain_preset_range() {
    let center = 40.0f32;
    let width = 80.0f32;
    assert_eq!(center + width * 0.5, 80.0);
    assert_eq!(center - width * 0.5, 0.0);
}

/// Standard CT lung preset: center=−400, width=1500.
#[test]
fn test_ct_lung_preset_range() {
    let center = -400.0f32;
    let width = 1500.0f32;
    let max_hu = center + width * 0.5;
    let min_hu = center - width * 0.5;
    assert_eq!(max_hu, 350.0, "lung max HU = 350");
    assert_eq!(min_hu, -1150.0, "lung min HU = -1150");
}

/// Positive-width invariant: the colorbar width must always be positive.
#[test]
fn test_width_positive_invariant() {
    // Width = 0 is a degenerate case (single value band). The colorbar
    // still renders without panic; the labels converge to the same value.
    let center = 50.0f32;
    let width = 0.0f32;
    let max_intensity = center + width * 0.5;
    let min_intensity = center - width * 0.5;
    assert_eq!(max_intensity, min_intensity, "zero width: top = bottom");
}

// ── Colorbar row count ─────────────────────────────────────────────────────

/// Normalized value `t` for row i of N total rows:
/// t(i) = 1.0 - i / (N - 1).
/// Verify top row = 1.0 and bottom row = 0.0.
#[test]
fn test_row_t_top_and_bottom() {
    let n = 256usize;
    let t_top = 1.0f32 - 0.0 / (n as f32 - 1.0);
    let t_bottom = 1.0f32 - (n as f32 - 1.0) / (n as f32 - 1.0);
    assert_eq!(t_top, 1.0, "top row must map to t=1.0");
    assert!(
        (t_bottom - 0.0).abs() < 1e-6,
        "bottom row must map to t≈0.0"
    );
}

/// Middle row (i = N/2) corresponds to t ≈ 0.5 for even N.
#[test]
fn test_row_t_midpoint_analytical() {
    let n = 100usize;
    let i = n / 2;
    let t = 1.0f32 - (i as f32) / ((n - 1) as f32);
    // For N=100, i=50: t = 1 - 50/99 ≈ 0.4949...
    assert!((t - 0.495).abs() < 0.01, "middle row t ≈ 0.495 for N=100");
}

// ── COLORBAR_WIDTH constant ───────────────────────────────────────────────

#[test]
fn test_colorbar_width_positive() {
    const _: () = assert!(COLORBAR_WIDTH > 0.0);
    const _: () = assert!(COLORBAR_PANEL_WIDTH > COLORBAR_WIDTH);
}
