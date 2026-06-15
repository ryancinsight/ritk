use super::*;

// ── bar_height_log tests ──────────────────────────────────────────────────

/// Peak count returns exactly available_height.
///
/// Proof: ln(peak+1)/ln(peak+1) = 1.0; 1.0 × h = h.
#[test]
fn bar_height_log_at_peak_returns_available_height() {
    let h = bar_height_log(100, 100, 80.0);
    assert!((h - 80.0).abs() < 1e-4, "expected 80.0, got {h}");
}

/// Zero count returns 0.0 regardless of peak.
///
/// Proof: ln(0+1) = ln(1) = 0.0; 0.0/anything = 0.0.
#[test]
fn bar_height_log_zero_count_returns_zero() {
    let h = bar_height_log(0, 100, 80.0);
    assert_eq!(h, 0.0, "zero count must produce 0 height");
}

/// Zero peak returns 0.0 (guard against division).
#[test]
fn bar_height_log_zero_peak_returns_zero() {
    let h = bar_height_log(50, 0, 80.0);
    assert_eq!(h, 0.0, "zero peak must produce 0 height");
}

/// Half-peak bin has height strictly between 0 and available_height.
///
/// Analytical: ln(51) / ln(101) × 80.0 ≈ 3.932 / 4.615 × 80.0 ≈ 68.18
#[test]
fn bar_height_log_half_peak_is_strictly_between_zero_and_max() {
    let h = bar_height_log(50, 100, 80.0);
    assert!(
        h > 0.0 && h < 80.0,
        "half-peak height {h} must be in (0, 80)"
    );
    // Analytical: ln(51)/ln(101) × 80.0
    let expected = (51f64.ln() / 101f64.ln() * 80.0) as f32;
    assert!(
        (h - expected).abs() < 1e-3,
        "expected {expected:.4}, got {h:.4}"
    );
}

// ── wl_to_x tests ────────────────────────────────────────────────────────

/// Centre value maps to the exact mid-point of the pixel range.
///
/// Analytical: t = (50 − 0) / (100 − 0) = 0.5; x = 0.0 + 0.5 × 200.0 = 100.0.
#[test]
fn wl_to_x_centre_value_maps_to_midpoint() {
    let x = wl_to_x(50.0, 0.0, 100.0, 0.0, 200.0);
    assert!((x - 100.0).abs() < 1e-4, "expected x=100.0, got {x}");
}

/// Value below hist_min clamps to x_left.
#[test]
fn wl_to_x_below_range_clamps_to_left() {
    let x = wl_to_x(-10.0, 0.0, 100.0, 10.0, 110.0);
    assert!(
        (x - 10.0).abs() < 1e-4,
        "expected x=10.0 (clamped), got {x}"
    );
}

/// Value above hist_max clamps to x_right.
#[test]
fn wl_to_x_above_range_clamps_to_right() {
    let x = wl_to_x(110.0, 0.0, 100.0, 10.0, 110.0);
    assert!(
        (x - 110.0).abs() < 1e-4,
        "expected x=110.0 (clamped), got {x}"
    );
}
