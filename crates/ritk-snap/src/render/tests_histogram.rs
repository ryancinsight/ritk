use super::*;

// ── compute_histogram tests ───────────────────────────────────────────────

/// Uniform data from 0.0 to 255.0 in 256 bins: each bin contains exactly 1.
#[test]
fn uniform_data_256_bins_each_count_one() {
    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let h = compute_histogram(&data, 0.0, 256.0, 256);
    assert_eq!(h.bins, 256);
    assert_eq!(h.counts.len(), 256);
    for (i, &c) in h.counts.iter().enumerate() {
        assert_eq!(c, 1, "bin {i} expected count 1, got {c}");
    }
}

/// All values equal to min are placed in bin 0.
#[test]
fn all_values_at_min_placed_in_bin_zero() {
    let data = vec![0.0f32; 100];
    let h = compute_histogram(&data, 0.0, 100.0, 10);
    assert_eq!(h.counts[0], 100, "all 100 values at min must be in bin 0");
    for i in 1..10 {
        assert_eq!(h.counts[i], 0, "bin {i} must be empty");
    }
}

/// Values exactly at max (= 100.0) clamp to the last bin.
#[test]
fn values_at_max_clamp_to_last_bin() {
    let data = vec![100.0f32; 50];
    let h = compute_histogram(&data, 0.0, 100.0, 10);
    // raw_f = (100.0 - 0.0) / 10.0 = 10.0 ≥ bins(10) → clamped to bin 9
    assert_eq!(h.counts[9], 50, "all 50 values at max must be in last bin");
    for i in 0..9 {
        assert_eq!(h.counts[i], 0, "bin {i} must be empty");
    }
}

/// Values below min clamp to bin 0.
#[test]
fn below_min_clamped_to_bin_zero() {
    let data = vec![-50.0f32, 5.0, 5.0];
    let h = compute_histogram(&data, 0.0, 10.0, 2);
    // -50.0 → raw_f < 0 → bin 0
    // 5.0 → raw_f = 1.0 → bin 1
    assert_eq!(h.counts[0], 1, "value below min must be in bin 0");
    assert_eq!(h.counts[1], 2, "5.0 values must be in bin 1");
}

/// Values above max clamp to the last bin.
#[test]
fn above_max_clamped_to_last_bin() {
    let data = vec![200.0f32, 5.0];
    let h = compute_histogram(&data, 0.0, 10.0, 2);
    // 200.0 → raw_f ≥ 2 → bin 1
    // 5.0 → raw_f = 1.0 → bin 1
    assert_eq!(h.counts[0], 0, "bin 0 must be empty");
    assert_eq!(h.counts[1], 2, "both values must be in last bin");
}

/// Empty input produces an all-zero count vector.
#[test]
fn empty_data_produces_all_zero_counts() {
    let h = compute_histogram(&[], 0.0, 100.0, 10);
    assert_eq!(h.bins, 10);
    assert_eq!(h.counts.len(), 10);
    assert!(
        h.counts.iter().all(|&c| c == 0),
        "all counts must be zero for empty input"
    );
}

/// Two equal-sized bins, exact half-split at bin boundary.
///
/// Analytical: w=5.0; values 0..=4 fall in [0,5) → bin 0;
/// values 5..=9 fall in [5,10) → bin 1.
#[test]
fn two_bin_exact_half_split() {
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let h = compute_histogram(&data, 0.0, 10.0, 2);
    assert_eq!(h.counts[0], 5, "values [0,5) → bin 0");
    assert_eq!(h.counts[1], 5, "values [5,10) → bin 1");
}

/// Degenerate range (max == min) returns an empty histogram.
#[test]
fn degenerate_max_equals_min_returns_empty() {
    let h = compute_histogram(&[1.0, 2.0, 3.0], 5.0, 5.0, 10);
    assert_eq!(h.bins, 0, "degenerate range must produce 0 bins");
    assert!(
        h.counts.is_empty(),
        "counts must be empty for degenerate range"
    );
}

/// `histogram_bin_center` returns analytically correct center for each bin.
///
/// w = (10.0 − 0.0) / 4 = 2.5; center(i) = 0.0 + (i + 0.5) × 2.5
#[test]
fn bin_center_matches_analytical_formula() {
    let h = compute_histogram(&[1.0f32], 0.0, 10.0, 4);
    // w = 2.5; centers = 1.25, 3.75, 6.25, 8.75
    let expected = [1.25f32, 3.75, 6.25, 8.75];
    for (i, &exp) in expected.iter().enumerate() {
        let got = histogram_bin_center(&h, i);
        assert!(
            (got - exp).abs() < 1e-5,
            "bin {i}: expected center {exp}, got {got}"
        );
    }
}

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
