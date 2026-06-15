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
