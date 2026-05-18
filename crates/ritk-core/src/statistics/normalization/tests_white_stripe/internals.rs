use super::super::{empirical_cdf_rank, find_mode_in_range, quantile_sorted, silverman_bandwidth};

// ── Internal: quantile_sorted ─────────────────────────────────────────

#[test]
fn test_quantile_sorted_basic() {
    let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((quantile_sorted(&sorted, 0.0) - 1.0).abs() < 1e-10);
    assert!((quantile_sorted(&sorted, 1.0) - 5.0).abs() < 1e-10);
    assert!((quantile_sorted(&sorted, 0.5) - 3.0).abs() < 1e-10);
    assert!((quantile_sorted(&sorted, 0.25) - 2.0).abs() < 1e-10);
    assert!((quantile_sorted(&sorted, 0.75) - 4.0).abs() < 1e-10);
}

#[test]
fn test_quantile_sorted_single() {
    let sorted = vec![42.0];
    assert!((quantile_sorted(&sorted, 0.0) - 42.0).abs() < 1e-10);
    assert!((quantile_sorted(&sorted, 0.5) - 42.0).abs() < 1e-10);
    assert!((quantile_sorted(&sorted, 1.0) - 42.0).abs() < 1e-10);
}

// ── Internal: silverman_bandwidth ─────────────────────────────────────

#[test]
fn test_silverman_bandwidth_positive() {
    let sorted: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
    let bw = silverman_bandwidth(&sorted);
    assert!(bw > 0.0, "Silverman bandwidth must be > 0, got {bw}");
}

#[test]
fn test_silverman_bandwidth_constant_data() {
    let sorted = vec![5.0; 100];
    let bw = silverman_bandwidth(&sorted);
    // Constant data: sigma=0, IQR=0. Fallback must produce a finite positive value.
    assert!(
        bw > 0.0 && bw.is_finite(),
        "Silverman bandwidth for constant data must be finite positive, got {bw}"
    );
}

// ── Internal: empirical_cdf_rank ──────────────────────────────────────

#[test]
fn test_empirical_cdf_rank_basic() {
    let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // Value 3.0: 3 values ≤ 3.0, so rank = (3 - 0.5) / 5 = 0.5.
    let rank = empirical_cdf_rank(&sorted, 3.0);
    assert!(
        (rank - 0.5).abs() < 1e-10,
        "CDF rank of median must be 0.5, got {rank}"
    );

    let rank_lo = empirical_cdf_rank(&sorted, 0.0);
    assert!(
        rank_lo < 0.1,
        "CDF rank below all values must be near 0, got {rank_lo}"
    );

    // Value above all: 5 values ≤ 6.0 → rank = (5 - 0.5) / 5 = 0.9.
    let rank_hi = empirical_cdf_rank(&sorted, 6.0);
    assert!(
        rank_hi > 0.8,
        "CDF rank above all values must be near 1, got {rank_hi}"
    );
}

// ── Internal: find_mode_in_range ──────────────────────────────────────

#[test]
fn test_find_mode_in_range_basic() {
    let grid = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let density = vec![0.1, 0.2, 0.8, 0.5, 0.3];

    let mode = find_mode_in_range(&grid, &density, 0.0, 1.0);
    assert!(
        (mode - 0.5).abs() < 1e-10,
        "full range mode must be 0.5, got {mode}"
    );

    let mode_upper = find_mode_in_range(&grid, &density, 0.6, 1.0);
    assert!(
        (mode_upper - 0.75).abs() < 1e-10,
        "upper range mode must be 0.75, got {mode_upper}"
    );
}
