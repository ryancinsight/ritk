//! Sigma-invariance and broad-sigma property tests for the direct Parzen path.

use super::super::types::ParzenConfig;
use super::super::*;

#[test]
fn direct_parzen_config_sigma_invariant() {
    // Two different sigma² values that produce the same half_width must produce
    // histograms with positive, finite totals. Both sigma_sq=0.9 and 1.0 yield
    // half_width=3 but different weight magnitudes.
    let num_bins = 32;
    let n = 10;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 2.5 + 0.5) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 1.8 + 1.0) % 30.0).collect();

    // sigma_sq=0.9 → half_width=ceil(3*0.949)=3
    // sigma_sq=1.0 → half_width=ceil(3*1.0)=3
    // Both have half_width=3 but different inv_2sigma_sq
    let cfg_09 = ParzenConfig::new(0.9);
    let cfg_10 = ParzenConfig::new(1.0);
    assert_eq!(cfg_09.half_width(), cfg_10.half_width());

    let hist_09 =
        compute_joint_histogram_direct(&fixed_vec, &moving_vec, num_bins, 0.9, 0.9, None, None);
    let hist_10 =
        compute_joint_histogram_direct(&fixed_vec, &moving_vec, num_bins, 1.0, 1.0, None, None);
    let slice_09 = hist_09.as_slice::<f32>().unwrap();
    let slice_10 = hist_10.as_slice::<f32>().unwrap();

    // Both totals must be positive and finite.
    let sum_09: f32 = slice_09.iter().sum();
    let sum_10: f32 = slice_10.iter().sum();
    assert!(
        sum_09 > 0.0 && sum_09.is_finite(),
        "sigma_sq=0.9 total must be positive and finite, got {sum_09}"
    );
    assert!(
        sum_10 > 0.0 && sum_10.is_finite(),
        "sigma_sq=1.0 total must be positive and finite, got {sum_10}"
    );
    // PERF-328-01: per-sample normalization by 1/(sum_f × sum_m) makes the
    // histogram total σ²-invariant. Both σ²=0.9 and σ²=1.0 yield totals ≈ n.
    let total_rel_err = (sum_09 - sum_10).abs() / n as f32;
    assert!(
        total_rel_err < 0.10,
        "normalized totals must be σ²-invariant: sum_09={sum_09}, sum_10={sum_10}, rel_err={total_rel_err}"
    );
}

// ─── Phase Five property tests (TEST-318-06) ───────────────────────────────

#[test]
fn direct_broad_sigma_produces_valid_histogram() {
    // Broad sigma (sigma_sq=4.0, σ=2 bins, half_width=6) should produce a
    // valid histogram. Raw weights scale with σ²: each interior sample
    // contributes ~√(2π×4)² = 8π ≈ 25.1 total. With n=50, use wide bounds.
    let num_bins = 32;
    let sigma_sq = 4.0_f32;
    let n = 50;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 5.0) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 3.0) % 30.0).collect();

    let hist_data = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
        None,
        None,
    );
    let slice = hist_data.as_slice::<f32>().unwrap();

    // All entries must be non-negative
    for (i, &v) in slice.iter().enumerate() {
        assert!(v >= 0.0, "histogram entry {i} is negative: {v}");
    }

    // Total weight must be positive and finite
    let total: f32 = slice.iter().sum();
    assert!(total > 0.0, "total weight must be positive, got {total}");
    assert!(
        total.is_finite(),
        "total weight must be finite, got {total}"
    );

    // PERF-328-01: per-sample normalization by 1/(sum_f × sum_m) means each
    // sample contributes ≈ 1.0 to the histogram total regardless of σ².
    // For n=50 with minor boundary truncation, total should be in [0.3n, 1.5n].
    let expected_min = n as f32 * 0.3;
    let expected_max = n as f32 * 1.5;
    assert!(
        total > expected_min,
        "normalized total weight {total} should be > n × 0.3 = {expected_min}"
    );
    assert!(
        total < expected_max,
        "normalized total weight {total} should be < n × 1.5 = {expected_max}"
    );
}

#[test]
fn direct_broad_sigma_matches_sparse_cache() {
    // SPARSE-329-01: Both direct and sparse paths apply full joint normalization.
    // At σ²=4, each interior sample contributes ≈1.0 (normalized). Broad sigma
    // causes larger boundary truncation. Verify (1) nonzero patterns match,
    // (2) totals are approximately equal (ratio ≈ 1.0).
    let num_bins = 32;
    let sigma_sq = 4.0_f32;
    let n = 100;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.42 + 0.1) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n)
        .map(|i| ((i * 7 + 3) as f32 * 0.031) % 30.0)
        .collect();

    let direct_data = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
        None,
        None,
    );
    let direct_slice = direct_data.as_slice::<f32>().unwrap();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed_vec, num_bins, sigma_sq, None);
    let sparse_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving_vec,
        num_bins,
        sigma_sq,
        None,
        None,
    );
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    // Structural check: nonzero pattern must match.
    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let d_nz = *d > 1e-10;
        let s_nz = *s > 1e-10;
        assert_eq!(
            d_nz, s_nz,
            "broad-sigma nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }

    // Totals: SPARSE-329-01 makes sparse and direct fully equivalent after
    // normalization. Both totals ≈ n for interior samples. σ²=4 (broad)
    // gives larger boundary truncation, so totals are slightly below n.
    let direct_total: f32 = direct_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    assert!(
        direct_total > 0.0,
        "direct histogram should have positive sum"
    );
    assert!(
        sparse_total > 0.0,
        "sparse histogram should have positive sum"
    );
    assert!(
        direct_total > n as f32 * 0.3,
        "direct total {direct_total} should be > n × 0.3 (broad σ boundary clipping)"
    );
    assert!(
        direct_total < n as f32 * 1.5,
        "direct total {direct_total} should be < n × 1.5"
    );
    // SPARSE-329-01: direct ≈ sparse (combined normalization). Ratio ≈ 1.0.
    let ratio = sparse_total / direct_total;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "sparse_total/direct_total ratio {ratio} should be ≈ 1.0 (SPARSE-329-01)"
    );
}
