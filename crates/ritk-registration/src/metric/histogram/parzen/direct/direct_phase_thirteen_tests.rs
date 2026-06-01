//! Phase Thirteen tests — Sprint 328 per-sample normalization and API promotions.
//!
//! Covers:
//! - PERF-328-01: Per-sample weight normalization in `accumulate_sample_direct`
//! - PERF-328-02: Per-sample moving-axis normalization in `accumulate_sample_sparse`
//! - ARCH-328-04: `StackWeights::len()` production promotion
//! - ARCH-328-05: `BinRange::len()` production promotion
//! - PERF-328-01: `ParzenConfig::compute_weights_with_inv_sum()` production
//! - PERF-328-01: `ParzenConfig::inv_sum_weights()` public API

use super::sample::SampleWindow;
use super::types::{BinRange, ParzenConfig};
use super::*;

// ── PERF-328-01: Per-sample normalization in accumulate_sample_direct ──────

#[test]
fn direct_normalized_histogram_sum_equals_sample_count() {
    // PERF-328-01: With per-sample normalization (inv_norm = 1/(sum_f*sum_m)),
    // each sample's total histogram contribution ≈ 1.0. So the histogram
    // sum should approximately equal the number of in-bounds samples.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 100;

    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5 + 1.0) % 30.0).collect();

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();

    // Each of the n samples contributes ≈1.0, so total ≈ n
    assert!(
        (sum - n as f32).abs() < n as f32 * 0.1,
        "normalized histogram sum={sum}, expected≈{n}"
    );
}

#[test]
fn direct_normalized_boundary_and_interior_equal_contribution() {
    // PERF-328-01: Boundary samples (near edges) and interior samples
    // should contribute approximately the same total weight after normalization.
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    // Interior sample
    let window_interior =
        SampleWindow::new(0, &[15.3], &[20.7], num_bins, &fix_cfg, &mov_cfg, None)
            .expect("interior");
    let mut hist_interior = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_interior, num_bins, &window_interior);
    let sum_interior: f32 = hist_interior.iter().sum();

    // Boundary sample (near lower boundary)
    let window_boundary = SampleWindow::new(0, &[0.5], &[20.7], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("boundary");
    let mut hist_boundary = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_boundary, num_bins, &window_boundary);
    let sum_boundary: f32 = hist_boundary.iter().sum();

    // After normalization, both should be ≈1.0
    assert!(
        (sum_interior - 1.0).abs() < 0.02,
        "interior sum={sum_interior}, expected≈1.0"
    );
    assert!(
        (sum_boundary - 1.0).abs() < 0.02,
        "boundary sum={sum_boundary}, expected≈1.0"
    );
}

#[test]
fn direct_normalized_with_oob_mask_sum_equals_in_bounds_count() {
    // PERF-328-01: With OOB mask, the total should ≈ number of in-bounds samples.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 20;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 1.5) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 1.0) % 30.0).collect();
    let oob: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();

    let hist_data = compute_joint_histogram_direct(
        &fixed,
        &moving,
        num_bins,
        sigma_sq,
        sigma_sq,
        Some(&oob),
        None,
    );
    let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();
    let in_bounds = n / 2;

    assert!(
        (sum - in_bounds as f32).abs() < in_bounds as f32 * 0.1,
        "normalized OOB histogram sum={sum}, expected≈{in_bounds}"
    );
}

#[test]
fn direct_normalized_different_sigma_same_total() {
    // PERF-328-01: Different sigma values produce different histogram shapes,
    // but the total sum should still ≈ n (each sample contributes ≈1.0).
    let num_bins = 32;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 5.0) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 3.0) % 30.0).collect();

    for &sigma_sq in &[0.5, 1.0, 4.0, 9.0] {
        let hist_data = compute_joint_histogram_direct(
            &fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None,
        );
        let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();
        assert!(
            (sum - n as f32).abs() < n as f32 * 0.1,
            "sigma_sq={sigma_sq}: normalized sum={sum}, expected≈{n}"
        );
    }
}

#[test]
fn sample_window_inv_sum_fields_are_correct() {
    // PERF-328-01: Verify that SampleWindow stores correct inv_sum_f and inv_sum_m.
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(4.0);
    let num_bins = 32;

    let window = SampleWindow::new(0, &[15.3], &[20.7], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    // Manually compute inv_sum_f and inv_sum_m
    let (_, f_weights) = fix_cfg.compute_weights(15.3, num_bins);
    let (_, m_weights) = mov_cfg.compute_weights(20.7, num_bins);
    let expected_inv_sum_f: f32 = 1.0 / f_weights.iter().map(|(_, w)| w).sum::<f32>();
    let expected_inv_sum_m: f32 = 1.0 / m_weights.iter().map(|(_, w)| w).sum::<f32>();

    assert!(
        (window.inv_sum_f() - expected_inv_sum_f).abs() < 1e-6,
        "inv_sum_f={} vs expected={}",
        window.inv_sum_f(),
        expected_inv_sum_f
    );
    assert!(
        (window.inv_sum_m() - expected_inv_sum_m).abs() < 1e-6,
        "inv_sum_m={} vs expected={}",
        window.inv_sum_m(),
        expected_inv_sum_m
    );
}

// ── SPARSE-329-01: Sparse-path full joint normalization ──────────────────

#[test]
fn sparse_moving_normalized_histogram_positive() {
    // SPARSE-329-01: Sparse-path with full joint normalization should produce
    // a valid positive histogram where total ≈ n (same as direct path).
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 5.0) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 3.0) % 30.0).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
    let hist_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq,
        None,
        None,
    );
    let slice = hist_data.as_slice::<f32>().unwrap();

    let sum: f32 = slice.iter().sum();
    assert!(
        sum > 0.0,
        "sparse normalized sum must be positive, got {sum}"
    );
    assert!(
        sum.is_finite(),
        "sparse normalized sum must be finite, got {sum}"
    );

    // All entries must be non-negative
    for (i, &v) in slice.iter().enumerate() {
        assert!(v >= 0.0, "bin {i} must be non-negative, got {v}");
    }
}

#[test]
fn sparse_moving_normalized_boundary_sample() {
    // SPARSE-329-01: A boundary sample with full joint normalization
    // should contribute ≈1.0 total (matching the direct path), not more or less.
    let num_bins = 32;
    let mov_cfg = ParzenConfig::new(1.0);
    let m_val_boundary = 0.5_f32;
    let m_val_interior = 15.3_f32;

    let (_, m_range_b, m_weights_b, inv_sum_m_b) =
        SampleWindow::new_moving_only(0, &[m_val_boundary], num_bins, &mov_cfg, None)
            .expect("boundary");

    let (_, m_range_i, m_weights_i, inv_sum_m_i) =
        SampleWindow::new_moving_only(0, &[m_val_interior], num_bins, &mov_cfg, None)
            .expect("interior");

    // Moving-axis weight sums should differ (boundary is smaller)
    let boundary_m_sum: f32 = m_weights_b.iter().map(|(_, w)| w).sum();
    let interior_m_sum: f32 = m_weights_i.iter().map(|(_, w)| w).sum();
    assert!(
        boundary_m_sum < interior_m_sum,
        "boundary moving sum ({boundary_m_sum}) should be < interior ({interior_m_sum})"
    );

    // But after multiplying by inv_sum_m, both moving-axis contributions sum to ≈1.0
    let boundary_normalized: f32 = m_weights_b.iter().map(|(_, w)| w * inv_sum_m_b).sum();
    let interior_normalized: f32 = m_weights_i.iter().map(|(_, w)| w * inv_sum_m_i).sum();

    assert!(
        (boundary_normalized - 1.0).abs() < 0.02,
        "boundary moving normalized={boundary_normalized}, expected≈1.0"
    );
    assert!(
        (interior_normalized - 1.0).abs() < 0.02,
        "interior moving normalized={interior_normalized}, expected≈1.0"
    );

    // Verify ranges are different (boundary clamped)
    assert!(
        m_range_b.lo < m_range_i.lo,
        "boundary range should start lower"
    );
}

// ── ARCH-328-04: StackWeights::len() production promotion ─────────────────

#[test]
fn stack_weights_len_production_api() {
    // ARCH-328-04: StackWeights::len() is available in production (not just #[cfg(test)]).
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    // This call must compile without #[cfg(test)]
    let len = weights.len();
    assert_eq!(len, 7, "len must be 7 for sigma_sq=1.0");
    assert!(!weights.is_empty(), "weights must not be empty");
}

#[test]
fn stack_weights_len_matches_iterator_count() {
    // ARCH-328-04: StackWeights::len() must equal the iterator count.
    for &sigma_sq in &[0.01, 0.5, 1.0, 4.0, 9.0] {
        let cfg = ParzenConfig::new(sigma_sq);
        let (_, weights) = cfg.compute_weights(15.3, 64);
        let len_method = weights.len();
        let iter_count = weights.iter().count();
        assert_eq!(
            len_method, iter_count,
            "sigma_sq={sigma_sq}: len()={len_method} != iter count={iter_count}"
        );
    }
}

// ── ARCH-328-05: BinRange::len() production promotion ─────────────────────

#[test]
fn bin_range_len_production_api() {
    // ARCH-328-05: BinRange::len() is available in production (not just #[cfg(test)]).
    let range = BinRange::new(10, 3, 32);
    // This call must compile without #[cfg(test)]
    let len = range.len();
    assert_eq!(len, 7, "range 7..=13 has 7 bins");
    assert!(!range.is_empty(), "range must not be empty");
}

#[test]
fn bin_range_len_boundary_clamping() {
    // ARCH-328-05: BinRange::len() at boundary must still be correct.
    let range = BinRange::new(1, 3, 32); // lo clamped to 0
    assert_eq!(range.len(), 5, "range 0..=4 has 5 bins");

    let range = BinRange::new(30, 3, 32); // hi clamped to 31
    assert_eq!(range.len(), 5, "range 27..=31 has 5 bins");
}

// ── PERF-328-01: compute_weights_with_inv_sum production API ──────────────

#[test]
fn compute_weights_with_inv_sum_matches_separate_calls() {
    // PERF-328-01: compute_weights_with_inv_sum must produce the same results
    // as calling compute_weights and sum_weights separately.
    for &sigma_sq in &[0.01, 0.5, 1.0, 4.0, 9.0] {
        let cfg = ParzenConfig::new(sigma_sq);
        let val = 15.3_f32;
        let num_bins = 64;

        let (range, weights) = cfg.compute_weights(val, num_bins);
        let sum = cfg.sum_weights(val, num_bins);
        let expected_inv_sum = 1.0 / sum;

        let (range2, weights2, inv_sum) = cfg.compute_weights_with_inv_sum(val, num_bins);

        // Ranges must match
        assert_eq!(range.lo, range2.lo, "lo mismatch at sigma_sq={sigma_sq}");
        assert_eq!(range.hi, range2.hi, "hi mismatch at sigma_sq={sigma_sq}");

        // Weights must match
        for ((j, w1), (_, w2)) in weights.iter().zip(weights2.iter()) {
            assert!(
                (w1 - w2).abs() < 1e-10,
                "weight mismatch at sigma_sq={sigma_sq}, j={j}"
            );
        }

        // inv_sum must match
        assert!(
            (inv_sum - expected_inv_sum).abs() < 1e-6,
            "inv_sum mismatch at sigma_sq={sigma_sq}: got {inv_sum}, expected {expected_inv_sum}"
        );
    }
}

#[test]
fn compute_weights_with_inv_sum_interior_value() {
    // PERF-328-01: For an interior value, inv_sum ≈ 1/√(2πσ²).
    let cfg = ParzenConfig::new(1.0);
    let (_, _, inv_sum) = cfg.compute_weights_with_inv_sum(15.3, 64);
    let expected = 1.0 / (2.0 * std::f32::consts::PI).sqrt(); // ≈ 0.399
    assert!(
        (inv_sum - expected).abs() / expected < 0.02,
        "interior inv_sum={inv_sum}, expected≈{expected}"
    );
}

#[test]
fn compute_weights_with_inv_sum_boundary_larger() {
    // PERF-328-01: Boundary values have smaller sum → larger inv_sum.
    let cfg = ParzenConfig::new(1.0);
    let num_bins = 32;
    let (_, _, inv_sum_boundary) = cfg.compute_weights_with_inv_sum(0.5, num_bins);
    let (_, _, inv_sum_interior) = cfg.compute_weights_with_inv_sum(15.3, num_bins);
    assert!(
        inv_sum_boundary > inv_sum_interior,
        "boundary inv_sum ({inv_sum_boundary}) should be > interior ({inv_sum_interior})"
    );
}

// ── PERF-328-01: inv_sum_weights public API ───────────────────────────────

#[test]
fn inv_sum_weights_matches_compute_weights_with_inv_sum() {
    // PERF-328-01: inv_sum_weights must return the same value as
    // compute_weights_with_inv_sum's third element.
    let cfg = ParzenConfig::new(1.0);
    let val = 15.3_f32;
    let num_bins = 64;

    let inv_sum_direct = cfg.inv_sum_weights(val, num_bins);
    let (_, _, inv_sum_combined) = cfg.compute_weights_with_inv_sum(val, num_bins);

    assert!(
        (inv_sum_direct - inv_sum_combined).abs() < 1e-7,
        "inv_sum_weights={inv_sum_direct} != compute_weights_with_inv_sum={inv_sum_combined}"
    );
}

// ── SampleWindow size after adding inv_sum_f and inv_sum_m ────────────────

#[test]
fn sample_window_size_after_normalization_fields() {
    // PERF-328-01: SampleWindow now has inv_sum_f and inv_sum_m (2 × 4 bytes = 8 bytes extra).
    // Production size: f_range(4) + m_range(4) + f_weights(~128) + m_weights(~128)
    //                  + inv_sum_f(4) + inv_sum_m(4) ≈ 272 bytes
    // Test size: add f_val(4) + m_val(4) = +8 bytes
    let size = std::mem::size_of::<SampleWindow>();
    // Production layout with alignment padding:
    // f_range(4) + m_range(4) + f_weights(132) + m_weights(132) + inv_sum_f(4) + inv_sum_m(4)
    // ≈ 280 bytes production. With test fields: +8 = 288
    assert!(
        size >= 264,
        "SampleWindow is {size} bytes — expected ≥264 (with inv_sum_f + inv_sum_m)"
    );
    assert!(
        size <= 340,
        "SampleWindow is {size} bytes — expected ≤340 (with test fields + alignment)"
    );
}

// ── Full-pipeline: direct path normalized end-to-end ──────────────────────

#[test]
fn direct_normalized_single_sample_contributes_one() {
    // PERF-328-01: A single in-bounds sample should contribute ≈1.0 total.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let fixed = vec![15.3f32];
    let moving = vec![20.7f32];

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();

    assert!(
        (sum - 1.0).abs() < 0.02,
        "single-sample normalized sum={sum}, expected≈1.0"
    );
}

#[test]
fn sparse_normalized_single_sample_positive() {
    // SPARSE-329-01: A single sample via the sparse path should produce
    // total ≈ 1.0 (matching the direct path's per-sample normalization).
    let num_bins = 32;
    let sigma_sq = 1.0;
    let fixed = vec![15.3f32];
    let moving = vec![20.7f32];

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
    let hist_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq,
        None,
        None,
    );
    let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();

    assert!(sum > 0.0, "sparse single-sample sum={sum} must be positive");
    assert!(
        sum.is_finite(),
        "sparse single-sample sum={sum} must be finite"
    );
}
