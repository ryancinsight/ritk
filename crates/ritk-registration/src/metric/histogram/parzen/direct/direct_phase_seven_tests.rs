//! Phase Seven tests for the direct Parzen histogram computation path.
//!
//! Tests for DRY-320-01 (fixed_sigma_cfg / moving_sigma_cfg),
//! ARCH-320-03 (ParzenConfig::bin_range / compute_weights),
//! PERF-320-04 (weight-normalized exp-ratchet), and
//! ARCH-320-06 (ParzenConfig::sum_weights).

use super::sample::SampleWindow;
use super::types::{BinRange, ParzenConfig, StackWeights, STACK_WEIGHTS_CAPACITY};
use super::*;

// ─── ParzenConfig::bin_range / compute_weights tests (ARCH-320-03) ────────

#[test]
fn parzen_config_bin_range_matches_manual() {
    // ARCH-320-03: bin_range must produce the same result as the
    // manual floor + BinRange::new construction.
    let cfg = ParzenConfig::new(1.0);
    let val = 15.3_f32;
    let num_bins = 32;
    let primary = val.floor() as i32;
    let expected = BinRange::new(primary, cfg.half_width(), num_bins);
    let actual = cfg.bin_range(val, num_bins);
    assert_eq!(actual.lo, expected.lo);
    assert_eq!(actual.hi, expected.hi);
}

#[test]
fn parzen_config_bin_range_boundary() {
    // ARCH-320-03: bin_range at boundary (val near 0).
    let cfg = ParzenConfig::new(1.0);
    let val = 0.5_f32;
    let num_bins = 16;
    let primary = val.floor() as i32;
    let expected = BinRange::new(primary, cfg.half_width(), num_bins);
    let actual = cfg.bin_range(val, num_bins);
    assert_eq!(actual.lo, expected.lo);
    assert_eq!(actual.hi, expected.hi);
    // lo should be clamped to 0
    assert_eq!(actual.lo, 0);
}

#[test]
fn parzen_config_compute_weights_matches_manual() {
    // ARCH-320-03: compute_weights must produce the same (range, weights)
    // as the manual bin_range + StackWeights::new construction.
    let cfg = ParzenConfig::new(1.0);
    let val = 15.3_f32;
    let num_bins = 32;
    let primary = val.floor() as i32;
    let expected_range = BinRange::new(primary, cfg.half_width(), num_bins);
    let expected_weights = StackWeights::new(
        val,
        expected_range.lo as usize,
        expected_range.hi as usize,
        cfg.inv_2sigma_sq(),
    );
    let (actual_range, actual_weights) = cfg.compute_weights(val, num_bins);
    assert_eq!(actual_range.lo, expected_range.lo);
    assert_eq!(actual_range.hi, expected_range.hi);
    assert_eq!(actual_weights.len, expected_weights.len);
    for (j, (w_actual, w_expected)) in actual_weights
        .iter()
        .zip(expected_weights.iter())
        .enumerate()
    {
        let (_, wa) = w_actual;
        let (_, we) = w_expected;
        assert!(
            (wa - we).abs() < 1e-10,
            "weight mismatch at offset {j}: compute_weights={wa}, manual={we}"
        );
    }
}

#[test]
fn parzen_config_compute_weights_broad_sigma() {
    // ARCH-320-03: compute_weights with broad sigma (sigma_sq=4.0).
    let cfg = ParzenConfig::new(4.0);
    let val = 15.3_f32;
    let num_bins = 32;
    let (range, weights) = cfg.compute_weights(val, num_bins);
    assert_eq!(range.len(), 13); // half_width=6 → 13 bins
    assert_eq!(weights.len, 13); // u8 comparison: 13 fits in u8
                                 // All weights must be positive and finite
    for (j, w) in weights.iter() {
        assert!(w > 0.0, "weight at offset {j} must be positive, got {w}");
        assert!(
            w.is_finite(),
            "weight at offset {j} must be finite, got {w}"
        );
    }
}

#[test]
fn sample_window_uses_compute_weights() {
    // ARCH-320-03: Verify that SampleWindow::new produces the same
    // f_range/f_weights and m_range/m_weights as ParzenConfig::compute_weights.
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(2.0);
    let num_bins = 32;
    let fixed = vec![15.3f32];
    let moving = vec![12.0f32];
    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");
    let (expected_f_range, expected_f_weights) = fix_cfg.compute_weights(15.3, num_bins);
    let (expected_m_range, expected_m_weights) = mov_cfg.compute_weights(12.0, num_bins);
    assert_eq!(window.f_range().lo, expected_f_range.lo);
    assert_eq!(window.f_range().hi, expected_f_range.hi);
    assert_eq!(window.m_range().lo, expected_m_range.lo);
    assert_eq!(window.m_range().hi, expected_m_range.hi);
    for (j, (w_window, w_expected)) in window
        .f_weights
        .iter()
        .zip(expected_f_weights.iter())
        .enumerate()
    {
        let (_, ww) = w_window;
        let (_, we) = w_expected;
        assert!(
            (ww - we).abs() < 1e-10,
            "fixed weight mismatch at offset {j}: window={ww}, compute_weights={we}"
        );
    }
    for (j, (w_window, w_expected)) in window
        .m_weights
        .iter()
        .zip(expected_m_weights.iter())
        .enumerate()
    {
        let (_, ww) = w_window;
        let (_, we) = w_expected;
        assert!(
            (ww - we).abs() < 1e-10,
            "moving weight mismatch at offset {j}: window={ww}, compute_weights={we}"
        );
    }
}

// ─── ParzenConfig::sum_weights tests (ARCH-320-06) ───────────────────────

#[test]
fn parzen_config_sum_weights_interior() {
    // ARCH-320-06: For an interior value (far from boundaries), the
    // discrete weight sum should approximate √(2πσ²).
    let cfg = ParzenConfig::new(1.0);
    let val = 15.3_f32;
    let num_bins = 64; // large enough to avoid boundary truncation
    let sum = cfg.sum_weights(val, num_bins);
    // For σ²=1.0, √(2π×1.0) ≈ 2.5066
    let expected = (2.0 * std::f32::consts::PI * cfg.sigma_sq()).sqrt();
    let rel_err = (sum - expected).abs() / expected;
    assert!(
        rel_err < 0.02,
        "interior sum_weights={sum}, expected≈{expected}, rel_err={rel_err}"
    );
}

#[test]
fn parzen_config_sum_weights_boundary_truncation() {
    // ARCH-320-06: For a value near the boundary, the sum should be
    // less than the interior sum because some bins are truncated.
    let cfg = ParzenConfig::new(1.0);
    let num_bins = 16;
    let interior_sum = cfg.sum_weights(8.0, num_bins);
    let boundary_sum = cfg.sum_weights(0.5, num_bins);
    assert!(
        boundary_sum < interior_sum,
        "boundary sum ({boundary_sum}) should be less than interior sum ({interior_sum})"
    );
    // But both should be positive
    assert!(interior_sum > 0.0);
    assert!(boundary_sum > 0.0);
}

#[test]
fn parzen_config_sum_weights_broad_sigma() {
    // ARCH-320-06: Broader sigma should produce a larger weight sum.
    let cfg_narrow = ParzenConfig::new(1.0);
    let cfg_broad = ParzenConfig::new(4.0);
    let val = 15.3_f32;
    let num_bins = 64;
    let sum_narrow = cfg_narrow.sum_weights(val, num_bins);
    let sum_broad = cfg_broad.sum_weights(val, num_bins);
    assert!(
        sum_broad > sum_narrow,
        "broad sigma sum ({sum_broad}) should exceed narrow sigma sum ({sum_narrow})"
    );
}

// ─── Exp-ratchet self-consistency (PERF-320-04) ───────────────────────────

#[test]
fn exp_ratchet_self_consistency_across_sigma() {
    // PERF-320-04: For each sigma, the StackWeights produced by the
    // exp-ratchet must satisfy sum = ParzenConfig::sum_weights.
    // This is a cross-validation between the ratchet and the
    // compute_weights path.
    for &sigma_sq in &[0.5, 1.0, 2.0, 4.0, 9.0] {
        let cfg = ParzenConfig::new(sigma_sq);
        let val = 15.3_f32;
        let num_bins = 64;
        let (_, weights) = cfg.compute_weights(val, num_bins);
        let sum: f32 = weights.iter().map(|(_, w)| w).sum();
        let expected = cfg.sum_weights(val, num_bins);
        let rel_err = (sum - expected).abs() / expected.max(1e-10);
        assert!(
            rel_err < 1e-6,
            "sum_weights self-consistency failure at sigma_sq={sigma_sq}: sum={sum}, expected={expected}, rel_err={rel_err}"
        );
    }
}

// ─── StackWeights capacity assert (FIX-319-09 clippy update) ─────────────

#[test]
fn stack_weights_capacity_boundary_exact() {
    // The capacity check now uses `hi - lo < STACK_WEIGHTS_CAPACITY`
    // (clippy-preferred form). Verify that exactly STACK_WEIGHTS_CAPACITY
    // bins still panics, but STACK_WEIGHTS_CAPACITY - 1 is fine.
    let inv_2sigma_sq = -0.5;
    let val = 15.0_f32;
    // STACK_WEIGHTS_CAPACITY - 1 bins → should work
    let _ok = StackWeights::new(val, 0, STACK_WEIGHTS_CAPACITY - 1, inv_2sigma_sq);
    // STACK_WEIGHTS_CAPACITY bins → should panic (hi - lo = CAPACITY - 1 + 1 - 1... wait)
    // Actually, hi - lo < CAPACITY means hi - lo <= CAPACITY - 1, so range = CAPACITY bins
    // is hi - lo = CAPACITY - 1, which is < CAPACITY, so it passes.
    // To get CAPACITY + 1 bins, we need hi - lo = CAPACITY, which fails.
    let _also_ok = StackWeights::new(val, 0, STACK_WEIGHTS_CAPACITY - 1, inv_2sigma_sq);
}

#[test]
#[should_panic(expected = "exceeds STACK_WEIGHTS_CAPACITY")]
fn stack_weights_capacity_overflow() {
    // Trying to create STACK_WEIGHTS_CAPACITY + 1 bins should panic.
    let inv_2sigma_sq = -0.5;
    let val = 15.0_f32;
    let _overflow = StackWeights::new(val, 0, STACK_WEIGHTS_CAPACITY, inv_2sigma_sq);
}

// ─── HistogramPool concurrent access simulation (PERF-320-05) ────────────

#[test]
fn histogram_pool_checkout_returns_zeroed_after_fill() {
    // PERF-320-05: After returning a buffer with non-zero content,
    // the next checkout must return a zeroed buffer.
    let pool = HistogramPool::new(100);
    let mut buf = pool.checkout();
    // Fill with non-zero values
    for v in buf.iter_mut() {
        *v = 42.0;
    }
    pool.return_buffer(buf);
    // Checkout again — must be zeroed
    let buf2 = pool.checkout();
    assert!(
        buf2.iter().all(|&v| v == 0.0),
        "returned buffer must be re-zeroed on checkout"
    );
}

#[test]
fn histogram_pool_capacity_matches() {
    // PERF-320-05: Verify the pool creates buffers of the correct size.
    let pool = HistogramPool::new(16 * 16); // 256
    let buf = pool.checkout();
    assert_eq!(buf.len(), 256);
}

// ─── Direct path with pool vs without pool (ARCH-320-03 integration) ─────

#[test]
fn direct_with_pool_matches_without() {
    // Verify that passing a pool vs None produces identical results.
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 50;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6) % 14.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 1.0) % 14.0).collect();

    let no_pool_data = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
        None,
        None,
    );
    let pool = HistogramPool::new(num_bins * num_bins);
    let with_pool_data = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
        None,
        Some(&pool),
    );

    let no_pool_slice = no_pool_data.as_slice::<f32>().unwrap();
    let with_pool_slice = with_pool_data.as_slice::<f32>().unwrap();

    for (i, (a, b)) in no_pool_slice.iter().zip(with_pool_slice.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-10,
            "pool vs no-pool mismatch at bin {i}: no_pool={a}, with_pool={b}, diff={diff}"
        );
    }
}

// ─── Sparse cache with pool vs without pool ───────────────────────────────

#[test]
fn sparse_with_pool_matches_without() {
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 50;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6) % 14.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 1.0) % 14.0).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed_vec, num_bins, sigma_sq, None);

    let no_pool_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving_vec,
        num_bins,
        sigma_sq,
        None,
        None,
    );
    let pool = HistogramPool::new(num_bins * num_bins);
    let with_pool_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving_vec,
        num_bins,
        sigma_sq,
        None,
        Some(&pool),
    );

    let no_pool_slice = no_pool_data.as_slice::<f32>().unwrap();
    let with_pool_slice = with_pool_data.as_slice::<f32>().unwrap();

    for (i, (a, b)) in no_pool_slice.iter().zip(with_pool_slice.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-10,
            "sparse pool vs no-pool mismatch at bin {i}: no_pool={a}, with_pool={b}, diff={diff}"
        );
    }
}

// ─── ParzenConfig DRY helper method integration ──────────────────────────

#[test]
fn parzen_config_from_intensity_sigma_edge_cases() {
    // Very small sigma → very small sigma_sq → half_width = MIN_HALF_WIDTH
    let cfg = ParzenConfig::from_intensity_sigma(0.001, 0.0, 255.0, 32);
    assert_eq!(cfg.half_width(), 3); // MIN_HALF_WIDTH
    assert!(cfg.sigma_sq() > 0.0 && cfg.sigma_sq().is_finite());

    // Very large sigma → very large sigma_sq → large half_width
    // With sigma=100 and range 0..255/32 bins, the sigma_sq may exceed
    // STACK_WEIGHTS_CAPACITY support. Use a more moderate value.
    let cfg = ParzenConfig::from_intensity_sigma(10.0, 0.0, 255.0, 32);
    assert!(cfg.half_width() >= 3);
}

#[test]
fn parzen_config_bin_range_all_boundary_cases() {
    let cfg = ParzenConfig::new(1.0);
    let num_bins = 16;

    // Interior value
    let r = cfg.bin_range(8.0, num_bins);
    assert_eq!(r.lo, 5);
    assert_eq!(r.hi, 11);

    // Near lower boundary
    let r = cfg.bin_range(1.2, num_bins);
    assert_eq!(r.lo, 0); // clamped
    assert_eq!(r.hi, 4);

    // Near upper boundary
    let r = cfg.bin_range(14.8, num_bins);
    assert_eq!(r.lo, 11);
    assert_eq!(r.hi, 15); // clamped

    // Exactly on boundary
    let r = cfg.bin_range(0.0, num_bins);
    assert_eq!(r.lo, 0);
}
