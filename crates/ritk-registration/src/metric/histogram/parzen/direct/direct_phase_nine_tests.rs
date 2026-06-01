//! Phase Nine tests for the direct Parzen histogram computation path.
//!
//! Tests for ARCH-322-03 (ParzenConfig field encapsulation),
//! DEAD-322-02 (dead code gating correctness), TEST-322-05
//! (SampleWindow edge cases), and TEST-322-06 (HistogramPool
//! concurrent stress test).

use super::sample::SampleWindow;
use super::types::ParzenConfig;
use super::*;

// ─── ParzenConfig encapsulation (ARCH-322-03) ─────────────────────────────

#[test]
fn parzen_config_private_fields_cannot_be_constructed_inconsistently() {
    // ARCH-322-03: Fields are private, so the only way to construct
    // a ParzenConfig is via new() or from_intensity_sigma(), which
    // enforce the invariants. This test verifies that the accessors
    // return consistent derived values for various sigma_sq inputs.
    for sigma_sq in [0.01, 0.1, 1.0, 4.0, 16.0, 100.0] {
        let cfg = ParzenConfig::new(sigma_sq);
        // sigma_sq() matches the constructor argument
        assert!(
            (cfg.sigma_sq() - sigma_sq).abs() < 1e-10,
            "sigma_sq() mismatch for {sigma_sq}: got {}",
            cfg.sigma_sq()
        );
        // half_width() is derived correctly via compute_half_width
        assert_eq!(
            cfg.half_width(),
            compute_half_width(sigma_sq),
            "half_width mismatch for sigma_sq={sigma_sq}"
        );
        // inv_2sigma_sq() is -0.5 / sigma_sq
        assert!(
            (cfg.inv_2sigma_sq() - (-0.5 / sigma_sq)).abs() < 1e-10,
            "inv_2sigma_sq mismatch for sigma_sq={sigma_sq}: got {}",
            cfg.inv_2sigma_sq()
        );
    }
}

#[test]
fn parzen_config_from_intensity_sigma_encapsulation() {
    // ARCH-322-03: from_intensity_sigma also produces consistent
    // derived values accessible only through the accessors.
    let cfg = ParzenConfig::from_intensity_sigma(8.0, 0.0, 255.0, 32);
    let sigma_sq = cfg.sigma_sq();
    // All derived values should be consistent with the computed sigma_sq
    assert_eq!(
        cfg.half_width(),
        compute_half_width(sigma_sq),
        "from_intensity_sigma half_width inconsistent"
    );
    assert!(
        (cfg.inv_2sigma_sq() - (-0.5 / sigma_sq)).abs() < 1e-10,
        "from_intensity_sigma inv_2sigma_sq inconsistent"
    );
}

// ─── Dead-code gating correctness (DEAD-322-02) ───────────────────────────
//
// These tests verify that the #[cfg(test)]-gated methods still work
// correctly in test builds. They won't compile in production builds,
// which is the point — dead code is eliminated at the source level.

#[test]
fn bin_range_iter_returns_correct_indices() {
    // DEAD-322-02: BinRange::iter() is #[cfg(test)]-gated.
    // Verify it works correctly in test builds.
    use super::types::BinRange;
    let range = BinRange::new(10, 3, 32);
    let indices: Vec<usize> = range.iter().collect();
    assert_eq!(indices, vec![7, 8, 9, 10, 11, 12, 13]);
}

#[test]
fn bin_range_len_matches_iter_count() {
    // DEAD-322-02: BinRange::len() and iter() should agree.
    use super::types::BinRange;
    let range = BinRange::new(15, 3, 32);
    assert_eq!(range.len(), range.iter().count());
}

#[test]
fn bin_range_len_at_boundary() {
    // DEAD-322-02: Near lower boundary, len should shrink.
    use super::types::BinRange;
    let range = BinRange::new(1, 3, 32);
    assert_eq!(range.len(), 5); // bins [0..=4] = 5 bins
    assert_eq!(range.iter().count(), 5);
}

#[test]
fn stack_weights_len_matches_iter() {
    // DEAD-322-02: StackWeights::len() should match iter count.
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    assert_eq!(weights.len(), weights.iter().count());
}

// ─── SampleWindow edge cases (TEST-322-05) ────────────────────────────────

#[test]
fn sample_window_at_exact_bin_center() {
    // TEST-322-05: A value exactly at a bin center (e.g. 5.0) should
    // produce a window with the primary bin at 5.
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![5.0_f32; 1];
    let moving = vec![10.0; 1];

    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("exact center should be in-bounds");

    assert_eq!(window.f_range().lo, 2); // 5 - 3 = 2
    assert_eq!(window.f_range().hi, 8); // 5 + 3 = 8
    assert_eq!(window.m_range().lo, 7); // 10 - 3 = 7
    assert_eq!(window.m_range().hi, 13); // 10 + 3 = 13
}

#[test]
fn sample_window_at_zero() {
    // TEST-322-05: A value at 0.0 (lower boundary) should clamp lo to 0.
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![0.0_f32; 1];
    let moving = vec![8.0; 1];

    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("value at 0 should be in-bounds");

    assert_eq!(window.f_range().lo, 0); // clamped from -3 → 0
    assert_eq!(window.f_range().hi, 3); // 0 + 3
}

#[test]
fn sample_window_at_upper_boundary() {
    // TEST-322-05: A value at num_bins-1 should clamp hi to num_bins-1.
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![15.0_f32; 1]; // num_bins - 1
    let moving = vec![8.0; 1];

    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("value at upper boundary should be in-bounds");

    assert_eq!(window.f_range().lo, 12); // 15 - 3
    assert_eq!(window.f_range().hi, 15); // clamped from 18 → 15
}

#[test]
fn sample_window_oob_mask_excludes_sample() {
    // TEST-322-05: An OOB mask value of 0.0 should cause new() to
    // return None (sample excluded).
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![8.0_f32; 1];
    let moving = vec![8.0; 1];
    let mask = vec![0.0_f32; 1]; // excluded

    assert!(
        SampleWindow::new(
            0,
            &fixed,
            &moving,
            num_bins,
            &fix_cfg,
            &mov_cfg,
            Some(&mask),
        )
        .is_none(),
        "OOB mask 0.0 should exclude sample"
    );
}

#[test]
fn sample_window_oob_mask_includes_sample() {
    // TEST-322-05: An OOB mask value of 1.0 should include the sample.
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![8.0_f32; 1];
    let moving = vec![8.0; 1];
    let mask = vec![1.0_f32; 1];

    assert!(
        SampleWindow::new(
            0,
            &fixed,
            &moving,
            num_bins,
            &fix_cfg,
            &mov_cfg,
            Some(&mask),
        )
        .is_some(),
        "OOB mask 1.0 should include sample"
    );
}

#[test]
fn sample_window_moving_only_at_boundary() {
    // TEST-322-05: new_moving_only at the lower boundary.
    let num_bins = 16;
    let mov_cfg = ParzenConfig::new(1.0);
    let moving = vec![0.5_f32; 1];

    let (m_val, m_range, _weights, _inv_sum_m) =
        SampleWindow::new_moving_only(0, &moving, num_bins, &mov_cfg, None)
            .expect("value near 0 should be in-bounds");

    assert!((m_val - 0.5).abs() < 1e-10);
    assert_eq!(m_range.lo, 0); // clamped from -3 → 0
}

#[test]
fn sample_window_different_sigmas() {
    // TEST-322-05: Using different sigma_sq for fixed and moving should
    // produce different bin ranges.
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0); // half_width = 3
    let mov_cfg = ParzenConfig::new(4.0); // half_width = 6
    let fixed = vec![15.0_f32; 1];
    let moving = vec![15.0; 1];

    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("should be in-bounds");

    assert_eq!(window.f_range().len(), 7); // 2*3+1
    assert_eq!(window.m_range().len(), 13); // 2*6+1
}

// ─── HistogramPool concurrent stress test (TEST-322-06) ───────────────────

#[test]
fn histogram_pool_concurrent_checkout_return() {
    // TEST-322-06: Verify that HistogramPool works correctly under
    // concurrent checkout/return via rayon. Each thread checks out
    // a buffer, writes a unique marker, and returns it. After all
    // threads complete, every buffer should be returned and the
    // pool should be in a consistent state.
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let num_bins_sq = 256;
    let pool = HistogramPool::new_with_capacity(num_bins_sq, 4);
    let thread_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let counter = AtomicUsize::new(0);

    (0..thread_count * 3).into_par_iter().for_each(|_| {
        let mut buf = pool.checkout();
        // Write a unique marker to verify the buffer is correctly zeroed
        let idx = counter.fetch_add(1, Ordering::Relaxed) % num_bins_sq;
        buf[idx] = 42.0;
        // Return buffer — next checkout should get a zeroed buffer
        pool.return_buffer(buf);
    });

    // After all threads return, check out a buffer and verify it's zeroed
    let final_buf = pool.checkout();
    assert!(
        final_buf.iter().all(|&v| v == 0.0),
        "buffer should be zeroed after checkout from pool"
    );
}

#[test]
fn histogram_pool_reuse_reduces_allocations() {
    // TEST-322-06: Verify that returning buffers to the pool allows
    // them to be reused (not re-allocated).
    let pool = HistogramPool::new_with_capacity(100, 1);

    // Check out the pre-allocated buffer
    let buf1 = pool.checkout();
    let ptr1 = buf1.as_ptr();

    // Return and check out again — should be the same allocation
    pool.return_buffer(buf1);
    let buf2 = pool.checkout();
    let ptr2 = buf2.as_ptr();

    assert_eq!(
        ptr1, ptr2,
        "returned buffer should be reused (same allocation)"
    );
}

#[test]
fn histogram_pool_checkout_without_return_still_works() {
    // TEST-322-06: If buffers are checked out but never returned
    // (e.g. due to a panic), the pool should still function by
    // allocating new buffers.
    let pool = HistogramPool::new_with_capacity(64, 2);

    // Check out both pre-allocated buffers without returning them
    let _b1 = pool.checkout();
    let _b2 = pool.checkout();

    // Third checkout should still work (allocates on demand)
    let b3 = pool.checkout();
    assert_eq!(b3.len(), 64);
}

// ─── Direct path boundary accumulation (TEST-322-05) ──────────────────────

#[test]
fn direct_histogram_boundary_values_accumulate() {
    // TEST-322-05: Values exactly at 0 and num_bins-1 should produce
    // non-zero histogram entries at the boundary bins.
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fixed = vec![0.0, 15.0]; // boundary values
    let moving = vec![0.0, 15.0];

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let hist = hist_data.as_slice::<f32>().unwrap();

    // Corner (0,0) should have non-zero weight
    assert!(hist[0] > 0.0, "hist[0][0] should be > 0, got {}", hist[0]);
    // Corner (15,15) should have non-zero weight
    assert!(
        hist[15 * num_bins + 15] > 0.0,
        "hist[15][15] should be > 0, got {}",
        hist[15 * num_bins + 15]
    );
}

#[test]
fn direct_histogram_single_sample() {
    // TEST-322-05: A single-sample input should produce a valid
    // histogram with weight only in the support window.
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fixed = vec![8.0];
    let moving = vec![8.0];

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let hist = hist_data.as_slice::<f32>().unwrap();

    // The total histogram sum should be positive
    let total: f32 = hist.iter().sum();
    assert!(total > 0.0, "histogram sum should be > 0, got {total}");

    // The center bin (8,8) should have the highest weight
    let center = hist[8 * num_bins + 8];
    assert!(center > 0.0, "center bin should be > 0, got {center}");

    // Off-diagonal bins outside the support window should be zero
    // (e.g. bin (0,0) is >6σ away from (8,8) with σ=1)
    assert!(hist[0] < 1e-20, "distant bin should be ≈0, got {}", hist[0]);
}

// ─── Exp-ratchet drift at STACK_WEIGHTS_CAPACITY=32 (TEST-323-05) ────────

#[test]
fn exp_ratchet_drift_at_max_capacity() {
    // TEST-323-05: With STACK_WEIGHTS_CAPACITY=32, the maximum range is
    // 31 bins (half_width=15, σ²=25.0). The exp-ratchet FMA chain
    // accumulates floating-point drift over 31 steps. Verify that
    // the drift stays within 1e-4 for each weight value.
    let sigma_sq = 25.0; // σ=5 bins → half_width=ceil(3×5)=15 → 31 bins
    let cfg = ParzenConfig::new(sigma_sq);
    let val = 30.5_f32; // near center of large histogram
    let num_bins = 64;
    let (range, weights) = cfg.compute_weights(val, num_bins);

    // Verify we get the expected 31 bins
    assert_eq!(
        range.len(),
        31,
        "expected 31 bins for sigma_sq=25, got {}",
        range.len()
    );

    // Cross-validate each weight against naive exp()
    for (j, w_ratchet) in weights.iter() {
        let b = range.lo as usize + j;
        let diff = val - b as f32;
        let w_naive = (diff * diff * cfg.inv_2sigma_sq()).exp();
        let abs_err = (w_ratchet - w_naive).abs();
        let rel_err = if w_naive.abs() > 1e-10 {
            abs_err / w_naive
        } else {
            abs_err
        };
        assert!(
            rel_err < 1e-4,
            "exp-ratchet drift too large at sigma_sq={sigma_sq}, bin={b}: \
             ratchet={w_ratchet}, naive={w_naive}, rel_err={rel_err}"
        );
    }
}

#[test]
fn exp_ratchet_drift_sigma_sq_9() {
    // TEST-323-05: sigma_sq=9.0 → σ=3, half_width=9, range=19 bins.
    // Verify ratchet precision at this moderately large range.
    let sigma_sq = 9.0;
    let cfg = ParzenConfig::new(sigma_sq);
    let val = 20.5_f32;
    let num_bins = 64;
    let (range, weights) = cfg.compute_weights(val, num_bins);

    assert_eq!(range.len(), 19, "expected 19 bins for sigma_sq=9");

    for (j, w_ratchet) in weights.iter() {
        let b = range.lo as usize + j;
        let diff = val - b as f32;
        let w_naive = (diff * diff * cfg.inv_2sigma_sq()).exp();
        let abs_err = (w_ratchet - w_naive).abs();
        let rel_err = if w_naive.abs() > 1e-10 {
            abs_err / w_naive
        } else {
            abs_err
        };
        assert!(
            rel_err < 1e-5,
            "exp-ratchet drift at sigma_sq={sigma_sq}, bin={b}: \
             ratchet={w_ratchet}, naive={w_naive}, rel_err={rel_err}"
        );
    }
}

// ─── BinRange edge cases (TEST-323-06) ─────────────────────────────────

#[test]
fn bin_range_primary_exactly_at_num_bins() {
    // TEST-323-06: When primary == num_bins (i.e. value rounds up to the
    // bin count), the range should clamp to the last bin.
    use super::types::BinRange;
    let range = BinRange::new(16, 3, 16); // primary=16, num_bins=16
                                          // lo = max(16-3, 0) = 13, hi = min(16+3, 15) = 15
                                          // Since 13 <= 15, the range is [13, 15]
    assert_eq!(range.lo, 13);
    assert_eq!(range.hi, 15);
}

#[test]
fn bin_range_double_clamping_at_both_boundaries() {
    // TEST-323-06: When the support window exceeds both boundaries
    // simultaneously (impossible for half_width > 0 and num_bins > 0,
    // but test the boundary where lo would be negative and hi would
    // exceed num_bins-1 simultaneously).
    use super::types::BinRange;
    // primary=7, half_width=10, num_bins=8 → lo=max(-3,0)=0, hi=min(17,7)=7
    let range = BinRange::new(7, 10, 8);
    assert_eq!(range.lo, 0);
    assert_eq!(range.hi, 7);
    assert_eq!(range.len(), 8); // full range [0..=7]
}

#[test]
fn bin_range_single_bin_at_boundary() {
    // TEST-323-06: When primary=0 and half_width=0, the range should be [0,0].
    use super::types::BinRange;
    let range = BinRange::new(0, 0, 16);
    assert_eq!(range.lo, 0);
    assert_eq!(range.hi, 0);
    assert_eq!(range.len(), 1);
}

// ─── Various num_bins integration tests (TEST-323-07) ─────────────────

#[test]
fn direct_histogram_small_num_bins() {
    // TEST-323-07: 4 bins — very small histogram.
    let num_bins = 4;
    let sigma_sq = 1.0;
    let fixed = vec![1.5, 2.0, 0.5];
    let moving = vec![1.0, 2.5, 0.0];
    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let hist = hist_data.as_slice::<f32>().unwrap();
    let total: f32 = hist.iter().sum();
    assert!(total > 0.0, "histogram sum should be > 0, got {total}");
    // All entries must be non-negative
    for (i, &v) in hist.iter().enumerate() {
        assert!(v >= 0.0, "hist[{i}] is negative: {v}");
    }
}

#[test]
fn direct_histogram_medium_num_bins() {
    // TEST-323-07: 64 bins — medium histogram.
    let num_bins = 64;
    let sigma_sq = 1.0;
    let n = 200;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.15) % 63.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| ((i * 3 + 1) as f32 * 0.08) % 63.0).collect();
    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let hist = hist_data.as_slice::<f32>().unwrap();
    let total: f32 = hist.iter().sum();
    assert!(total > 0.0 && total.is_finite(), "total={total}");
    // With 64 bins, there should be many non-zero entries
    let nonzero = hist.iter().filter(|&&v| v > 1e-10).count();
    assert!(nonzero > 0, "should have non-zero entries");
}

#[test]
fn sparse_cache_various_num_bins() {
    // TEST-323-07: Sparse cache path with different num_bins values.
    for &num_bins in &[4, 16, 32, 64] {
        let sigma_sq = 1.0;
        let n = 100;
        let fixed: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.3) % (num_bins as f32 - 1.0))
            .collect();
        let moving: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.5 + 1.0) % (num_bins as f32 - 1.0))
            .collect();

        let direct_data = compute_joint_histogram_direct(
            &fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None,
        );
        let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
        let sparse_data = compute_joint_histogram_from_cache_sparse(
            &sparse_w_fixed,
            &moving,
            num_bins,
            sigma_sq,
            None,
            None,
        );

        let direct_slice = direct_data.as_slice::<f32>().unwrap();
        let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

        // Both direct and sparse paths accumulate raw w_f × w_m products identically.
        // Verify nonzero structure matches.
        for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
            let d_nz = *d > 1e-10;
            let s_nz = *s > 1e-10;
            assert_eq!(
                d_nz, s_nz,
                "num_bins={num_bins}: nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
            );
        }
    }
}

// ─── StackWeightsIter trait verification (PERF-323-02) ─────────────────

#[test]
fn stack_weights_iter_is_clone() {
    // PERF-323-02: StackWeightsIter implements Clone, enabling
    // replay of weight sequences for cross-validation.
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    let iter1 = weights.iter();
    let iter2 = iter1.clone();
    // Both iterators should produce the same sequence
    for (a, b) in iter1.zip(iter2) {
        assert_eq!(a.0, b.0, "bin offset mismatch");
        assert!((a.1 - b.1).abs() < 1e-10, "weight mismatch");
    }
}

#[test]
fn stack_weights_iter_is_exact_size() {
    // PERF-323-02: StackWeightsIter implements ExactSizeIterator.
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    let iter = weights.iter();
    assert_eq!(iter.len(), weights.len as usize);
    // After advancing, len should decrease
    let mut iter = weights.iter();
    iter.next();
    assert_eq!(iter.len(), weights.len as usize - 1);
}

#[test]
fn stack_weights_iter_is_double_ended() {
    // PERF-323-02: StackWeightsIter implements DoubleEndedIterator.
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    let mut iter = weights.iter();
    // Front-to-back and back-to-front should produce the same elements
    // in reverse order
    let front: Vec<(usize, f32)> = weights.iter().collect();
    let mut back: Vec<(usize, f32)> = Vec::with_capacity(weights.len as usize);
    while let Some(item) = iter.next_back() {
        back.push(item);
    }
    // back should be the reverse of front
    assert_eq!(front.len(), back.len());
    for (i, (f, b)) in front.iter().zip(back.iter().rev()).enumerate() {
        assert_eq!(f.0, b.0, "offset mismatch at {i}");
        assert!((f.1 - b.1).abs() < 1e-10, "weight mismatch at {i}");
    }
}

// ─── SampleWindow encapsulation (ARCH-323-01) ───────────────────────────

#[test]
fn sample_window_range_accessors_match_fields() {
    // ARCH-323-01: The f_range() and m_range() accessors must return
    // the same values that were set during construction.
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(2.0);
    let fixed = vec![15.3_f32];
    let moving = vec![12.0_f32];
    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("should be in-bounds");

    // The accessor should match what ParzenConfig::compute_weights produces
    let (expected_f_range, _) = fix_cfg.compute_weights(15.3, num_bins);
    let (expected_m_range, _) = mov_cfg.compute_weights(12.0, num_bins);

    assert_eq!(window.f_range().lo, expected_f_range.lo);
    assert_eq!(window.f_range().hi, expected_f_range.hi);
    assert_eq!(window.m_range().lo, expected_m_range.lo);
    assert_eq!(window.m_range().hi, expected_m_range.hi);
}

#[test]
fn sample_window_size_production() {
    // MEM-323-03: Document the production size of SampleWindow.
    // In production builds (without #[cfg(test)] fields), the struct is:
    // f_range: BinRange (2×u16 = 4 bytes)
    // m_range: BinRange (4 bytes)
    // f_weights: StackWeights (~128 bytes)
    // m_weights: StackWeights (~128 bytes)
    // Total ≈ 264 bytes (with alignment padding)
    //
    // In test builds, add f_val + m_val (2×4 = 8 bytes).
    //
    // Note: The actual size may vary due to alignment padding.
    // This test documents the measured size rather than asserting
    // an exact value, since alignment differs by platform.
    let size = std::mem::size_of::<SampleWindow>();
    // On 64-bit platforms: ~264 bytes (production), ~272 bytes (test)
    // We just verify it's within a reasonable range — the important
    // thing is that it's stack-allocated and Copy.
    assert!(size <= 320, "SampleWindow is {size} bytes — expected ≤320");
    assert!(size >= 256, "SampleWindow is {size} bytes — expected ≥256");
}

#[test]
fn stack_weights_size_documentation() {
    // MEM-323-03 / MEM-325-01: Document the size of StackWeights.
    // weights: [f32; 32] = 128 bytes
    // len: u8 = 1 byte + 3 bytes padding (alignment to 4)
    // Total = 132 bytes (with alignment padding)
    let size = std::mem::size_of::<super::types::StackWeights>();
    assert!(size >= 129, "StackWeights is {size} bytes — expected ≥129");
    assert!(size <= 144, "StackWeights is {size} bytes — expected ≤144");
}

#[test]
fn bin_range_size_documentation() {
    // MEM-323-03: Document the size of BinRange.
    // lo: u16 = 2 bytes, hi: u16 = 2 bytes = 4 bytes
    let size = std::mem::size_of::<super::types::BinRange>();
    assert_eq!(size, 4, "BinRange should be 4 bytes (2×u16), got {size}");
}
