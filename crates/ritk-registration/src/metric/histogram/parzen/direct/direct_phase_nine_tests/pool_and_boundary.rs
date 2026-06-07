//! HistogramPool stress tests, direct path boundary, exp-ratchet drift, BinRange edge cases,
//! various num_bins integration, StackWeightsIter traits, and SampleWindow encapsulation/size tests.

use super::super::sample::SampleWindow;
use super::super::types::{BinRange, ParzenConfig};
use super::super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

// ─── HistogramPool concurrent stress test (TEST-322-06) ───────────────────

#[test]
fn histogram_pool_concurrent_checkout_return() {
    let num_bins_sq = 256;
    let pool = HistogramPool::new_with_capacity(num_bins_sq, 4);
    let thread_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let counter = AtomicUsize::new(0);

    moirai::for_each_index_with::<moirai::Adaptive, _>(thread_count * 3, |_| {
        let mut buf = pool.checkout();
        let idx = counter.fetch_add(1, Ordering::Relaxed) % num_bins_sq;
        buf[idx] = 42.0;
        pool.return_buffer(buf);
    });

    let final_buf = pool.checkout();
    assert!(
        final_buf.iter().all(|&v| v == 0.0),
        "buffer should be zeroed after checkout from pool"
    );
}

#[test]
fn histogram_pool_reuse_reduces_allocations() {
    let pool = HistogramPool::new_with_capacity(100, 1);
    let buf1 = pool.checkout();
    let ptr1 = buf1.as_ptr();
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
    let pool = HistogramPool::new_with_capacity(64, 2);
    let _b1 = pool.checkout();
    let _b2 = pool.checkout();
    let b3 = pool.checkout();
    assert_eq!(b3.len(), 64);
}

// ─── Direct path boundary accumulation (TEST-322-05) ──────────────────────

#[test]
fn direct_histogram_boundary_values_accumulate() {
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fixed = vec![0.0, 15.0];
    let moving = vec![0.0, 15.0];

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let hist = hist_data.as_slice::<f32>().unwrap();

    assert!(hist[0] > 0.0, "hist[0][0] should be > 0, got {}", hist[0]);
    assert!(
        hist[15 * num_bins + 15] > 0.0,
        "hist[15][15] should be > 0, got {}",
        hist[15 * num_bins + 15]
    );
}

#[test]
fn direct_histogram_single_sample() {
    let num_bins = 16;
    let sigma_sq = 1.0;
    let fixed = vec![8.0];
    let moving = vec![8.0];

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let hist = hist_data.as_slice::<f32>().unwrap();

    let total: f32 = hist.iter().sum();
    assert!(total > 0.0, "histogram sum should be > 0, got {total}");

    let center = hist[8 * num_bins + 8];
    assert!(center > 0.0, "center bin should be > 0, got {center}");
    assert!(hist[0] < 1e-20, "distant bin should be ≈0, got {}", hist[0]);
}

// ─── Exp-ratchet drift at STACK_WEIGHTS_CAPACITY=32 (TEST-323-05) ────────

#[test]
fn exp_ratchet_drift_at_max_capacity() {
    let sigma_sq = 25.0;
    let cfg = ParzenConfig::new(sigma_sq);
    let val = 30.5_f32;
    let num_bins = 64;
    let (range, weights) = cfg.compute_weights(val, num_bins);

    assert_eq!(
        range.len(),
        31,
        "expected 31 bins for sigma_sq=25, got {}",
        range.len()
    );

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
        assert!(rel_err < 1e-4, "exp-ratchet drift too large at sigma_sq={sigma_sq}, bin={b}: ratchet={w_ratchet}, naive={w_naive}, rel_err={rel_err}");
    }
}

#[test]
fn exp_ratchet_drift_sigma_sq_9() {
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
        assert!(rel_err < 1e-5, "exp-ratchet drift at sigma_sq={sigma_sq}, bin={b}: ratchet={w_ratchet}, naive={w_naive}, rel_err={rel_err}");
    }
}

// ─── BinRange edge cases (TEST-323-06) ─────────────────────────────────

#[test]
fn bin_range_primary_exactly_at_num_bins() {
    let range = BinRange::new(16, 3, 16);
    assert_eq!(range.lo, 13);
    assert_eq!(range.hi, 15);
}

#[test]
fn bin_range_double_clamping_at_both_boundaries() {
    let range = BinRange::new(7, 10, 8);
    assert_eq!(range.lo, 0);
    assert_eq!(range.hi, 7);
    assert_eq!(range.len(), 8);
}

#[test]
fn bin_range_single_bin_at_boundary() {
    let range = BinRange::new(0, 0, 16);
    assert_eq!(range.lo, 0);
    assert_eq!(range.hi, 0);
    assert_eq!(range.len(), 1);
}

// ─── Various num_bins integration tests (TEST-323-07) ─────────────────

#[test]
fn direct_histogram_small_num_bins() {
    let num_bins = 4;
    let sigma_sq = 1.0;
    let fixed = vec![1.5, 2.0, 0.5];
    let moving = vec![1.0, 2.5, 0.0];
    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let hist = hist_data.as_slice::<f32>().unwrap();
    let total: f32 = hist.iter().sum();
    assert!(total > 0.0, "histogram sum should be > 0, got {total}");
    for (i, &v) in hist.iter().enumerate() {
        assert!(v >= 0.0, "hist[{i}] is negative: {v}");
    }
}

#[test]
fn direct_histogram_medium_num_bins() {
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
    let nonzero = hist.iter().filter(|&&v| v > 1e-10).count();
    assert!(nonzero > 0, "should have non-zero entries");
}

#[test]
fn sparse_cache_various_num_bins() {
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
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    let iter1 = weights.iter();
    let iter2 = iter1.clone();
    for (a, b) in iter1.zip(iter2) {
        assert_eq!(a.0, b.0, "bin offset mismatch");
        assert!((a.1 - b.1).abs() < 1e-10, "weight mismatch");
    }
}

#[test]
fn stack_weights_iter_is_exact_size() {
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    let iter = weights.iter();
    assert_eq!(iter.len(), weights.len as usize);
    let mut iter = weights.iter();
    iter.next();
    assert_eq!(iter.len(), weights.len as usize - 1);
}

#[test]
fn stack_weights_iter_is_double_ended() {
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    let mut iter = weights.iter();
    let front: Vec<(usize, f32)> = weights.iter().collect();
    let mut back: Vec<(usize, f32)> = Vec::with_capacity(weights.len as usize);
    while let Some(item) = iter.next_back() {
        back.push(item);
    }
    assert_eq!(front.len(), back.len());
    for (i, (f, b)) in front.iter().zip(back.iter().rev()).enumerate() {
        assert_eq!(f.0, b.0, "offset mismatch at {i}");
        assert!((f.1 - b.1).abs() < 1e-10, "weight mismatch at {i}");
    }
}

// ─── SampleWindow encapsulation and size (ARCH-323-01, MEM-323-03) ────

#[test]
fn sample_window_range_accessors_match_fields() {
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(2.0);
    let fixed = vec![15.3_f32];
    let moving = vec![12.0_f32];
    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("should be in-bounds");

    let (expected_f_range, _) = fix_cfg.compute_weights(15.3, num_bins);
    let (expected_m_range, _) = mov_cfg.compute_weights(12.0, num_bins);

    assert_eq!(window.f_range().lo, expected_f_range.lo);
    assert_eq!(window.f_range().hi, expected_f_range.hi);
    assert_eq!(window.m_range().lo, expected_m_range.lo);
    assert_eq!(window.m_range().hi, expected_m_range.hi);
}

#[test]
fn sample_window_size_production() {
    let size = std::mem::size_of::<SampleWindow>();
    assert!(size <= 320, "SampleWindow is {size} bytes — expected ≤320");
    assert!(size >= 256, "SampleWindow is {size} bytes — expected ≥256");
}

#[test]
fn stack_weights_size_documentation() {
    let size = std::mem::size_of::<super::super::types::StackWeights>();
    assert!(size >= 129, "StackWeights is {size} bytes — expected ≥129");
    assert!(size <= 144, "StackWeights is {size} bytes — expected ≤144");
}

#[test]
fn bin_range_size_documentation() {
    let size = std::mem::size_of::<BinRange>();
    assert_eq!(size, 4, "BinRange should be 4 bytes (2×u16), got {size}");
}
