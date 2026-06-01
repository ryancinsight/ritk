//! Phase Ten tests for the direct Parzen histogram computation path.
//!
//! Tests for:
//! - TEST-324-06: weight-sum approximation to continuous Gaussian
//! - TEST-324-07: StackWeights zero-padding beyond active length
//! - TEST-324-08: SampleWindow OOB mask partial / full coverage
//! - MEM-324-04: BinRange `Ord` derivation and u16 overflow protection
//! - PERF-324-05: merge_histograms correctness
//! - ARCH-324-03: accumulate_sample_sparse concrete `&[SparseWFixedEntry]` signature
//! - TEST-324-06: direct vs sparse parity at large sigma

use super::sample::SampleWindow;
use super::types::{ParzenConfig, STACK_WEIGHTS_CAPACITY};
use super::*;

// ─── Weight-sum ≈ √(2πσ²) (TEST-324-06) ──────────────────────────────────

#[test]
fn weight_sum_matches_continuous_gaussian_sigma_1() {
    // For σ²=1 (σ=1), the continuous Gaussian integral over ℝ equals √(2π) ≈ 2.5066.
    // The discrete sum over integer bins should approximate this.
    let cfg = ParzenConfig::new(1.0);
    let expected = (2.0 * std::f32::consts::PI).sqrt(); // ≈ 2.5066

    // Interior value (far from boundaries)
    let sum_interior = cfg.sum_weights(15.3, 32);
    let diff_interior = (sum_interior - expected).abs();
    assert!(
        diff_interior < 0.05,
        "interior sum {sum_interior} differs from √(2π)≈{expected} by {diff_interior}"
    );

    // Value exactly on a bin center
    let sum_center = cfg.sum_weights(16.0, 32);
    let diff_center = (sum_center - expected).abs();
    assert!(
        diff_center < 0.05,
        "bin-center sum {sum_center} differs from √(2π)≈{expected} by {diff_center}"
    );
}

#[test]
fn weight_sum_matches_continuous_gaussian_sigma_4() {
    // For σ²=4 (σ=2), √(2π×4) = 2√(2π) ≈ 5.0133.
    let cfg = ParzenConfig::new(4.0);
    let expected = 2.0 * (2.0 * std::f32::consts::PI).sqrt(); // ≈ 5.0133

    let sum = cfg.sum_weights(15.0, 64);
    let diff = (sum - expected).abs();
    assert!(
        diff < 0.1,
        "sum {sum} differs from 2√(2π)≈{expected} by {diff}"
    );
}

#[test]
fn weight_sum_boundary_deviation() {
    // A boundary value (val=0.5) loses part of the Gaussian tail to
    // truncation, so its weight sum should be strictly less than an
    // interior value's sum.
    let cfg = ParzenConfig::new(1.0);
    let boundary_sum = cfg.sum_weights(0.5, 32);
    let interior_sum = cfg.sum_weights(15.3, 32);
    assert!(
        boundary_sum < interior_sum,
        "boundary sum {boundary_sum} should be < interior sum {interior_sum}"
    );
}

// ─── StackWeights zero-padding (TEST-324-07) ─────────────────────────────

#[test]
fn stack_weights_zero_padding_beyond_len() {
    // All slots beyond the active `len` must be zero-filled.
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    for j in weights.len as usize..STACK_WEIGHTS_CAPACITY {
        assert_eq!(
            weights.weights[j], 0.0,
            "slot {j} beyond len={} must be zero-filled padding",
            weights.len
        );
    }
}

#[test]
fn stack_weights_zero_padding_invariant_after_construction() {
    // With σ²=0.01 (very narrow), the window is only 7 bins (MIN_HALF_WIDTH=3).
    // All entries beyond those 7 must still be zero.
    let cfg = ParzenConfig::new(0.01);
    assert_eq!(cfg.half_width(), 3, "minimum half-width must be 3");
    assert_eq!(cfg.support_bins(), 7, "minimum support is 7 bins");

    let (_, weights) = cfg.compute_weights(15.3, 32);
    assert_eq!(weights.len, 7, "active weight count must be 7");

    for j in weights.len as usize..STACK_WEIGHTS_CAPACITY {
        assert_eq!(
            weights.weights[j], 0.0,
            "slot {j} beyond len=7 must be zero-filled padding"
        );
    }
}

// ─── SampleWindow OOB mask (TEST-324-08) ─────────────────────────────────

#[test]
fn sample_window_oob_mask_partial_coverage() {
    // 4 samples: mask [1.0, 0.0, 1.0, 0.0] — indices 0,2 in-bounds; 1,3 OOB.
    let fixed = vec![15.3f32, 0.5, 20.0, 31.0];
    let moving = vec![12.0f32, 5.0, 18.0, 25.0];
    let oob_mask = vec![1.0f32, 0.0, 1.0, 0.0];
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    for (i, &mask_val) in oob_mask.iter().enumerate() {
        let window = SampleWindow::new(i, &fixed, &moving, 32, &fix_cfg, &mov_cfg, Some(&oob_mask));
        if mask_val >= 0.5 {
            assert!(
                window.is_some(),
                "sample {i} with mask {mask_val} should be in-bounds"
            );
        } else {
            assert!(
                window.is_none(),
                "sample {i} with mask {mask_val} should be out-of-bounds"
            );
        }
    }
}

#[test]
fn sample_window_all_oob_produces_empty_histogram() {
    // When all samples are OOB, the resulting histogram must be all zeros.
    let num_bins = 32;
    let sigma_sq = 1.0f32;
    let fixed = vec![15.3f32, 20.0];
    let moving = vec![12.0f32, 18.0];
    let oob_mask = vec![0.0f32, 0.0]; // all excluded

    let hist_data = compute_joint_histogram_direct(
        &fixed,
        &moving,
        num_bins,
        sigma_sq,
        sigma_sq,
        Some(&oob_mask),
        None,
    );
    let hist = hist_data.as_slice::<f32>().unwrap();
    for (i, &v) in hist.iter().enumerate() {
        assert_eq!(v, 0.0, "all-OOB histogram bin {i} must be zero, got {v}");
    }
}

// ─── BinRange Ord + u16 (MEM-324-04) ─────────────────────────────────────

#[test]
fn bin_range_ord_ordering() {
    use super::types::BinRange;

    let a = BinRange::new(5, 3, 32); // lo=2, hi=8
    let b = BinRange::new(10, 3, 32); // lo=7, hi=13
    assert!(a < b, "BinRange a={a:?} should be < b={b:?} under Ord");
}

#[test]
fn bin_range_u16_overflow_protection() {
    use super::types::BinRange;

    // Large num_bins (65535) — the u16 fields must handle this correctly.
    let range = BinRange::new(100, 3, 65535);
    assert_eq!(range.lo, 97);
    assert_eq!(range.hi, 103);

    // Near the upper boundary of u16 range.
    let range = BinRange::new(65530, 3, 65535);
    assert_eq!(range.lo, 65527);
    assert_eq!(range.hi, 65533); // 65530+3=65533, clamped by min(num_bins-1=65534) → 65533
}

// ─── merge_histograms (PERF-324-05) ──────────────────────────────────────

#[test]
fn merge_histograms_correctness() {
    let mut a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    merge_histograms(&mut a, &b);
    assert_eq!(a, vec![5.0f32, 7.0, 9.0]);
}

// ─── accumulate_sample_sparse concrete type (ARCH-324-03) ────────────────

#[test]
fn accumulate_sample_sparse_concrete_type() {
    // Verify that accumulate_sample_sparse accepts &[SparseWFixedEntry]
    // (the new concrete signature) and produces the same result as the
    // direct path when given matching normalization factors.
    // This is primarily a compilation + API surface test.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);

    let f_val = 15.3_f32;
    let m_val = 20.7_f32;

    // Build a SampleWindow for the direct path (both axes pre-computed).
    let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    // Direct-path accumulation
    let mut hist_direct = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_direct, num_bins, &window);

    // Build sparse entries as Vec<SparseWFixedEntry>, pass as &[…]
    let (f_range, f_weights) = fix_cfg.compute_weights(f_val, num_bins);
    let (m_range, m_weights) = mov_cfg.compute_weights(m_val, num_bins);

    let entries: Vec<SparseWFixedEntry> = f_weights
        .iter()
        .map(|(j, w_f)| SparseWFixedEntry::new(f_range.lo + j as u16, w_f))
        .collect();

    // PERF-328-01: pass combined normalization so sparse matches direct
    let sum_f: f32 = entries.iter().map(|e| e.weight).sum();
    let sum_m: f32 = m_weights.iter().map(|(_, w)| w).sum();
    let inv_norm = (1.0_f32 / sum_f) * (1.0_f32 / sum_m);
    let mut hist_sparse = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_sparse(
        &mut hist_sparse,
        num_bins,
        m_range,
        &m_weights,
        inv_norm,
        &entries,
    );

    // Both paths must produce identical histograms
    for (i, (d, s)) in hist_direct.iter().zip(hist_sparse.iter()).enumerate() {
        let diff = (d - s).abs();
        assert!(
            diff < 1e-10,
            "mismatch at bin {i}: direct={d}, sparse={s}, diff={diff}"
        );
    }
}

// ─── Direct vs sparse parity at large sigma (TEST-324-06) ────────────────

#[test]
fn direct_histogram_large_sigma_sparse_parity() {
    // With σ²=16 (σ=4, half_width=12), the direct and sparse-cache
    // paths must produce the same histogram (within tolerance).
    let num_bins = 32;
    let sigma_sq = 16.0_f32;
    let n = 10;

    let fixed: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 3.1 + 2.0) % (num_bins as f32 - 1.0))
        .collect();
    let moving: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 2.7 + 5.0) % (num_bins as f32 - 1.0))
        .collect();

    // Direct path
    let direct_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let direct = direct_data.as_slice::<f32>().unwrap();

    // Sparse-cache path
    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
    let sparse_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq,
        None,
        None,
    );
    let sparse = sparse_data.as_slice::<f32>().unwrap();

    // Both direct and sparse paths now produce normalized histograms (SPARSE-329-01).
    for (i, (d, s)) in direct.iter().zip(sparse.iter()).enumerate() {
        let d_nz = *d > 1e-6;
        let s_nz = *s > 1e-6;
        assert_eq!(
            d_nz, s_nz,
            "large-sigma nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }

    // Totals: both must be positive and finite. SPARSE-329-01 makes
    // sparse ≈ direct for any σ²; ratio ≈ 1.0.
    let direct_total: f32 = direct.iter().sum();
    let sparse_total: f32 = sparse.iter().sum();
    assert!(
        direct_total > 0.0,
        "direct histogram should have positive sum"
    );
    assert!(
        sparse_total > 0.0,
        "sparse histogram should have positive sum"
    );
    let ratio = sparse_total / direct_total;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "large-sigma: sparse/direct ratio {ratio} should be ≈ 1.0 (SPARSE-329-01)"
    );
}
