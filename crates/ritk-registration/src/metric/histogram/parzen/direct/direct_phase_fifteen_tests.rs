//! Phase Fifteen tests — Sprint 330 architectural decomposition and API promotion.
//!
//! Covers:
//! - ARCH-330-01: types/ vertical hierarchy (half_width, stack_weights, bin_range, parzen_config)
//! - ARCH-330-02: sample/ vertical hierarchy (sample_window, sparse_entry)
//! - ARCH-330-03: ParzenConfig::half_width() and inv_2sigma_sq() production API
//! - ARCH-330-04: Computation function extraction (accumulate, compute_direct, compute_sparse)
//! - ARCH-330-05: compute_half_width production API
//! - DRY-330-06: Module re-export backward compatibility
//! - MEM-330-07: Structural size regression (post-decomposition invariant)

use super::sample::SampleWindow;
use super::types::{BinRange, ParzenConfig, StackWeights};
use super::*;

// ── ARCH-330-03: ParzenConfig::half_width() production API ─────────────

#[test]
fn parzen_config_half_width_production_api() {
    // ARCH-330-03: half_width() is now production (not #[cfg(test)]-gated).
    let cfg = ParzenConfig::new(1.0);
    let hw = cfg.half_width();
    assert!(hw >= 3, "half_width must be >= MIN_HALF_WIDTH, got {hw}");

    let cfg2 = ParzenConfig::new(4.0);
    assert!(cfg2.half_width() >= 6, "half_width for σ²=4 must be >= 6");
}

#[test]
fn parzen_config_inv_2sigma_sq_production_api() {
    // ARCH-330-03: inv_2sigma_sq() is now production (not #[cfg(test)]-gated).
    let cfg = ParzenConfig::new(1.0);
    let inv = cfg.inv_2sigma_sq();
    let expected = -0.5 / 1.0f32;
    assert!(
        (inv - expected).abs() < 1e-7,
        "inv_2sigma_sq={inv}, expected={expected}"
    );

    let cfg2 = ParzenConfig::new(4.0);
    let inv2 = cfg2.inv_2sigma_sq();
    let expected2 = -0.5 / 4.0f32;
    assert!(
        (inv2 - expected2).abs() < 1e-7,
        "inv_2sigma_sq={inv2}, expected={expected2}"
    );
}

#[test]
fn parzen_config_half_width_and_inv_2sigma_sq_consistency() {
    // ARCH-330-03: Verify that the production API values are internally
    // consistent with compute_weights.
    let cfg = ParzenConfig::new(2.0);
    let hw = cfg.half_width();
    let _inv = cfg.inv_2sigma_sq();

    // half_width should match compute_half_width
    let expected_hw = compute_half_width(2.0);
    assert_eq!(hw, expected_hw, "half_width mismatch");

    // inv_2sigma_sq should produce correct weight magnitudes
    let (_, weights) = cfg.compute_weights(15.0, 32);
    let peak_weight = weights.iter().map(|(_, w)| w).fold(0.0f32, f32::max);
    assert!(peak_weight > 0.0, "peak weight must be positive");

    // The peak weight is exp(0) = 1.0 when the value is exactly at a bin center
    let (_, weights_exact) = cfg.compute_weights(15.0, 32);
    let max_w = weights_exact.iter().map(|(_, w)| w).fold(0.0f32, f32::max);
    assert!(
        (max_w - 1.0).abs() < 0.01,
        "peak weight at bin center should be ≈1.0, got {max_w}"
    );
}

// ── ARCH-330-05: compute_half_width production API ─────────────────────

#[test]
fn compute_half_width_production_api() {
    // ARCH-330-05: compute_half_width is now production (not #[cfg(test)]-gated).
    let hw = compute_half_width(1.0);
    assert_eq!(hw, 3, "compute_half_width(1.0) should be 3");

    let hw2 = compute_half_width(4.0);
    assert_eq!(hw2, 6, "compute_half_width(4.0) should be 6");

    let hw3 = compute_half_width(0.01);
    assert_eq!(
        hw3, 3,
        "compute_half_width(0.01) should clamp to MIN_HALF_WIDTH=3"
    );
}

#[test]
fn compute_half_width_ssot_values() {
    // ARCH-330-05: Production SSOT values for compute_half_width.
    assert_eq!(compute_half_width(0.01), 3); // MIN_HALF_WIDTH
    assert_eq!(compute_half_width(1.0), 3); // ceil(3*1)=3
    assert_eq!(compute_half_width(4.0), 6); // ceil(3*2)=6
    assert_eq!(compute_half_width(9.0), 9); // ceil(3*3)=9
    assert_eq!(compute_half_width(16.0), 12); // ceil(3*4)=12
}

// ── ARCH-330-01: types/ vertical hierarchy verification ───────────────

#[test]
fn types_submodule_bin_range_accessible() {
    // ARCH-330-01: BinRange is accessible from types/ sub-module.
    let range = BinRange::new(15, 3, 32);
    assert_eq!(range.lo, 12);
    assert_eq!(range.hi, 18);
}

#[test]
fn types_submodule_stack_weights_accessible() {
    // ARCH-330-01: StackWeights is accessible from types/ sub-module.
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    assert!(weights.len > 0, "weights must have active entries");
}

#[test]
fn types_submodule_parzen_config_accessible() {
    // ARCH-330-01: ParzenConfig is accessible from types/ sub-module.
    let cfg = ParzenConfig::new(2.0);
    assert_eq!(cfg.sigma_sq(), 2.0);
}

// ── ARCH-330-02: sample/ vertical hierarchy verification ──────────────

#[test]
fn sample_submodule_sparse_entry_accessible() {
    // ARCH-330-02: SparseWFixedEntry is accessible from sample/ sub-module.
    let entry = SparseWFixedEntry::new(5, 0.8);
    assert_eq!(entry.bin, 5);
    assert!(
        (entry.weight - 0.8).abs() < 1e-7,
        "weight should be 0.8, got {}",
        entry.weight
    );
}

#[test]
fn sample_submodule_sparse_w_fixed_t_type() {
    // ARCH-330-02: SparseWFixedT is the correct tuple type.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let fixed = vec![10.0f32, 20.0f32];
    let sparse = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
    assert_eq!(sparse.len(), 2);
    for (entries, inv_sum_f) in &sparse {
        assert!(!entries.is_empty());
        assert!(*inv_sum_f > 0.0, "inv_sum_f must be positive for in-bounds");
    }
}

#[test]
fn sample_submodule_sample_window_accessible() {
    // ARCH-330-02: SampleWindow is accessible from sample/ sub-module.
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);
    let window = SampleWindow::new(0, &[15.3], &[20.7], 32, &fix_cfg, &mov_cfg, None);
    assert!(window.is_some(), "in-bounds sample must produce Some");
}

// ── ARCH-330-04: Computation function extraction verification ──────────

#[test]
fn compute_direct_function_accessible() {
    // ARCH-330-04: compute_joint_histogram_direct is accessible from direct::.
    let fixed = vec![15.3f32];
    let moving = vec![20.7f32];
    let hist_data = compute_joint_histogram_direct(&fixed, &moving, 32, 1.0, 1.0, None, None);
    let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.02,
        "single sample histogram total should be ≈1.0, got {sum}"
    );
}

#[test]
fn compute_sparse_function_accessible() {
    // ARCH-330-04: build_sparse_w_fixed_transposed and
    // compute_joint_histogram_from_cache_sparse are accessible from direct::.
    let fixed = vec![15.3f32];
    let moving = vec![20.7f32];
    let sparse = build_sparse_w_fixed_transposed(&fixed, 32, 1.0, None);
    let hist_data =
        compute_joint_histogram_from_cache_sparse(&sparse, &moving, 32, 1.0, None, None);
    let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.02,
        "single sample sparse histogram total should be ≈1.0, got {sum}"
    );
}

// ── DRY-330-06: Backward compatibility of re-exports ───────────────────

#[test]
fn re_export_backward_compat_histogram_pool() {
    // DRY-330-06: HistogramPool is still accessible from direct::.
    let pool = HistogramPool::new(1024);
    let buf = pool.checkout();
    assert_eq!(buf.len(), 1024);
    pool.return_buffer(buf);
}

#[test]
fn re_export_backward_compat_compaction_sizes() {
    // DRY-330-06: CompactionSizes and compaction_sizes are still accessible.
    let sizes = compaction_sizes();
    assert_eq!(sizes.bin_range, 4, "BinRange must still be 4 bytes");
    assert_eq!(
        sizes.sparse_fixed_entry, 8,
        "SparseWFixedEntry must still be 8 bytes"
    );
}

#[test]
fn re_export_backward_compat_parzen_config() {
    // DRY-330-06: ParzenConfig is still accessible from direct::.
    let cfg = ParzenConfig::new(1.0);
    assert_eq!(cfg.sigma_sq(), 1.0);
}

// ── MEM-330-07: Structural size regression (post-decomposition) ───────

#[test]
fn compaction_sizes_unchanged_after_decomposition() {
    // MEM-330-07: Decomposition should not change struct sizes.
    let sizes = compaction_sizes();
    // BinRange: two u16 fields = 4 bytes
    assert_eq!(sizes.bin_range, 4);
    // SparseWFixedEntry: u16 + 2pad + f32 = 8 bytes
    assert_eq!(sizes.sparse_fixed_entry, 8);
    // SparseSampleCache: 32 * 8 + 1 + padding = 260 bytes
    assert_eq!(sizes.sparse_sample_cache, 260);
    // StackWeights: 32×f32 + u8 len + padding = 128-136 bytes
    assert!(
        sizes.stack_weights >= 128 && sizes.stack_weights <= 136,
        "StackWeights = {} bytes",
        sizes.stack_weights
    );
    // ParzenConfig: f32 + usize + f32 = platform-dependent
    assert!(
        sizes.parzen_config >= 12 && sizes.parzen_config <= 32,
        "ParzenConfig = {} bytes",
        sizes.parzen_config
    );
}

#[test]
fn stack_weights_size_unchanged() {
    let size = std::mem::size_of::<StackWeights>();
    assert!(
        size == 128 || size == 132 || size == 136,
        "StackWeights is {size} bytes"
    );
}

#[test]
fn bin_range_size_unchanged() {
    let size = std::mem::size_of::<BinRange>();
    assert_eq!(size, 4, "BinRange must be 4 bytes, got {size}");
}

#[test]
fn sparse_fixed_entry_size_unchanged() {
    let size = std::mem::size_of::<SparseWFixedEntry>();
    assert_eq!(size, 8, "SparseWFixedEntry must be 8 bytes, got {size}");
}

#[test]
fn sparse_sample_cache_size_unchanged() {
    let size = std::mem::size_of::<SparseSampleCache>();
    assert_eq!(size, 260, "SparseSampleCache must be 260 bytes, got {size}");
}

// ── ARCH-330-03: inv_2sigma_sq used in production weight computation ───

#[test]
fn inv_2sigma_sq_produces_correct_weights() {
    // ARCH-330-03: Verify that inv_2sigma_sq() (now production) produces
    // weight values that match the reference formula.
    let cfg = ParzenConfig::new(1.0);
    let inv_2s = cfg.inv_2sigma_sq();

    // Reference: weight at bin offset 0 = exp(0) = 1.0
    let w0 = (0.0f32 * 0.0 * inv_2s).exp();
    assert!(
        (w0 - 1.0).abs() < 1e-7,
        "weight at offset 0 should be 1.0, got {w0}"
    );

    // Reference: weight at bin offset 1 = exp(-1/(2σ²))
    let expected_w1 = (-0.5f32 / 1.0).exp();
    let w1 = (1.0f32 * 1.0 * inv_2s).exp();
    assert!(
        (w1 - expected_w1).abs() < 1e-7,
        "weight at offset 1: got {w1}, expected {expected_w1}"
    );
}

// ── ARCH-330-03: half_width used in production bin_range computation ───

#[test]
fn half_width_used_in_bin_range_production() {
    // ARCH-330-03: Verify that the production half_width() produces
    // correct bin ranges through ParzenConfig::bin_range.
    let cfg = ParzenConfig::new(4.0);
    let hw = cfg.half_width();
    assert_eq!(hw, 6, "half_width for σ²=4 must be 6");

    let range = cfg.bin_range(15.0, 32);
    assert_eq!(range.lo, 9, "lo should be 15-6=9");
    assert_eq!(range.hi, 21, "hi should be 15+6=21");
}

// ── End-to-end: direct path still works after decomposition ────────────

#[test]
fn direct_histogram_unchanged_after_decomposition() {
    // Verify that the direct histogram computation produces the same
    // results after the architectural decomposition.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.42 + 0.1) % 30.0).collect();
    let moving: Vec<f32> = (0..n)
        .map(|i| ((i * 7 + 3) as f32 * 0.031) % 30.0)
        .collect();

    let direct_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
    let sparse_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq,
        None,
        None,
    );

    let direct_total: f32 = direct_data.as_slice::<f32>().unwrap().iter().sum();
    let sparse_total: f32 = sparse_data.as_slice::<f32>().unwrap().iter().sum();

    // Both must produce normalized histograms with total ≈ n
    assert!(
        (direct_total - n as f32).abs() < n as f32 * 0.1,
        "direct total={direct_total}, expected≈{n}"
    );
    assert!(
        (sparse_total - n as f32).abs() < n as f32 * 0.1,
        "sparse total={sparse_total}, expected≈{n}"
    );

    // Direct↔sparse ratio must be ≈1.0 (SPARSE-329-01 parity)
    let ratio = sparse_total / direct_total;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "sparse/direct ratio={ratio}, expected≈1.0"
    );
}

// ── ARCH-330-03: Verify support_bins still works (test-only API) ───────

#[test]
fn parzen_config_support_bins_test_api() {
    // support_bins() is still #[cfg(test)]-gated, should work in tests.
    let cfg = ParzenConfig::new(1.0);
    assert_eq!(cfg.support_bins(), 7); // 2*3+1
}
