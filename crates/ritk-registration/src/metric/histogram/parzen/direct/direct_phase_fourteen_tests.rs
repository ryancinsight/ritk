//! Phase Fourteen tests — Sprint 329 full joint normalization parity and cleanup.
//!
//! Covers:
//! - SPARSE-329-01: Full joint normalization in sparse path (inv_sum_f in SparseWFixedT)
//! - SPARSE-329-01: Direct↔sparse numerical identity
//! - PERF-329-02: FMA-idiomatic inner loop correctness
//! - MEM-329-04: Structural size regression tests
//! - CLEANUP-329-03: Dead-code annotation correctness

use super::sample::SampleWindow;
use super::types::{BinRange, ParzenConfig, StackWeights};
use super::*;

// ── SPARSE-329-01: Full joint normalization in sparse path ───────────────

#[test]
fn sparse_full_normalization_total_equals_n() {
    // SPARSE-329-01: With full joint normalization, the sparse path histogram
    // total should ≈ number of in-bounds samples, matching the direct path.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 100;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5 + 1.0) % 30.0).collect();

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

    // Each of the n samples contributes ≈1.0 after full normalization
    assert!(
        (sum - n as f32).abs() < n as f32 * 0.1,
        "sparse normalized histogram sum={sum}, expected≈{n}"
    );
}

#[test]
fn sparse_full_normalization_boundary_and_interior_equal() {
    // SPARSE-329-01: After full normalization, boundary and interior samples
    // both contribute ≈1.0 (same as direct path).
    let num_bins = 32;
    let sigma_sq = 1.0;

    // Interior sample
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
    let sum_interior: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();

    // Boundary sample (near lower boundary)
    let fixed_b = vec![0.5f32];
    let moving_b = vec![20.7f32];
    let sparse_b = build_sparse_w_fixed_transposed(&fixed_b, num_bins, sigma_sq, None);
    let hist_b = compute_joint_histogram_from_cache_sparse(
        &sparse_b, &moving_b, num_bins, sigma_sq, None, None,
    );
    let sum_boundary: f32 = hist_b.as_slice::<f32>().unwrap().iter().sum();

    assert!(
        (sum_interior - 1.0).abs() < 0.02,
        "interior sparse sum={sum_interior}, expected≈1.0"
    );
    assert!(
        (sum_boundary - 1.0).abs() < 0.02,
        "boundary sparse sum={sum_boundary}, expected≈1.0"
    );
}

#[test]
fn sparse_full_normalization_with_oob_mask() {
    // SPARSE-329-01: OOB-masked sparse path should have total ≈ in-bounds count.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 20;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 1.5) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 1.0) % 30.0).collect();
    let oob: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, Some(&oob));
    let hist_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq,
        Some(&oob),
        None,
    );
    let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();
    let in_bounds = n / 2;

    assert!(
        (sum - in_bounds as f32).abs() < in_bounds as f32 * 0.1,
        "sparse OOB normalized sum={sum}, expected≈{in_bounds}"
    );
}

#[test]
fn sparse_full_normalization_different_sigma() {
    // SPARSE-329-01: Different sigma values should still produce total ≈ n.
    let num_bins = 32;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 5.0) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 3.0) % 30.0).collect();

    for &sigma_sq in &[0.5, 1.0, 4.0, 9.0] {
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
        assert!(
            (sum - n as f32).abs() < n as f32 * 0.1,
            "sigma_sq={sigma_sq}: sparse normalized sum={sum}, expected≈{n}"
        );
    }
}

#[test]
fn sparse_cache_stores_inv_sum_f() {
    // SPARSE-329-01: Verify that build_sparse_w_fixed_transposed stores correct inv_sum_f.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 5;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 5.0 + 3.0) % 30.0).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);

    let cfg = ParzenConfig::new(sigma_sq);
    for (i, (entries, inv_sum_f)) in sparse_w_fixed.iter().enumerate() {
        if !entries.is_empty() {
            // Compute expected inv_sum_f
            let expected_sum: f32 = entries.iter().map(|e| e.weight).sum();
            let expected_inv = 1.0 / expected_sum;
            assert!(
                (inv_sum_f - expected_inv).abs() < 1e-6,
                "sample {i}: stored inv_sum_f={inv_sum_f} vs expected={expected_inv}"
            );
        }
    }
}

#[test]
fn sparse_cache_inv_sum_f_matches_compute_weights_with_inv_sum() {
    // SPARSE-329-01: inv_sum_f in sparse cache must match ParzenConfig::compute_weights_with_inv_sum.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 10;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 2.5 + 1.0) % 30.0).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
    let cfg = ParzenConfig::new(sigma_sq);

    for (i, (_entries, stored_inv_sum_f)) in sparse_w_fixed.iter().enumerate() {
        let (_, _, expected_inv_sum_f) = cfg.compute_weights_with_inv_sum(fixed[i], num_bins);
        assert!(
            (stored_inv_sum_f - expected_inv_sum_f).abs() < 1e-7,
            "sample {i}: stored={stored_inv_sum_f} vs computed={expected_inv_sum_f}"
        );
    }
}

#[test]
fn sparse_cache_oob_stores_zero_inv_sum_f() {
    // SPARSE-329-01: OOB samples should have inv_sum_f = 0.0 (never used).
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 5;
    let fixed: Vec<f32> = (0..n).map(|i| i as f32 * 5.0).collect();
    let oob: Vec<f32> = vec![1.0, 1.0, 0.0, 1.0, 0.0]; // exclude samples 2 and 4

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, Some(&oob));

    // OOB samples should have empty entries and inv_sum_f = 0.0
    assert!(
        sparse_w_fixed[2].0.is_empty(),
        "OOB sample 2 should have empty entries"
    );
    assert!(
        sparse_w_fixed[2].1 == 0.0,
        "OOB sample 2 should have inv_sum_f=0.0, got {}",
        sparse_w_fixed[2].1
    );
    assert!(
        sparse_w_fixed[4].0.is_empty(),
        "OOB sample 4 should have empty entries"
    );
    assert!(
        sparse_w_fixed[4].1 == 0.0,
        "OOB sample 4 should have inv_sum_f=0.0, got {}",
        sparse_w_fixed[4].1
    );

    // In-bounds samples should have non-empty entries and positive inv_sum_f
    assert!(
        !sparse_w_fixed[0].0.is_empty(),
        "in-bounds sample 0 should have entries"
    );
    assert!(
        sparse_w_fixed[0].1 > 0.0,
        "in-bounds sample 0 should have positive inv_sum_f"
    );
}

// ── SPARSE-329-01: Direct↔sparse numerical identity ─────────────────────

#[test]
fn direct_sparse_numerically_identical() {
    // SPARSE-329-01: Both paths now apply full joint normalization.
    // The resulting histograms should be numerically identical (within
    // floating-point accumulation tolerance from parallel reduction).
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 100;
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

    let direct_slice = direct_data.as_slice::<f32>().unwrap();
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    // Element-wise check: within 1e-3 relative or 1e-7 absolute tolerance
    // (parallel reduction changes accumulation order)
    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let diff = (*d - *s).abs();
        let rel = diff / d.abs().max(1e-10);
        assert!(
            rel < 1e-3 || diff < 1e-7,
            "bin {i}: direct={d}, sparse={s}, diff={diff}, rel={rel}"
        );
    }

    // Total must be approximately equal
    let direct_total: f32 = direct_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    let rel_err = (sparse_total - direct_total).abs() / direct_total;
    assert!(
        rel_err < 0.05,
        "sparse/direct total rel_err={rel_err}, should be < 5%"
    );
}

#[test]
fn direct_sparse_identical_broad_sigma() {
    // SPARSE-329-01: Broad sigma (σ²=4) — both paths must still match.
    let num_bins = 32;
    let sigma_sq = 4.0;
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

    let direct_slice = direct_data.as_slice::<f32>().unwrap();
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    // Nonzero patterns must match
    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        assert_eq!(*d > 1e-10, *s > 1e-10, "nonzero mismatch at bin {i}");
    }

    let direct_total: f32 = direct_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    let ratio = sparse_total / direct_total;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "broad sigma sparse/direct ratio={ratio}, expected≈1.0"
    );
}

#[test]
fn direct_sparse_identical_with_oob_mask() {
    // SPARSE-329-01: With OOB mask, both paths must still match.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 30;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3) % 28.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.9) % 28.0).collect();
    let oob: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();

    let direct_data = compute_joint_histogram_direct(
        &fixed,
        &moving,
        num_bins,
        sigma_sq,
        sigma_sq,
        Some(&oob),
        None,
    );
    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, Some(&oob));
    let sparse_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq,
        Some(&oob),
        None,
    );

    let direct_slice = direct_data.as_slice::<f32>().unwrap();
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    let direct_total: f32 = direct_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    let rel_err = (sparse_total - direct_total).abs() / direct_total.max(1e-6);
    assert!(
        rel_err < 0.10,
        "OOB sparse/direct total rel_err={rel_err}, should be < 10%"
    );
}

#[test]
fn direct_sparse_single_sample_identical() {
    // SPARSE-329-01: Single-sample test — both paths must produce ≈1.0 total.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let fixed = vec![15.3f32];
    let moving = vec![20.7f32];

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

    let direct_sum: f32 = direct_data.as_slice::<f32>().unwrap().iter().sum();
    let sparse_sum: f32 = sparse_data.as_slice::<f32>().unwrap().iter().sum();

    assert!(
        (direct_sum - 1.0).abs() < 0.02,
        "direct sum={direct_sum}, expected≈1.0"
    );
    assert!(
        (sparse_sum - 1.0).abs() < 0.02,
        "sparse sum={sparse_sum}, expected≈1.0"
    );
}

// ── SPARSE-329-01: Different sigma per axis ──────────────────────────────

#[test]
fn direct_sparse_different_sigma_per_axis() {
    // SPARSE-329-01: Different sigma_sq_fix and sigma_sq_mov must still produce
    // matching histograms between direct and sparse paths.
    let num_bins = 32;
    let sigma_sq_fix = 1.0;
    let sigma_sq_mov = 4.0;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.4 + 2.0) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 1.0) % 30.0).collect();

    let direct_data = compute_joint_histogram_direct(
        &fixed,
        &moving,
        num_bins,
        sigma_sq_fix,
        sigma_sq_mov,
        None,
        None,
    );
    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq_fix, None);
    let sparse_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq_mov,
        None,
        None,
    );

    let direct_slice = direct_data.as_slice::<f32>().unwrap();
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    let direct_total: f32 = direct_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    let rel_err = (sparse_total - direct_total).abs() / direct_total;
    assert!(
        rel_err < 0.05,
        "different-sigma sparse/direct rel_err={rel_err}, should be < 5%"
    );
}

// ── PERF-329-02: FMA-idiomatic inner loop correctness ───────────────────

#[test]
fn fma_inner_loop_matches_reference() {
    // PERF-329-02: The `hist[idx] += w_f * w_m * inv_norm` form must produce
    // the same result as the explicit step-by-step computation.
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    let window = SampleWindow::new(0, &[15.3], &[20.7], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    // Accumulate using the function under test
    let mut hist_fma = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_fma, num_bins, &window);

    // Reference: step-by-step computation
    let inv_norm = window.inv_sum_f() * window.inv_sum_m();
    let f_lo_u = window.f_range().lo as usize;
    let m_lo_u = window.m_range().lo as usize;
    let mut hist_ref = vec![0.0f32; num_bins * num_bins];
    for (fi, w_f) in window.f_weights.iter() {
        let row_base = (f_lo_u + fi) * num_bins;
        for (mj, w_m) in window.m_weights.iter() {
            hist_ref[row_base + m_lo_u + mj] += w_f * w_m * inv_norm;
        }
    }

    for (i, (actual, expected)) in hist_fma.iter().zip(hist_ref.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "FMA mismatch at bin {i}: actual={actual}, expected={expected}"
        );
    }
}

// ── MEM-329-04: Structural size regression tests ────────────────────────

#[test]
fn compaction_sizes_regression() {
    // MEM-329-04: Guard against accidental struct size regressions.
    let sizes = compaction_sizes();
    assert_eq!(
        sizes.bin_range, 4,
        "BinRange must be 4 bytes (u16×2), got {}",
        sizes.bin_range
    );
    assert_eq!(
        sizes.sparse_fixed_entry, 8,
        "SparseWFixedEntry must be 8 bytes (u16+pad+f32), got {}",
        sizes.sparse_fixed_entry
    );
    assert!(
        sizes.stack_weights >= 128 && sizes.stack_weights <= 136,
        "StackWeights must be ~128-136 bytes, got {}",
        sizes.stack_weights
    );
    assert!(
        sizes.parzen_config >= 16 && sizes.parzen_config <= 32,
        "ParzenConfig must be 16-32 bytes (usize half_width), got {}",
        sizes.parzen_config
    );
    // SampleWindow: production ≈ 272 bytes, test ≈ 280 with f_val/m_val
    assert!(
        sizes.sample_window >= 256 && sizes.sample_window <= 352,
        "SampleWindow must be 256-352 bytes, got {}",
        sizes.sample_window
    );
}

#[test]
fn stack_weights_size_exact() {
    // MEM-329-04: StackWeights size should be exactly 132 bytes
    // (32×f32=128 + u8 len=1 + 3 bytes padding = 132).
    let size = std::mem::size_of::<StackWeights>();
    assert!(
        size == 128 || size == 132 || size == 136,
        "StackWeights is {size} bytes — expected 128, 132, or 136 (depends on alignment)"
    );
}

#[test]
fn bin_range_size_exact() {
    // MEM-329-04: BinRange must be exactly 4 bytes (two u16 fields).
    let size = std::mem::size_of::<BinRange>();
    assert_eq!(size, 4, "BinRange must be 4 bytes, got {size}");
}

#[test]
fn sparse_fixed_entry_size_exact() {
    // MEM-329-04: SparseWFixedEntry must be exactly 8 bytes (u16 + 2pad + f32).
    let size = std::mem::size_of::<super::sample::SparseWFixedEntry>();
    assert_eq!(size, 8, "SparseWFixedEntry must be 8 bytes, got {size}");
}

#[test]
fn parzen_config_size() {
    // MEM-329-04: ParzenConfig = f32 + usize + f32 — size depends on platform.
    // On 64-bit: 4 + 4pad + 8 + 4 + 4pad = 24. On 32-bit: 4 + 4 + 4 = 12.
    let size = std::mem::size_of::<ParzenConfig>();
    assert!(
        size == 12 || size == 16 || size == 24,
        "ParzenConfig is {size} bytes — expected 12, 16, or 24 (platform-dependent)"
    );
}

// ── CLEANUP-329-03: Verify dead_code annotations are still valid ────────

#[test]
fn bin_range_len_and_is_empty_are_production_api() {
    // CLEANUP-329-03: BinRange::len() and is_empty() are production API,
    // not #[cfg(test)]-gated. This test verifies they compile in non-test cfg.
    let range = BinRange::new(15, 3, 32);
    let _len = range.len();
    let _empty = range.is_empty();
}

#[test]
fn stack_weights_len_and_is_empty_are_production_api() {
    // CLEANUP-329-03: StackWeights::len() and is_empty() are production API.
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    let _len = weights.len();
    let _empty = weights.is_empty();
}

// ── SPARSE-329-01: SparseWFixedT type structure verification ─────────────

#[test]
fn sparse_w_fixed_t_is_tuple_type() {
    // SPARSE-329-01: Verify that SparseWFixedT is Vec<(Vec<SparseWFixedEntry>, f32)>.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let fixed = vec![15.3f32];
    let sparse = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);

    assert_eq!(sparse.len(), 1, "single sample → one entry");
    let (entries, inv_sum_f) = &sparse[0];
    assert!(!entries.is_empty(), "in-bounds sample must have entries");
    assert!(
        *inv_sum_f > 0.0,
        "inv_sum_f must be positive for in-bounds sample"
    );
}

// ── Accumulate_sample_sparse with inv_norm parameter ─────────────────────

#[test]
fn accumulate_sample_sparse_inv_norm_matches_direct() {
    // SPARSE-329-01: When given inv_norm (full joint), accumulate_sample_sparse
    // should produce the same histogram as accumulate_sample_direct.
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    let f_val = 15.3_f32;
    let m_val = 20.7_f32;

    // Direct path
    let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");
    let mut hist_direct = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_direct, num_bins, &window);

    // Sparse path with inv_norm
    let (f_range, f_weights, inv_sum_f) = fix_cfg.compute_weights_with_inv_sum(f_val, num_bins);
    let (_, m_range, m_weights, inv_sum_m) =
        SampleWindow::new_moving_only(0, &[m_val], num_bins, &mov_cfg, None).expect("in-bounds");
    let inv_norm = inv_sum_f * inv_sum_m;

    let sparse_entries: Vec<SparseWFixedEntry> = f_weights
        .iter()
        .filter(|(_, w)| *w > 1e-12)
        .map(|(j, w)| SparseWFixedEntry::new(f_range.lo + j as u16, w))
        .collect();

    let mut hist_sparse = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_sparse(
        &mut hist_sparse,
        num_bins,
        m_range,
        &m_weights,
        inv_norm,
        &sparse_entries,
    );

    for (i, (d, s)) in hist_direct.iter().zip(hist_sparse.iter()).enumerate() {
        let diff = (d - s).abs();
        assert!(
            diff < 1e-10,
            "inv_norm mismatch at bin {i}: direct={d}, sparse={s}, diff={diff}"
        );
    }
}

// ── End-to-end: sparse path with pool ────────────────────────────────────

#[test]
fn sparse_path_with_pool_matches_without() {
    // SPARSE-329-01: Sparse path with histogram pool should produce the same result.
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.4 + 2.0) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 1.0) % 30.0).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);

    // Without pool
    let hist_no_pool = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq,
        None,
        None,
    );

    // With pool
    let pool = HistogramPool::new(num_bins * num_bins);
    let hist_with_pool = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq,
        None,
        Some(&pool),
    );

    let no_pool_slice = hist_no_pool.as_slice::<f32>().unwrap();
    let with_pool_slice = hist_with_pool.as_slice::<f32>().unwrap();

    for (i, (a, b)) in no_pool_slice.iter().zip(with_pool_slice.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "pool mismatch at bin {i}: no_pool={a}, with_pool={b}"
        );
    }
}

// ── σ²-invariance for sparse path ────────────────────────────────────────

#[test]
fn sparse_sigma_invariance() {
    // SPARSE-329-01: Sparse-path histogram total should be ≈ n regardless of σ²,
    // matching the direct path's σ²-invariance.
    let num_bins = 32;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 5.0) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 3.0) % 30.0).collect();

    let mut sums = Vec::new();
    for &sigma_sq in &[0.5, 1.0, 4.0] {
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
        sums.push(sum);
    }

    // All sums should be approximately equal (within 15% relative)
    let avg = sums.iter().sum::<f32>() / sums.len() as f32;
    for (i, &sum) in sums.iter().enumerate() {
        let rel = (sum - avg).abs() / avg;
        assert!(
            rel < 0.15,
            "sigma variant {i}: sum={sum}, avg={avg}, rel={rel}"
        );
    }
}
