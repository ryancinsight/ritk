//! Size regression, FMA correctness, and end-to-end tests.

use super::super::sample::{SampleWindow, SparseWFixedEntry};
use super::super::types::{BinRange, ParzenConfig, StackWeights};
use super::super::*;

// ── PERF-329-02: FMA-idiomatic inner loop correctness ───────────────────

#[test]
fn fma_inner_loop_matches_reference() {
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    let window = SampleWindow::new(0, &[15.3], &[20.7], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    let mut hist_fma = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_fma, num_bins, &window);

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
    let sizes = compaction_sizes();
    assert_eq!(sizes.bin_range, 4, "BinRange must be 4 bytes (u16×2), got {}", sizes.bin_range);
    assert_eq!(sizes.sparse_fixed_entry, 8, "SparseWFixedEntry must be 8 bytes (u16+pad+f32), got {}", sizes.sparse_fixed_entry);
    assert!(sizes.stack_weights >= 128 && sizes.stack_weights <= 136, "StackWeights must be ~128-136 bytes, got {}", sizes.stack_weights);
    assert!(sizes.parzen_config >= 16 && sizes.parzen_config <= 32, "ParzenConfig must be 16-32 bytes (usize half_width), got {}", sizes.parzen_config);
    assert!(sizes.sample_window >= 256 && sizes.sample_window <= 352, "SampleWindow must be 256-352 bytes, got {}", sizes.sample_window);
}

#[test]
fn stack_weights_size_exact() {
    let size = std::mem::size_of::<StackWeights>();
    assert!(size == 128 || size == 132 || size == 136, "StackWeights is {size} bytes — expected 128, 132, or 136 (depends on alignment)");
}

#[test]
fn bin_range_size_exact() {
    let size = std::mem::size_of::<BinRange>();
    assert_eq!(size, 4, "BinRange must be 4 bytes, got {size}");
}

#[test]
fn sparse_fixed_entry_size_exact() {
    let size = std::mem::size_of::<SparseWFixedEntry>();
    assert_eq!(size, 8, "SparseWFixedEntry must be 8 bytes, got {size}");
}

#[test]
fn parzen_config_size() {
    let size = std::mem::size_of::<ParzenConfig>();
    assert!(size == 12 || size == 16 || size == 24, "ParzenConfig is {size} bytes — expected 12, 16, or 24 (platform-dependent)");
}

// ── CLEANUP-329-03: Verify dead_code annotations are still valid ────────

#[test]
fn bin_range_len_and_is_empty_are_production_api() {
    let range = BinRange::new(15, 3, 32);
    let _len = range.len();
    let _empty = range.is_empty();
}

#[test]
fn stack_weights_len_and_is_empty_are_production_api() {
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    let _len = weights.len();
    let _empty = weights.is_empty();
}

// ── SPARSE-329-01: SparseWFixedT type structure verification ─────────────

#[test]
fn sparse_w_fixed_t_is_tuple_type() {
    let num_bins = 32;
    let sigma_sq = 1.0;
    let fixed = vec![15.3f32];
    let sparse = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);

    assert_eq!(sparse.len(), 1, "single sample → one entry");
    let (entries, inv_sum_f) = &sparse[0];
    assert!(!entries.is_empty(), "in-bounds sample must have entries");
    assert!(*inv_sum_f > 0.0, "inv_sum_f must be positive for in-bounds sample");
}

// ── Accumulate_sample_sparse with inv_norm parameter ─────────────────────

#[test]
fn accumulate_sample_sparse_inv_norm_matches_direct() {
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    let f_val = 15.3_f32;
    let m_val = 20.7_f32;

    let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");
    let mut hist_direct = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_direct, num_bins, &window);

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
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.4 + 2.0) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 1.0) % 30.0).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);

    let hist_no_pool = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed, &moving, num_bins, sigma_sq, None, None,
    );

    let pool = HistogramPool::new(num_bins * num_bins);
    let hist_with_pool = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed, &moving, num_bins, sigma_sq, None, Some(&pool),
    );

    let no_pool_slice = hist_no_pool.as_slice::<f32>().unwrap();
    let with_pool_slice = hist_with_pool.as_slice::<f32>().unwrap();

    for (i, (a, b)) in no_pool_slice.iter().zip(with_pool_slice.iter()).enumerate() {
        assert!((a - b).abs() < 1e-10, "pool mismatch at bin {i}: no_pool={a}, with_pool={b}");
    }
}

// ── σ²-invariance for sparse path ────────────────────────────────────────

#[test]
fn sparse_sigma_invariance() {
    let num_bins = 32;
    let n = 50;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6 + 5.0) % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 3.0) % 30.0).collect();

    let mut sums = Vec::new();
    for &sigma_sq in &[0.5, 1.0, 4.0] {
        let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
        let hist_data = compute_joint_histogram_from_cache_sparse(
            &sparse_w_fixed, &moving, num_bins, sigma_sq, None, None,
        );
        let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();
        sums.push(sum);
    }

    let avg = sums.iter().sum::<f32>() / sums.len() as f32;
    for (i, &sum) in sums.iter().enumerate() {
        let rel = (sum - avg).abs() / avg;
        assert!(rel < 0.15, "sigma variant {i}: sum={sum}, avg={avg}, rel={rel}");
    }
}
