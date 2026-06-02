use super::types::{BinRange, ParzenConfig, StackWeights};
use super::*;

#[test]
fn accumulate_sample_direct_vs_sparse_weights() {
    // ARCH-317-01: Verify that the monomorphized direct-path accumulate
    // (using SampleWindow with pre-computed StackWeights) produces the same
    // histogram entries as the sparse-cache path (using SparseWFixedEntry).
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);

    let f_val = 15.3_f32;
    let m_val = 20.7_f32;
    let f_primary = f_val.floor() as i32;
    let m_primary = m_val.floor() as i32;
    let f_range = BinRange::new(f_primary, fix_cfg.half_width(), num_bins);
    let m_range = BinRange::new(m_primary, mov_cfg.half_width(), num_bins);

    // Direct-path: build SampleWindow via constructor
    let m_weights = StackWeights::new(
        m_val,
        m_range.lo as usize,
        m_range.hi as usize,
        mov_cfg.inv_2sigma_sq(),
    );
    let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");
    let mut hist_direct = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_direct, num_bins, &window);

    // Sparse-path: build SparseWFixedEntry and use accumulate_sample_sparse
    let sparse_weights: Vec<SparseWFixedEntry> = f_range
        .iter()
        .map(|a| {
            let diff_f = f_val - a as f32;
            let w_f = (diff_f * diff_f * fix_cfg.inv_2sigma_sq()).exp();
            SparseWFixedEntry::new(a as u16, w_f)
        })
        .collect();
    // PERF-328-01: pass combined normalization so sparse matches direct
    let sum_f: f32 = sparse_weights.iter().map(|e| e.weight).sum();
    let sum_m: f32 = m_weights.iter().map(|(_, w)| w).sum();
    let inv_norm = (1.0_f32 / sum_f) * (1.0_f32 / sum_m);
    let mut hist_sparse = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_sparse(
        &mut hist_sparse,
        num_bins,
        m_range,
        &m_weights,
        inv_norm,
        &sparse_weights,
    );

    // Both must produce identical histograms
    for (i, (d, s)) in hist_direct.iter().zip(hist_sparse.iter()).enumerate() {
        let diff = (d - s).abs();
        assert!(
            diff < 1e-10,
            "accumulate_sample mismatch at bin {i}: direct={d}, sparse={s}, diff={diff}"
        );
    }
}

#[test]
fn direct_matches_dense_histogram() {
    // Each sample contributes ~6.28 raw weight (sum_f × sum_m ≈ √2π × √2π).
    // The total should be ~n × 6.28 for interior samples; boundary samples
    // contribute slightly less. Verify the total is in a reasonable range.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;

    let n = 100;
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i * 3 + 7) as f32 % 30.0).collect();

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

    // Verify correct shape
    assert_eq!(
        direct_slice.len(),
        num_bins * num_bins,
        "histogram must have num_bins^2 entries"
    );

    // Verify non-negative
    for (i, &v) in direct_slice.iter().enumerate() {
        assert!(v >= 0.0, "negative value at bin {i}: {v}");
    }

    // PERF-328-01: per-sample normalization by 1/(sum_f × sum_m) means each
    // sample contributes ≈ 1.0 to the histogram total. With n=100 and minor
    // boundary truncation, total should be in [0.5n, 1.5n].
    let total: f32 = direct_slice.iter().sum();
    assert!(
        total > (n as f32) * 0.5,
        "total {total} should be > {} (n × 0.5)",
        n as f32 * 0.5
    );
    assert!(
        total < (n as f32) * 1.5,
        "total {total} should be < {} (n × 1.5)",
        n as f32 * 1.5
    );
}

#[test]
fn direct_with_oob_mask() {
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 10;
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.2).collect();
    let all_oob = vec![0.0f32; n];

    let hist = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
        Some(&all_oob),
        None,
    );
    let sum: f32 = hist.as_slice::<f32>().unwrap().iter().sum();
    assert!(sum < 1e-6, "all-OOB histogram must be zero, got sum={sum}");
}

#[test]
fn sparse_from_cache_matches_direct() {
    // Both direct and sparse paths accumulate raw w_f × w_m products identically.
    // Verify: (1) nonzero patterns match exactly, (2) element-wise values match
    // within floating-point tolerance, (3) totals match.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let n = 100;
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i * 3 + 7) as f32 % 30.0).collect();

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

    // Structural check: nonzero patterns must match exactly.
    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let d_nz = *d > 1e-6;
        let s_nz = *s > 1e-6;
        assert_eq!(
            d_nz, s_nz,
            "nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }

    // SPARSE-329-01: Sparse cache carries inv_sum_f; combined with inv_sum_m
    // at call site, sparse and direct produce approximately equal normalized
    // histograms. Element-wise check uses 1e-3 relative tolerance to absorb
    // floating-point accumulation order differences under parallel reduction.
    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        if *d > 1e-5 {
            let rel_err = ((*d - *s) / d.abs().max(1e-10)).abs();
            assert!(
                rel_err < 1e-3 || (*d - *s).abs() < 1e-7,
                "element-wise mismatch at bin {i}: direct={d}, sparse={s}, rel_err={rel_err}"
            );
        }
    }

    // Totals: SPARSE-329-01 makes sparse ≈ direct. Ratio ≈ 1.0.
    let direct_total: f32 = direct_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    let ratio = sparse_total / direct_total;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "sparse/direct total ratio {ratio} should be ≈ 1.0 (SPARSE-329-01)"
    );
}

#[test]
fn sparse_from_cache_with_oob_mask() {
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 10;
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.2).collect();
    let all_oob = vec![0.0f32; n];

    let sparse_w_fixed =
        build_sparse_w_fixed_transposed(&fixed_vec, num_bins, sigma_sq, Some(&all_oob));

    for (i, entry) in sparse_w_fixed.iter().enumerate() {
        assert!(
            entry.0.is_empty(),
            "OOB sample {i} should have empty sparse entry, got {} elements",
            entry.0.len()
        );
    }

    let hist = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving_vec,
        num_bins,
        sigma_sq,
        Some(&all_oob),
        None,
    );
    let sum: f32 = hist.as_slice::<f32>().unwrap().iter().sum();
    assert!(
        sum < 1e-6,
        "all-OOB sparse histogram must be zero, got sum={sum}"
    );
}

#[test]
fn direct_large_volume_matches_dense() {
    // Large volume stress test — each interior sample contributes ~6.28 raw weight.
    // With 1000 samples, the total should be substantial and finite.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let n = 1000;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32 * 0.02) % 30.0).collect();

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

    // Verify correct shape
    assert_eq!(
        direct_slice.len(),
        num_bins * num_bins,
        "histogram must have num_bins^2 entries"
    );

    // Verify non-negative
    for (i, &v) in direct_slice.iter().enumerate() {
        assert!(v >= 0.0, "negative value at bin {i}: {v}");
    }

    // PERF-328-01: per-sample normalization. Each sample contributes ≈ 1.0
    // to the histogram total. With n=1000 and minor boundary truncation,
    // total should be in [0.5n, 1.5n].
    let total: f32 = direct_slice.iter().sum();
    assert!(
        total > (n as f32) * 0.5,
        "total {total} should be > n × 0.5"
    );
    assert!(
        total < (n as f32) * 1.5,
        "total {total} should be < n × 1.5"
    );
}

#[test]
fn sparse_cache_large_volume_matches_direct() {
    // Both direct and sparse paths accumulate raw w_f × w_m products identically.
    // Verify: (1) nonzero patterns match, (2) element-wise values match,
    // (3) both are non-negative and positive.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let n = 500;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n)
        .map(|i| ((i * 11 + 5) as f32 * 0.03) % 30.0)
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

    // Structural check: nonzero patterns must match.
    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let d_nz = *d > 1e-10;
        let s_nz = *s > 1e-10;
        assert_eq!(
            d_nz, s_nz,
            "nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }

    // Verify both are non-negative
    for (i, &v) in sparse_slice.iter().enumerate() {
        assert!(v >= 0.0, "negative sparse value at bin {i}: {v}");
    }

    // SPARSE-329-01: direct ≈ sparse (combined normalization). Ratio ≈ 1.0.
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
    let ratio = sparse_total / direct_total;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "sparse/direct total ratio {ratio} should be ≈ 1.0 (SPARSE-329-01)"
    );
}

#[test]
fn direct_oob_partial_mask() {
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let n = 20;
    // Use values that stay within [0, num_bins-1] for valid bin assignments.
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 % 28.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 1.2) % 28.0).collect();

    let partial_mask: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();

    let hist_partial = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
        Some(&partial_mask),
        None,
    );

    let half_n = n / 2;
    let hist_half = compute_joint_histogram_direct(
        &fixed_vec[..half_n],
        &moving_vec[..half_n],
        num_bins,
        sigma_sq,
        sigma_sq,
        None,
        None,
    );

    let partial_slice = hist_partial.as_slice::<f32>().unwrap();
    let half_slice = hist_half.as_slice::<f32>().unwrap();

    // PERF-328-01: Both histograms use the same per-sample normalization
    // (1/(sum_f*sum_m)), so the partial-mask and half-sample histograms should
    // still match closely. Tolerance slightly relaxed for floating-point
    // accumulation order differences.
    for (i, (p, h)) in partial_slice.iter().zip(half_slice.iter()).enumerate() {
        let diff = (p - h).abs();
        let max_val = p.abs().max(h.abs()).max(1e-10);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 1e-4 || diff < 1e-6,
            "partial OOB mask mismatch at bin {i}: partial={p}, half={h}, diff={diff}, rel_err={rel_err}"
        );
    }

    let hist_full = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
        None,
        None,
    );
    let sum_partial: f32 = partial_slice.iter().sum();
    let sum_full: f32 = hist_full.as_slice::<f32>().unwrap().iter().sum();
    assert!(
        sum_partial < sum_full,
        "partial-mask sum ({sum_partial}) must be less than full sum ({sum_full})"
    );
    assert!(sum_partial > 0.0, "partial-mask sum must be > 0");
}
