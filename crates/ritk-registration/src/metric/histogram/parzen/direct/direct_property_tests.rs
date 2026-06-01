//! Property-based tests for the direct Parzen histogram computation path.

use super::sample::SampleWindow;
use super::types::ParzenConfig;
use super::*;

// ─── Property tests (TEST-317-06) ──────────────────────────────────────────

#[test]
fn direct_histogram_weights_monotonically_decrease_from_peak() {
    // TEST-317-06: For a single sample, the Parzen weights must be
    // monotonically decreasing as you move away from the primary bin.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);
    let fixed = vec![15.3f32]; // primary bin = 15
    let moving = vec![12.0f32]; // primary bin = 12
    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    // Fixed weights: for each weight, the weight at a bin closer to the
    // primary must be >= weight at a bin farther from the primary.
    let f_primary: usize = 15;
    for (j, w) in window.f_weights.iter() {
        let bin = window.f_range().lo as usize + j;
        let dist = bin.abs_diff(f_primary);
        if dist > 0 {
            // Find the weight at distance dist-1
            let closer_w = if f_primary >= window.f_range().lo as usize
                && f_primary <= window.f_range().hi as usize
            {
                // Primary is in range; weight at dist-1 is the peak (dist=0) or a closer bin
                let closer_bin = if bin < f_primary { bin + 1 } else { bin - 1 };
                if closer_bin >= window.f_range().lo as usize
                    && closer_bin <= window.f_range().hi as usize
                {
                    let closer_j = closer_bin - window.f_range().lo as usize;
                    window.f_weights.weights[closer_j]
                } else {
                    continue; // closer bin out of range
                }
            } else {
                continue;
            };
            assert!(
                w <= closer_w + 1e-7,
                "fixed weight at bin {bin} (dist={dist}) = {w} > closer={closer_w}, not monotonically decreasing"
            );
        }
    }

    // Moving weights: same check
    let m_primary: usize = 12;
    for (j, w) in window.m_weights.iter() {
        let bin = window.m_range().lo as usize + j;
        let dist = bin.abs_diff(m_primary);
        if dist > 0 {
            let closer_bin = if bin < m_primary { bin + 1 } else { bin - 1 };
            if closer_bin >= window.m_range().lo as usize
                && closer_bin <= window.m_range().hi as usize
            {
                let closer_j = closer_bin - window.m_range().lo as usize;
                let closer_w = window.m_weights.weights[closer_j];
                assert!(
                    w <= closer_w + 1e-7,
                    "moving weight at bin {bin} (dist={dist}) = {w} > closer={closer_w}, not monotonically decreasing"
                );
            }
        }
    }
}

#[test]
fn direct_histogram_symmetry_identical_images() {
    // TEST-317-06: When fixed == moving, the joint histogram must be
    // symmetric: H[a, b] == H[b, a].
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 20;
    let values: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3) % 15.0).collect();

    let hist_data =
        compute_joint_histogram_direct(&values, &values, num_bins, sigma_sq, sigma_sq, None, None);
    let slice = hist_data.as_slice::<f32>().unwrap();

    for a in 0..num_bins {
        for b in 0..num_bins {
            let ab = slice[a * num_bins + b];
            let ba = slice[b * num_bins + a];
            let diff = (ab - ba).abs();
            assert!(
                diff < 1e-5,
                "symmetry violation at ({a},{b}): H[{a},{b}]={ab}, H[{b},{a}]={ba}, diff={diff}"
            );
        }
    }
}

#[test]
fn direct_single_sample_concentrates_weight() {
    // TEST-317-06: A single sample should concentrate its weight at
    // (primary_f, primary_m) and the weight should decrease with
    // distance from that peak.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let f_val = 15.0_f32; // exactly on bin center
    let m_val = 20.0_f32;
    let fixed = vec![f_val];
    let moving = vec![m_val];

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let slice = hist_data.as_slice::<f32>().unwrap();

    let f_primary = f_val.floor() as usize; // 15
    let m_primary = m_val.floor() as usize; // 20
    let peak = slice[f_primary * num_bins + m_primary];
    assert!(peak > 0.0, "peak weight must be positive, got {peak}");

    // Any cell at distance > 0 from the peak must have weight ≤ peak
    for a in 0..num_bins {
        for b in 0..num_bins {
            let dist_a = (a as i32 - f_primary as i32).unsigned_abs();
            let dist_b = (b as i32 - m_primary as i32).unsigned_abs();
            if dist_a + dist_b > 0 {
                let w = slice[a * num_bins + b];
                assert!(w <= peak + 1e-7, "cell ({a},{b}) weight {w} > peak {peak}");
            }
        }
    }
}

#[test]
fn direct_histogram_normalization_total_weight() {
    // PERF-328-01: per-sample normalization by 1/(sum_f × sum_m) means each
    // sample contributes ≈ 1.0 to the histogram total. For n=50 samples
    // (with minor boundary truncation), total should be in [0.5n, 1.5n].
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let n = 50;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.6) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.8 + 1.0) % 30.0).collect();

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
    let total: f32 = slice.iter().sum();

    let expected_min = n as f32 * 0.5;
    let expected_max = n as f32 * 1.5;
    assert!(
        total > expected_min,
        "normalized total weight {total} should be > n × 0.5 = {expected_min}"
    );
    assert!(
        total < expected_max,
        "normalized total weight {total} should be < n × 1.5 = {expected_max}"
    );

    // The total must be positive and finite
    assert!(total > 0.0, "total weight must be positive, got {total}");
    assert!(
        total.is_finite(),
        "total weight must be finite, got {total}"
    );
}

#[test]
fn direct_boundary_bins_populated() {
    // TEST-317-06: Even samples near the boundary (primary near 0 or
    // num_bins-1) should contribute non-zero weight to boundary bins.
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let fixed = vec![0.5f32]; // near bin 0
    let moving = vec![14.8f32]; // near bin 14 (near num_bins-1=15)

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let slice = hist_data.as_slice::<f32>().unwrap();

    // Bin 0 should have non-zero fixed-axis weight
    let bin0_sum: f32 = slice[0..num_bins].iter().sum();
    assert!(bin0_sum > 0.0, "bin 0 row sum must be > 0, got {bin0_sum}");

    // Bin 15 column should have non-zero moving-axis weight
    let mut bin15_col = 0.0f32;
    for a in 0..num_bins {
        bin15_col += slice[a * num_bins + 15];
    }
    // Note: primary=14 with half_width=3 → range [11,15], so bin 15 should have weight
    assert!(
        bin15_col > 0.0,
        "bin 15 column sum must be > 0, got {bin15_col}"
    );
}

#[test]
fn direct_sparse_cache_path_matches_after_parity() {
    // SPARSE-329-01: Both direct and sparse paths now apply full joint
    // normalization (inv_norm = inv_sum_f × inv_sum_m). The histograms
    // should be numerically identical (within floating-point tolerance
    // from parallel accumulation order differences).
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let n = 200;
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

    // Structural check: nonzero pattern must match exactly.
    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let d_nz = *d > 1e-6;
        let s_nz = *s > 1e-6;
        assert_eq!(
            d_nz, s_nz,
            "nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }

    // PERF-328-01: direct normalizes by 1/(sum_f × sum_m), sparse by 1/sum_m.
    // For σ²=1, sum_f ≈ √(2π) ≈ 2.51. Ratio check with 15% tolerance.
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
    let rel_err = (sparse_total - direct_total).abs() / direct_total;
    assert!(
        rel_err < 0.10,
        "sparse/direct total ratio error {rel_err} should be < 10% (both normalize by 1/(sum_f×sum_m))"
    );
}

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

#[test]
#[should_panic(expected = "fixed_norm must not be empty")]
fn direct_rejects_empty_input() {
    // TEST-318-06: Empty input must panic with a clear message.
    let empty: Vec<f32> = vec![];
    let _ = compute_joint_histogram_direct(&empty, &empty, 32, 1.0, 1.0, None, None);
}

#[test]
#[should_panic(expected = "sigma_sq must be positive")]
fn direct_rejects_zero_sigma() {
    // TEST-318-06: Zero sigma must panic.
    let _ = ParzenConfig::new(0.0);
}

#[test]
#[should_panic(expected = "sigma_sq must be positive, got NaN")]
fn direct_rejects_nan_sigma() {
    // TEST-318-06: NaN sigma must panic.
    let _ = ParzenConfig::new(f32::NAN);
}

#[test]
fn direct_single_bin_histogram() {
    // TEST-318-06: Single-bin histogram (degenerate case) — all weight
    // concentrates in the one cell.
    let num_bins = 1;
    let sigma_sq = 1.0_f32;
    let fixed = vec![0.0f32];
    let moving = vec![0.0f32];

    let hist_data =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let slice = hist_data.as_slice::<f32>().unwrap();

    assert_eq!(slice.len(), 1, "single-bin histogram must have 1 entry");
    assert!(
        slice[0] > 0.0,
        "single-bin entry must be positive, got {}",
        slice[0]
    );
}

#[test]
fn direct_marginal_consistency_with_oob_mask() {
    // TEST-318-06: With a partial OOB mask, the marginal sums must
    // still be consistent (both axes sum to the same total).
    // PERF-328-01: Use values within [0, num_bins-1] to avoid OOB NaN.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let n = 20;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3) % 28.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.9) % 28.0).collect();
    let partial_mask: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();

    let hist_data = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
        Some(&partial_mask),
        None,
    );
    let slice = hist_data.as_slice::<f32>().unwrap();

    // Compute marginals
    let mut fixed_marginal = vec![0.0f32; num_bins];
    let mut moving_marginal = vec![0.0f32; num_bins];
    for a in 0..num_bins {
        for b in 0..num_bins {
            let v = slice[a * num_bins + b];
            fixed_marginal[a] += v;
            moving_marginal[b] += v;
        }
    }
    let sum_fixed: f32 = fixed_marginal.iter().sum();
    let sum_moving: f32 = moving_marginal.iter().sum();
    let diff = (sum_fixed - sum_moving).abs();
    assert!(
        diff < 1e-4,
        "marginal sums must match with OOB mask: fixed={sum_fixed}, moving={sum_moving}, diff={diff}"
    );
}
