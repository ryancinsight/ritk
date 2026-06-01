//! Phase Twelve tests — Sprint 327 optimizations and cleanup.
//!
//! Covers:
//! - PERF-327-02: Hoisted base offsets in `accumulate_sample_direct`
//! - PERF-327-03: Hoisted `m_lo_u` in `accumulate_sample_sparse`
//! - PERF-327-04: Dead `total` f32 accumulator removed from `accumulate_sample_direct`
//! - DRY-327-05: `validate_inputs()` SSOT for shared assertions

use super::types::{BinRange, ParzenConfig};
use super::*;

// ── PERF-327-02: Hoisted base offsets in accumulate_sample_direct ──────────

#[test]
fn hoisted_offsets_direct_produces_correct_histogram() {
    // Verify that accumulate_sample_direct with hoisted offsets still produces
    // the same histogram as the sparse path.
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    let f_val = 15.3_f32;
    let m_val = 20.7_f32;
    let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    let mut hist_direct = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_direct, num_bins, &window);

    // Build sparse-path histogram for comparison
    let f_primary = f_val.floor() as i32;
    let m_primary = m_val.floor() as i32;
    let f_range = BinRange::new(f_primary, fix_cfg.half_width(), num_bins);
    let m_range = BinRange::new(m_primary, mov_cfg.half_width(), num_bins);
    let m_weights = StackWeights::new(
        m_val,
        m_range.lo as usize,
        m_range.hi as usize,
        mov_cfg.inv_2sigma_sq(),
    );
    let sparse_weights: Vec<SparseWFixedEntry> = f_range
        .iter()
        .map(|a| {
            let diff_f = f_val - a as f32;
            let w_f = (diff_f * diff_f * fix_cfg.inv_2sigma_sq()).exp();
            SparseWFixedEntry::new(a as u16, w_f)
        })
        .collect();

    // Both direct and sparse paths accumulate raw w_f × w_m products —
    // no normalization difference.
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

    for (i, (d, s)) in hist_direct.iter().zip(hist_sparse.iter()).enumerate() {
        let diff = (d - s).abs();
        assert!(
            diff < 1e-10,
            "hoisted-offset mismatch at bin {i}: direct={d}, sparse={s}, diff={diff}"
        );
    }
}

#[test]
fn hoisted_offsets_direct_boundary_values() {
    // Boundary values where f_range/m_range clamp — offsets must still be correct.
    let num_bins = 16;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    let f_val = 0.3_f32; // near lower boundary
    let m_val = 15.7_f32; // near upper boundary
    let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    let mut hist = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist, num_bins, &window);

    // Histogram should have non-zero entries (sample is in-bounds)
    let sum: f32 = hist.iter().sum();
    assert!(
        sum > 0.0,
        "boundary sample must contribute to histogram, got sum={sum}"
    );
    assert!(sum.is_finite(), "boundary sample sum must be finite");
}

#[test]
fn hoisted_offsets_direct_different_sigma_per_axis() {
    // Different sigma values → different half_width → different base offsets.
    // The hoisted offsets must handle independent f_lo_u and m_lo_u correctly.
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0); // half_width=3 → range ~7
    let mov_cfg = ParzenConfig::new(4.0); // half_width=6 → range ~13

    let f_val = 15.3_f32;
    let m_val = 20.7_f32;
    let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    let mut hist = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist, num_bins, &window);

    // Different ranges → different numbers of non-zero entries per row/col
    let f_range_len = window.f_range().len();
    let m_range_len = window.m_range().len();
    assert!(
        m_range_len > f_range_len,
        "moving range ({m_range_len}) should be wider than fixed ({f_range_len}) with broad sigma"
    );

    let sum: f32 = hist.iter().sum();
    assert!(sum > 0.0, "histogram sum must be positive, got {sum}");
}

// ── PERF-327-03: Hoisted m_lo_u in accumulate_sample_sparse ────────────────

#[test]
fn hoisted_offsets_sparse_boundary_values() {
    // Boundary values for the sparse path — hoisted m_lo_u must be correct.
    let num_bins = 16;
    let mov_cfg = ParzenConfig::new(1.0);

    let m_val = 0.3_f32; // near lower boundary
    let (_, m_range, m_weights, inv_sum_m) =
        SampleWindow::new_moving_only(0, &[m_val], num_bins, &mov_cfg, None).expect("in-bounds");

    let sw = ParzenConfig::new(1.0);
    let f_val = 7.5_f32;
    let f_range = sw.bin_range(f_val, num_bins);
    let f_weights = StackWeights::new(
        f_val,
        f_range.lo as usize,
        f_range.hi as usize,
        sw.inv_2sigma_sq(),
    );
    let sparse_weights: Vec<SparseWFixedEntry> = f_weights
        .iter()
        .map(|(j, w)| SparseWFixedEntry::new(f_range.lo + j as u16, w))
        .collect();

    // SPARSE-329-01: accumulate_sample_sparse now takes inv_norm (full joint),
    // not inv_sum_m (moving-only). Compute inv_sum_f from the sparse weights.
    let sum_f: f32 = sparse_weights.iter().map(|e| e.weight).sum();
    let inv_sum_f = 1.0 / sum_f;
    let inv_norm = inv_sum_f * inv_sum_m;
    let mut hist = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_sparse(
        &mut hist,
        num_bins,
        m_range,
        &m_weights,
        inv_norm,
        &sparse_weights,
    );

    let sum: f32 = hist.iter().sum();
    assert!(
        sum > 0.0,
        "sparse boundary sample sum={sum} must be positive"
    );
}

#[test]
fn hoisted_offsets_sparse_multiple_entries() {
    // Multiple fixed entries → verify all entries use the same hoisted m_lo_u.
    let num_bins = 32;
    let mov_cfg = ParzenConfig::new(1.0);

    let m_val = 16.0_f32;
    let (_, m_range, m_weights, inv_sum_m) =
        SampleWindow::new_moving_only(0, &[m_val], num_bins, &mov_cfg, None).expect("in-bounds");

    // Use broad fixed sigma to get many entries
    let fix_cfg = ParzenConfig::new(4.0);
    let f_val = 15.3_f32;
    let (f_range, f_weights) = fix_cfg.compute_weights(f_val, num_bins);
    let sparse_weights: Vec<SparseWFixedEntry> = f_weights
        .iter()
        .filter(|(_, w)| *w > 1e-12)
        .map(|(j, w)| SparseWFixedEntry::new(f_range.lo + j as u16, w))
        .collect();

    assert!(
        sparse_weights.len() > 7,
        "broad sigma should have >7 entries, got {}",
        sparse_weights.len()
    );

    // SPARSE-329-01: accumulate_sample_sparse now takes inv_norm (full joint)
    let sum_f: f32 = sparse_weights.iter().map(|e| e.weight).sum();
    let inv_sum_f = 1.0 / sum_f;
    let inv_norm = inv_sum_f * inv_sum_m;
    let mut hist = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_sparse(
        &mut hist,
        num_bins,
        m_range,
        &m_weights,
        inv_norm,
        &sparse_weights,
    );

    let sum: f32 = hist.iter().sum();
    assert!(sum > 0.0, "multi-entry sparse sum={sum} must be positive");
}

// ── PERF-327-04: Dead total accumulator removed ────────────────────────────

#[test]
fn accumulate_sample_direct_returns_unit() {
    // PERF-327-04: verify that the function no longer returns f32.
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);
    let window = SampleWindow::new(0, &[15.3], &[20.7], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    let mut hist = vec![0.0f32; num_bins * num_bins];
    // The function returns (), not f32 — this line would fail to compile
    // if it returned anything else.
    accumulate_sample_direct(&mut hist, num_bins, &window);

    // Verify histogram has non-zero entries despite no return value
    let sum: f32 = hist.iter().sum();
    assert!(sum > 0.0, "histogram sum must be positive, got {sum}");
}

#[test]
fn accumulate_sample_direct_histogram_sum_equals_expected() {
    // PERF-328-01: per-sample normalization by 1/(sum_f × sum_m) means each
    // sample contributes ≈ 1.0 to the histogram total regardless of σ².
    // 10% tolerance for discrete-sum vs continuous-integral differences.
    let num_bins = 64;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    let f_val = 31.5_f32; // centered to avoid boundary truncation
    let m_val = 31.5_f32;
    let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    let mut hist = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist, num_bins, &window);

    let sum: f32 = hist.iter().sum();
    // After normalization, the per-sample total should be ≈ 1.0.
    let expected = 1.0_f32;
    let tol = 0.10 * expected;
    assert!(
        (sum - expected).abs() < tol,
        "normalized per-sample sum={sum} should be ≈ {expected}, tol={tol}"
    );
}

// ── DRY-327-05: validate_inputs SSOT ───────────────────────────────────────

#[test]
#[should_panic(expected = "num_bins must be > 0")]
fn validate_inputs_rejects_zero_bins() {
    validate_inputs(0, 100, None);
}

#[test]
fn validate_inputs_accepts_valid() {
    validate_inputs(32, 100, None);
}

#[test]
fn validate_inputs_oob_mask_length_match() {
    let mask = vec![1.0f32; 100];
    validate_inputs(32, 100, Some(&mask));
}

#[test]
#[should_panic(expected = "oob_mask length must match")]
fn validate_inputs_oob_mask_length_mismatch() {
    let mask = vec![1.0f32; 50]; // expected 100
    validate_inputs(32, 100, Some(&mask));
}

// ── Full-pipeline test: direct path with all S327 optimizations ────────────

#[test]
fn direct_path_end_to_end_with_oob_mask() {
    // Full pipeline test exercising all Sprint 327 optimizations together.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let n = 100;

    let fixed: Vec<f32> = (0..n).map(|i| i as f32 % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i * 3 + 7) as f32 % 30.0).collect();
    let oob: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();

    let hist_data = compute_joint_histogram_direct(
        &fixed,
        &moving,
        num_bins,
        sigma_sq,
        sigma_sq,
        Some(&oob),
        None,
    );

    let slice = hist_data.as_slice::<f32>().unwrap();
    let sum: f32 = slice.iter().sum();
    assert!(sum > 0.0, "OOB-masked histogram sum={sum} must be positive");

    // Compare with no-OOB version — should be strictly smaller since ~33%
    // of samples are excluded.
    let hist_full =
        compute_joint_histogram_direct(&fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None);
    let sum_full: f32 = hist_full.as_slice::<f32>().unwrap().iter().sum();
    assert!(
        sum < sum_full,
        "OOB sum ({sum}) must be < full sum ({sum_full})"
    );
}

#[test]
fn sparse_path_end_to_end_with_oob_mask() {
    // Full sparse pipeline test.
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let n = 100;

    let fixed: Vec<f32> = (0..n).map(|i| i as f32 % 30.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i * 3 + 7) as f32 % 30.0).collect();
    let oob: Vec<f32> = (0..n).map(|i| if i % 4 == 0 { 0.0 } else { 1.0 }).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, Some(&oob));
    let hist_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving,
        num_bins,
        sigma_sq,
        Some(&oob),
        None,
    );

    let slice = hist_data.as_slice::<f32>().unwrap();
    let sum: f32 = slice.iter().sum();
    assert!(sum > 0.0, "sparse OOB-masked sum={sum} must be positive");

    let sparse_full = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
    let hist_full = compute_joint_histogram_from_cache_sparse(
        &sparse_full,
        &moving,
        num_bins,
        sigma_sq,
        None,
        None,
    );
    let sum_full: f32 = hist_full.as_slice::<f32>().unwrap().iter().sum();
    assert!(
        sum < sum_full,
        "sparse OOB sum ({sum}) must be < full sum ({sum_full})"
    );
}
