use super::sample::SampleWindow;
use super::types::{BinRange, ParzenConfig, StackWeights, MAX_PARZEN_BINS, STACK_WEIGHTS_CAPACITY};
use super::*;

// ─── BinRange tests (ARCH-316-04) ───────────────────────────────────────────

#[test]
fn bin_range_interior_value() {
    // Interior value: no clamping needed.
    let range = BinRange::new(10, 3, 32);
    assert_eq!(range.lo, 7);
    assert_eq!(range.hi, 13);
    assert_eq!(range.len(), 7);
    assert!(!range.is_empty());
}

#[test]
fn bin_range_near_lower_boundary() {
    // Value near 0: lo should clamp to 0.
    let range = BinRange::new(1, 3, 32);
    assert_eq!(range.lo, 0);
    assert_eq!(range.hi, 4);
    assert_eq!(range.len(), 5);
}

#[test]
fn bin_range_near_upper_boundary() {
    // Value near num_bins-1: hi should clamp to num_bins-1.
    let range = BinRange::new(30, 3, 32);
    assert_eq!(range.lo, 27);
    assert_eq!(range.hi, 31);
    assert_eq!(range.len(), 5);
}

#[test]
fn bin_range_primary_exceeds_num_bins() {
    // When primary > num_bins-1, both lo and hi clamp to the boundary.
    let range = BinRange::new(22, 3, 16);
    assert_eq!(range.lo, 15); // clamped from 19 to 15
    assert_eq!(range.hi, 15);
    assert_eq!(range.len(), 1);
}

#[test]
fn bin_range_primary_negative() {
    // When primary is negative, both lo and hi clamp to 0.
    let range = BinRange::new(-5, 3, 32);
    assert_eq!(range.lo, 0);
    assert_eq!(range.hi, 0);
    assert_eq!(range.len(), 1);
}

#[test]
fn bin_range_iter_produces_correct_indices() {
    let range = BinRange::new(5, 2, 32);
    let indices: Vec<usize> = range.iter().collect();
    assert_eq!(indices, vec![3, 4, 5, 6, 7]);
}

// ─── ParzenConfig tests (ARCH-317-01) ──────────────────────────────────────

#[test]
fn parzen_config_derives_half_width() {
    let cfg = ParzenConfig::new(1.0); // sigma=1, half_width=ceil(3*1)=3
    assert_eq!(cfg.half_width(), 3);
    assert!((cfg.inv_2sigma_sq() - (-0.5)).abs() < 1e-7);
}

#[test]
fn parzen_config_minimum_half_width() {
    let cfg = ParzenConfig::new(0.01); // very narrow sigma
    assert_eq!(cfg.half_width(), 3); // MIN_HALF_WIDTH
}

#[test]
fn parzen_config_broad_sigma() {
    let cfg = ParzenConfig::new(4.0); // sigma=2, half_width=ceil(6)=6
    assert_eq!(cfg.half_width(), 6);
    assert!((cfg.inv_2sigma_sq() - (-0.125)).abs() < 1e-7);
}

// ─── SampleWindow tests (MEM-316-01, FIX-316-07, ARCH-317-01) ─────────────

#[test]
fn sample_window_in_bounds() {
    let fixed = vec![15.3, 20.7];
    let moving = vec![12.0, 18.5];
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);
    let window = SampleWindow::new(0, &fixed, &moving, 32, &fix_cfg, &mov_cfg, None);
    assert!(window.is_some());
    let w = window.unwrap();
    assert_eq!(w.f_val, 15.3);
    assert_eq!(w.m_val, 12.0);
    assert_eq!(w.f_range().lo, 12);
    assert_eq!(w.f_range().hi, 18);
    assert_eq!(w.m_range().lo, 9);
    assert_eq!(w.m_range().hi, 15);
    // Verify pre-computed weights
    assert!(w.f_weights.len() > 0, "fixed weights should be populated");
    assert!(w.m_weights.len() > 0, "moving weights should be populated");
    assert_eq!(w.f_weights.len as usize, w.f_range().len());
    assert_eq!(w.m_weights.len as usize, w.m_range().len());
}

#[test]
fn sample_window_oob_mask_excludes() {
    let fixed = vec![15.3];
    let moving = vec![12.0];
    let oob = vec![0.0f32]; // excluded
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);
    let window = SampleWindow::new(0, &fixed, &moving, 32, &fix_cfg, &mov_cfg, Some(&oob));
    assert!(window.is_none());
}

#[test]
fn sample_window_in_bounds_mask() {
    let fixed = vec![15.3];
    let moving = vec![12.0];
    let oob = vec![1.0f32]; // in-bounds
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);
    let window = SampleWindow::new(0, &fixed, &moving, 32, &fix_cfg, &mov_cfg, Some(&oob));
    assert!(window.is_some());
}

#[test]
fn sample_window_moving_only_in_bounds() {
    let moving = vec![12.0, 18.5];
    let mov_cfg = ParzenConfig::new(1.0);
    let result = SampleWindow::new_moving_only(1, &moving, 32, &mov_cfg, None);
    assert!(result.is_some());
    let (m_val, m_range, m_weights, _inv_sum_m) = result.unwrap();
    assert_eq!(m_val, 18.5);
    assert_eq!(m_range.lo, 15);
    assert_eq!(m_range.hi, 21);
    assert_eq!(m_weights.len(), m_range.len());
}

#[test]
fn sample_window_moving_only_oob() {
    let moving = vec![12.0];
    let oob = vec![0.0f32];
    let mov_cfg = ParzenConfig::new(1.0);
    let result = SampleWindow::new_moving_only(0, &moving, 32, &mov_cfg, Some(&oob));
    assert!(result.is_none());
}

// ─── StackWeights SIMD-alignment tests (PERF-316-03) ───────────────────────
#[test]
fn stack_weights_array_size_is_simd_aligned() {
    // PERF-316-03: The weight array must be 32 elements (128 bytes = 4× AVX2 __m256).
    assert_eq!(STACK_WEIGHTS_CAPACITY, 32);
    assert!(STACK_WEIGHTS_CAPACITY >= MAX_PARZEN_BINS);
}

#[test]
fn stack_weights_padding_slots_are_zero() {
    // PERF-316-03: All slots beyond `len` must be zero-filled.
    let mw = StackWeights::new(15.3, 12, 18, -0.5);
    for i in mw.len()..STACK_WEIGHTS_CAPACITY {
        assert_eq!(
            mw.weights[i], 0.0,
            "slot {i} beyond len must be zero-filled padding"
        );
    }
}

// ─── StackWeights correctness tests ────────────────────────────────────────

#[test]
fn stack_weights_correct() {
    // Verify that StackWeights::new produces the same values as
    // explicit exp() computation for a known input.
    let val = 15.3_f32;
    let sigma_sq = 1.0_f32;
    let inv_2sigma_sq = -0.5 / sigma_sq;
    let lo = 12;
    let hi = 18;
    let mw = StackWeights::new(val, lo, hi, inv_2sigma_sq);

    let expected_len = hi - lo + 1;
    assert_eq!(mw.len(), expected_len, "StackWeights len mismatch");

    for (j, w) in mw.iter() {
        let b = lo + j;
        let diff = val - b as f32;
        let expected_w = (diff * diff * inv_2sigma_sq).exp();
        let diff_actual = (w - expected_w).abs();
        assert!(
            diff_actual < 1e-7,
            "StackWeights iter mismatch at bin {b} (offset {j}): expected {expected_w}, got {w}, diff={diff_actual}"
        );
    }

    // Also test with a value near bin 0 to exercise clamping paths
    let val_edge = 1.2_f32;
    let lo_edge = 0;
    let hi_edge = 4;
    let mw_edge = StackWeights::new(val_edge, lo_edge, hi_edge, inv_2sigma_sq);
    assert_eq!(mw_edge.len(), 5); // 0..=4

    for (j, w) in mw_edge.iter() {
        let b = lo_edge + j;
        let diff = val_edge - b as f32;
        let expected_w = (diff * diff * inv_2sigma_sq).exp();
        let diff_actual = (w - expected_w).abs();
        assert!(
            diff_actual < 1e-7,
            "StackWeights edge iter mismatch at bin {b}: expected {expected_w}, got {w}, diff={diff_actual}"
        );
    }
}

#[test]
fn stack_weights_is_copy() {
    let mw = StackWeights::new(15.3, 12, 18, -0.5);
    let mw_copy = mw;
    for ((j1, w1), (j2, w2)) in mw.iter().zip(mw_copy.iter()) {
        assert_eq!(j1, j2);
        assert!((w1 - w2).abs() < 1e-10);
    }
}

// ─── Monomorphized direct-path accumulate tests (ARCH-317-01) ──────────────

#[test]
fn accumulate_sample_direct_matches_sparse_weights() {
    // ARCH-317-01: Verify that the monomorphized direct-path accumulate
    // (using SampleWindow with pre-computed StackWeights) produces the
    // same histogram entries as the sparse-cache path (using
    // SparseWFixedEntry).
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
    let m_weights = StackWeights::new(
        m_val,
        m_range.lo as usize,
        m_range.hi as usize,
        mov_cfg.inv_2sigma_sq(),
    );

    // Build a SampleWindow for the direct path
    let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
        .expect("in-bounds");

    // Direct-path accumulation
    let mut hist_direct = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_direct, num_bins, &window);

    // Sparse-path accumulation (using SparseWFixedEntry)
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
fn accumulate_sample_direct_total_weight() {
    // ARCH-317-01 / PERF-327-04 / PERF-328-01: accumulate_sample_direct
    // no longer returns a total — the return is `()`. Per PERF-328-01,
    // per-sample normalization by 1/(sum_f × sum_m) means the histogram
    // total for one sample should be ≈ 1.0. Verify correct accumulation
    // by summing the histogram entries directly.
    let num_bins = 32;
    let fix_cfg = ParzenConfig::new(1.0);
    let mov_cfg = ParzenConfig::new(1.0);

    let fixed = vec![15.3f32];
    let moving = vec![20.7f32];
    let window = SampleWindow::new(0, &fixed, &moving, num_bins, &fix_cfg, &mov_cfg, None)
        .expect("sample should be in-bounds");

    let mut hist = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist, num_bins, &window);

    let hist_sum: f32 = hist.iter().sum();
    assert!(
        hist_sum > 0.5,
        "normalized histogram sum must be > 0.5 (≈1.0 per sample), got {hist_sum}"
    );
    assert!(
        hist_sum < 1.5,
        "normalized histogram sum must be < 1.5 (≈1.0 per sample), got {hist_sum}"
    );
    assert!(
        hist_sum.is_finite(),
        "histogram sum must be finite, got {hist_sum}"
    );
}

// ─── compute_half_width SSOT test ──────────────────────────────────────────

#[test]
fn compute_half_width_ssot_values() {
    // SSOT: verify the canonical compute_half_width produces correct values.
    assert_eq!(compute_half_width(0.01), 3); // MIN_HALF_WIDTH
    assert_eq!(compute_half_width(1.0), 3); // ceil(3*1)=3
    assert_eq!(compute_half_width(4.0), 6); // ceil(3*2)=6
    assert_eq!(compute_half_width(9.0), 9); // ceil(3*3)=9
    assert_eq!(compute_half_width(16.0), 12); // ceil(3*4)=12
}

// ─── ParzenConfig::from_intensity_sigma tests (SSOT-318-03) ──────────────

#[test]
fn parzen_config_from_intensity_sigma_basic() {
    // SSOT-318-03: from_intensity_sigma should produce the same sigma_sq
    // as the manual sigma_sq_in_bins formula.
    let sigma = 8.0_f32;
    let min = 0.0_f32;
    let max = 255.0_f32;
    let num_bins = 32;
    let cfg = ParzenConfig::from_intensity_sigma(sigma, min, max, num_bins);
    // Manual: bin_width = 255/31 ≈ 8.226, sigma_in_bins = 8/8.226 ≈ 0.9725
    // sigma_sq ≈ 0.9458
    let num_bins_f = (num_bins - 1) as f32;
    let bin_width = (max - min) / num_bins_f;
    let sigma_in_bins = sigma / bin_width;
    let expected_sigma_sq = sigma_in_bins * sigma_in_bins;
    assert!(
        (cfg.sigma_sq() - expected_sigma_sq).abs() < 1e-5,
        "sigma_sq mismatch: got {}, expected {}",
        cfg.sigma_sq(),
        expected_sigma_sq
    );
    // half_width should be derived from the computed sigma_sq
    assert_eq!(cfg.half_width(), compute_half_width(expected_sigma_sq));
}

#[cfg(feature = "direct-parzen")]
#[test]
fn parzen_config_from_intensity_sigma_self_consistent() {
    // SSOT-319-02: from_intensity_sigma should produce the same sigma_sq
    // as the manual formula (no more sigma_sq_in_bins to delegate to).
    let sigma = 255.0 / 32.0; // typical Mattes sigma
    let min = 0.0;
    let max = 255.0;
    let num_bins = 32;

    let cfg = ParzenConfig::from_intensity_sigma(sigma, min, max, num_bins);

    // Manual: bin_width = (max - min) / (num_bins - 1), sigma_in_bins = sigma / bin_width
    let num_bins_f = (num_bins - 1) as f32;
    let bin_width = (max - min) / num_bins_f;
    let sigma_in_bins = sigma / bin_width;
    let expected_sigma_sq = sigma_in_bins * sigma_in_bins;

    assert!(
        (cfg.sigma_sq() - expected_sigma_sq).abs() < 1e-5,
        "from_intensity_sigma ({}) must match manual formula ({})",
        cfg.sigma_sq(),
        expected_sigma_sq
    );
}

#[test]
#[should_panic(expected = "max must be > min")]
fn parzen_config_from_intensity_sigma_rejects_inverted_range() {
    let _ = ParzenConfig::from_intensity_sigma(8.0, 255.0, 0.0, 32);
}

#[test]
#[should_panic(expected = "num_bins must be > 0")]
fn parzen_config_from_intensity_sigma_rejects_zero_bins() {
    let _ = ParzenConfig::from_intensity_sigma(8.0, 0.0, 255.0, 0);
}

// ─── ParzenConfig PartialEq test (ARCH-318-08) ────────────────────────────

#[test]
fn parzen_config_partial_eq() {
    let a = ParzenConfig::new(1.0);
    let b = ParzenConfig::new(1.0);
    assert_eq!(a, b, "identical ParzenConfigs must be equal");
    let c = ParzenConfig::new(4.0);
    assert_ne!(a, c, "different ParzenConfigs must not be equal");
}

// ─── ParzenConfig half_width invariants (FIX-318-01 replacement) ─────────

#[test]
fn parzen_config_half_width_invariants() {
    let cfg = ParzenConfig::new(1.0);
    assert_eq!(cfg.half_width(), 3);
    let cfg2 = ParzenConfig::new(4.0);
    assert_eq!(cfg2.half_width(), 6);
}

// ─── Broad sigma StackWeights test (FIX-318-01) ────────────────────────────

#[test]
fn stack_weights_broad_sigma() {
    // FIX-318-01 / FIX-319-09: sigma_sq=4.0 → half_width=6 → range=13 bins.
    // StackWeights must handle this without overflow now that
    // STACK_WEIGHTS_CAPACITY=32.
    let cfg = ParzenConfig::new(4.0);
    assert_eq!(cfg.half_width(), 6);
    let val = 15.3_f32;
    let primary = val.floor() as i32;
    let range = BinRange::new(primary, cfg.half_width(), 32);
    assert_eq!(range.len(), 13);
    let weights = StackWeights::new(
        val,
        range.lo as usize,
        range.hi as usize,
        cfg.inv_2sigma_sq(),
    );
    assert_eq!(weights.len(), 13);
    // Verify all weights are positive and finite
    for (j, w) in weights.iter() {
        assert!(w > 0.0, "weight at offset {j} must be positive, got {w}");
        assert!(
            w.is_finite(),
            "weight at offset {j} must be finite, got {w}"
        );
    }
    // Peak weight at primary bin matches Gaussian for val = 15.3,
    // primary = 15, so offset = 0.3 from bin center, inv_2sigma_sq = -0.125
    let expected_peak = f32::exp(0.3f32 * 0.3f32 * (-0.125f32)); // ~0.9888
    let peak_offset = primary as usize - range.lo as usize;
    assert!(
        (weights.weights[peak_offset] - expected_peak).abs() < 1e-6,
        "peak weight should be {expected_peak}, got {}",
        weights.weights[peak_offset]
    );
}

// ─── ParzenConfig::support_bins tests (TEST-319-11) ───────────────────────
#[test]
fn parzen_config_support_bins() {
    let cfg = ParzenConfig::new(1.0); // half_width=3
    assert_eq!(cfg.support_bins(), 7); // 2*3+1

    let cfg2 = ParzenConfig::new(4.0); // half_width=6
    assert_eq!(cfg2.support_bins(), 13); // 2*6+1

    let cfg3 = ParzenConfig::new(0.01); // MIN_HALF_WIDTH=3
    assert_eq!(cfg3.support_bins(), 7);
}

// ─── Exp-ratchet precision test (PERF-319-04) ────────────────────────────
#[test]
fn stack_weights_exp_ratchet_precision() {
    // PERF-319-04: StackWeights::new now uses exp-ratchet. Verify that
    // the ratchet produces values within 1e-5 of the naive computation
    // for a range of sigma values and bin positions.
    for &sigma_sq in &[0.5, 1.0, 2.0, 4.0] {
        let cfg = ParzenConfig::new(sigma_sq);
        let inv_2sigma_sq = cfg.inv_2sigma_sq();
        let val = 15.3_f32;
        let primary = val.floor() as i32;
        let lo = (primary - cfg.half_width() as i32).max(0) as usize as u16;
        let hi = ((primary + cfg.half_width() as i32).min(31)).max(0) as usize as u16;

        let sw = StackWeights::new(val, lo as usize, hi as usize, inv_2sigma_sq);

        for (j, w_ratchet) in sw.iter() {
            let b = lo as usize + j;
            let diff = val - b as f32;
            let w_naive = (diff * diff * inv_2sigma_sq).exp();
            let abs_err = (w_ratchet - w_naive).abs();
            assert!(
                abs_err < 1e-5,
                "ratchet drift at sigma_sq={sigma_sq}, bin={b}: ratchet={w_ratchet}, naive={w_naive}, err={abs_err}"
            );
        }
    }
}
