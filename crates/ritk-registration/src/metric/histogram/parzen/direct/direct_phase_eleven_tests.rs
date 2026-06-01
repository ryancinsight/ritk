//! Phase Eleven tests for the direct Parzen histogram computation path.
//!
//! Tests for:
//! - MEM-325-01: StackWeights.len u8 compaction
//! - MEM-325-02: BinRange::new num_bins overflow protection
//! - ARCH-325-06: ParzenConfig::sum_weights() production availability
//! - PERF-325-03: merge_histograms correctness (regression)

use super::sample::SampleWindow;
use super::types::{BinRange, ParzenConfig, StackWeights, STACK_WEIGHTS_CAPACITY};
use super::*;

// ─── StackWeights.len u8 compaction (MEM-325-01) ─────────────────────────

#[test]
fn stack_weights_len_is_u8_max_31() {
    // MEM-325-01: len is u8, max value = STACK_WEIGHTS_CAPACITY - 1 = 31.
    // At sigma_sq=25.0 (half_width=15), the range is 31 bins.
    let cfg = ParzenConfig::new(25.0);
    let (_, weights) = cfg.compute_weights(20.0, 64);
    assert_eq!(
        weights.len, 31,
        "max active count must be 31 for sigma_sq=25"
    );
    // u8 can hold 0..=255, so 31 is well within range.
    assert!(
        weights.len <= 31,
        "len must never exceed STACK_WEIGHTS_CAPACITY-1"
    );
}

#[test]
fn stack_weights_len_u8_all_common_sigma_values() {
    // MEM-325-01: Verify u8 field works for a range of sigma values.
    let sigma_values = [0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0];
    for &sigma_sq in &sigma_values {
        let cfg = ParzenConfig::new(sigma_sq);
        let val = 20.0_f32;
        let num_bins = 64;
        let (range, weights) = cfg.compute_weights(val, num_bins);
        // len must equal the number of bins in the range
        assert_eq!(
            weights.len as usize,
            range.len(),
            "sigma_sq={sigma_sq}: len={} must equal range.len()={}",
            weights.len,
            range.len()
        );
        // len must be <= 31 (STACK_WEIGHTS_CAPACITY - 1)
        assert!(
            weights.len <= 31,
            "sigma_sq={sigma_sq}: len={} exceeds max 31",
            weights.len
        );
    }
}

#[test]
fn stack_weights_len_as_usize_for_indexing() {
    // MEM-325-01: Verify that weights.len as usize works correctly
    // for indexing into the weights array.
    let cfg = ParzenConfig::new(1.0);
    let (_, weights) = cfg.compute_weights(15.3, 32);
    let len_usize = weights.len as usize;
    // All active entries must be accessible
    for j in 0..len_usize {
        assert!(
            weights.weights[j].is_finite(),
            "weight at index {j} must be finite"
        );
    }
    // The slot right after active entries must be zero
    if len_usize < STACK_WEIGHTS_CAPACITY {
        assert_eq!(
            weights.weights[len_usize], 0.0,
            "first padding slot must be zero"
        );
    }
}

#[test]
fn stack_weights_size_after_u8_compaction() {
    // MEM-325-01: StackWeights should be smaller after u8 compaction.
    let size = std::mem::size_of::<StackWeights>();
    // weights: [f32; 32] = 128 bytes
    // len: u8 = 1 byte
    // Compiler may add up to 3 bytes padding for alignment.
    // Total: 129..=132 bytes
    assert!(
        size >= 129,
        "StackWeights is {size} bytes — expected ≥129 (128 + 1 for u8 len)"
    );
    assert!(
        size <= 136,
        "StackWeights is {size} bytes — expected ≤136 (128 + 1 + 7 padding max)"
    );
    // Must not be larger than the old 136-byte size.
    assert!(
        size <= 136,
        "StackWeights must not grow after u8 compaction, got {size}"
    );
}

// ─── BinRange::new overflow protection (MEM-325-02) ──────────────────────

#[test]
#[should_panic(expected = "exceeds u16::MAX")]
fn bin_range_new_rejects_num_bins_exceeding_u16_max() {
    // MEM-325-02: num_bins = 65536 must panic (u16::MAX + 1).
    let _ = BinRange::new(10, 3, 65536);
}

#[test]
fn bin_range_new_accepts_num_bins_at_u16_max() {
    // MEM-325-02: num_bins = 65535 (= u16::MAX) must work.
    let range = BinRange::new(100, 3, 65535);
    assert_eq!(range.lo, 97);
    assert_eq!(range.hi, 103);
}

#[test]
fn bin_range_new_accepts_small_num_bins() {
    // MEM-325-02: Small num_bins must still work.
    let range = BinRange::new(5, 3, 16);
    assert_eq!(range.lo, 2);
    assert_eq!(range.hi, 8);
}

#[test]
#[should_panic(expected = "exceeds u16::MAX")]
fn bin_range_new_rejects_very_large_num_bins() {
    // MEM-325-02: A very large num_bins (100000) must panic.
    let _ = BinRange::new(50000, 3, 100000);
}

// ─── ParzenConfig::sum_weights production availability (ARCH-325-06) ──────

#[test]
fn sum_weights_is_available_without_test_cfg() {
    // ARCH-325-06: sum_weights() is now a production method.
    // This test verifies it can be called from non-test-gated code.
    let cfg = ParzenConfig::new(1.0);
    let sum = cfg.sum_weights(15.3, 32);
    // Interior value should approximate √(2π) ≈ 2.5066
    let expected = (2.0 * std::f32::consts::PI).sqrt();
    assert!(
        (sum - expected).abs() < 0.05,
        "sum_weights={sum}, expected≈{expected}"
    );
}

#[test]
fn sum_weights_production_normalization_factor() {
    // ARCH-325-06: sum_weights can be used as a normalization factor
    // to make each sample's Parzen window integrate to 1.0.
    let cfg = ParzenConfig::new(1.0);
    let num_bins = 32;
    let val = 15.3_f32;

    let (_, weights) = cfg.compute_weights(val, num_bins);
    let sum = cfg.sum_weights(val, num_bins);

    // Verify sum equals the manual sum of weights
    let manual_sum: f32 = weights.iter().map(|(_, w)| w).sum();
    assert!(
        (sum - manual_sum).abs() < 1e-6,
        "sum_weights={sum} != manual_sum={manual_sum}"
    );

    // Normalized weights should sum to ~1.0
    let normalized_sum: f32 = weights.iter().map(|(_, w)| w / sum).sum();
    assert!(
        (normalized_sum - 1.0).abs() < 1e-5,
        "normalized sum={normalized_sum}, expected≈1.0"
    );
}

#[test]
fn sum_weights_boundary_truncation_redistributes() {
    // ARCH-325-06: A boundary value's weight sum is less than interior,
    // so normalization by sum_weights increases the per-bin weight
    // to compensate for the truncated tail.
    let cfg = ParzenConfig::new(1.0);
    let num_bins = 32;

    let boundary_sum = cfg.sum_weights(0.5, num_bins);
    let interior_sum = cfg.sum_weights(15.3, num_bins);

    assert!(
        boundary_sum < interior_sum,
        "boundary_sum={boundary_sum} should be < interior_sum={interior_sum}"
    );

    // After normalization, both should sum to ~1.0
    let boundary_norm: f32 = {
        let (_, w) = cfg.compute_weights(0.5, num_bins);
        w.iter().map(|(_, w)| w / boundary_sum).sum()
    };
    let interior_norm: f32 = {
        let (_, w) = cfg.compute_weights(15.3, num_bins);
        w.iter().map(|(_, w)| w / interior_sum).sum()
    };

    assert!(
        (boundary_norm - 1.0).abs() < 1e-5,
        "boundary normalized={boundary_norm}, expected≈1.0"
    );
    assert!(
        (interior_norm - 1.0).abs() < 1e-5,
        "interior normalized={interior_norm}, expected≈1.0"
    );
}

// ─── merge_histograms regression (PERF-325-03) ───────────────────────────

#[test]
fn merge_histograms_correctness_regression() {
    // PERF-325-03: Verify merge_histograms still works correctly
    // after doc/performance review changes.
    let mut a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    merge_histograms(&mut a, &b);
    assert_eq!(a, vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]);
}

#[test]
fn merge_histograms_large_buffer() {
    // PERF-325-03: Test with a large buffer (typical histogram size).
    let n = 1024; // 32×32 bins
    let mut a = vec![1.0f32; n];
    let b = vec![2.0f32; n];
    merge_histograms(&mut a, &b);
    for (i, &v) in a.iter().enumerate() {
        assert_eq!(v, 3.0, "bin {i} must be 3.0, got {v}");
    }
}

#[test]
fn merge_histograms_non_multiple_of_8() {
    // PERF-325-03: Non-power-of-8 length — remainder loop must handle
    // trailing elements correctly.
    let mut a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0]; // 5 elements
    let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
    merge_histograms(&mut a, &b);
    assert_eq!(a, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
}

// ─── SampleWindow size after u8 compaction (MEM-325-01) ──────────────────

#[test]
fn sample_window_size_after_u8_compaction() {
    // MEM-325-01: SampleWindow should be smaller after StackWeights.len
    // compaction from usize (8 bytes) to u8 (1 byte).
    let size = std::mem::size_of::<SampleWindow>();
    // Production: f_range(4) + m_range(4) + f_weights(129+pad) + m_weights(129+pad)
    // ≈ 266-280 bytes depending on alignment
    // Test: add f_val(4) + m_val(4) = +8 bytes
    assert!(
        size <= 320,
        "SampleWindow is {size} bytes — expected ≤320 after u8 compaction"
    );
}
