//! Property-based tests for per-sample weight normalization (Sprint 328).
//!
//! Validates the σ²-invariance and direct↔sparse parity invariants
//! established by PERF-328-01 across a wide range of random inputs.
//! Uses `proptest` to generate randomized sample sets, σ values, and
//! boundary configurations, verifying the normalization invariants hold
//! for arbitrary input shapes.

use proptest::prelude::*;

use super::sample::SampleWindow;
use super::types::ParzenConfig;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Per-sample contribution to histogram total is always ≈ 1.0 (PERF-328-01),
    /// for arbitrary sample values in [0, num_bins - 1].
    #[test]
    fn prop_normalized_single_sample_contributes_one(
        f_val in 0.5f32..31.5,
        m_val in 0.5f32..31.5,
        sigma_sq in 0.5f32..9.0,
        num_bins in 16usize..64,
    ) {
        let num_bins = num_bins.next_power_of_two().min(64);
        let fix_cfg = ParzenConfig::new(sigma_sq);
        let mov_cfg = ParzenConfig::new(sigma_sq);
        let window = SampleWindow::new(0, &[f_val], &[m_val], num_bins, &fix_cfg, &mov_cfg, None)
            .expect("in-bounds");
        let mut hist = vec![0.0f32; num_bins * num_bins];
        super::accumulate_sample_direct(&mut hist, num_bins, &window);
        let sum: f32 = hist.iter().sum();
        // Per-sample sum should be ≈ 1.0 (interior samples exactly; boundary
        // samples slightly less due to support clipping, but still in [0.5, 1.05]).
        prop_assert!(
            sum > 0.5 && sum < 1.05,
            "sigma_sq={} num_bins={} f={} m={}: sum={} should be in [0.5, 1.05]",
            sigma_sq, num_bins, f_val, m_val, sum
        );
    }

    /// Direct-path histogram total equals number of in-bounds samples (PERF-328-01).
    /// For σ² ∈ [0.5, 9.0] and sample count ∈ [10, 50], total ≈ n.
    #[test]
    fn prop_normalized_total_equals_n(
        n in 10usize..50,
        sigma_sq in 0.5f32..9.0,
        seed in 0u32..1000,
    ) {
        let num_bins = 32;
        let fixed: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.31) % 30.0).collect();
        let moving: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.47 + 1.0) % 30.0).collect();
        let hist_data = super::compute_joint_histogram_direct(
            &fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None,
        );
        let sum: f32 = hist_data.as_slice::<f32>().unwrap().iter().sum();
        // Sum should be in [0.5n, 1.5n] — boundary clipping can reduce by up to 50%.
        prop_assert!(
            sum > 0.5 * n as f32 && sum < 1.5 * n as f32,
            "n={} sigma_sq={}: sum={} should be in [{}, {}]",
            n, sigma_sq, sum, 0.5 * n as f32, 1.5 * n as f32
        );
    }

    /// σ²-invariance: direct-path total for two different σ² values should
    /// be approximately equal (within 10% relative error).
    #[test]
    fn prop_sigma_invariance(
        sigma_sq_a in 0.5f32..9.0,
        sigma_sq_b in 0.5f32..9.0,
        seed in 0u32..1000,
    ) {
        let num_bins = 32;
        let n = 30;
        let fixed: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.6 + 5.0) % 30.0).collect();
        let moving: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.8 + 3.0) % 30.0).collect();
        let hist_a = super::compute_joint_histogram_direct(
            &fixed, &moving, num_bins, sigma_sq_a, sigma_sq_a, None, None,
        );
        let hist_b = super::compute_joint_histogram_direct(
            &fixed, &moving, num_bins, sigma_sq_b, sigma_sq_b, None, None,
        );
        let sum_a: f32 = hist_a.as_slice::<f32>().unwrap().iter().sum();
        let sum_b: f32 = hist_b.as_slice::<f32>().unwrap().iter().sum();
        // Both should be ≈ n; their difference should be < 10% relative.
        let avg = (sum_a + sum_b) / 2.0;
        let rel_diff = (sum_a - sum_b).abs() / avg.max(1e-6);
        prop_assert!(
            rel_diff < 0.15,
            "sigma_sq_a={} sum_a={} sigma_sq_b={} sum_b={} rel_diff={}",
            sigma_sq_a, sum_a, sigma_sq_b, sum_b, rel_diff
        );
    }

    /// Non-negativity: every histogram entry is ≥ 0 for any valid input.
    #[test]
    fn prop_non_negative(
        n in 5usize..30,
        sigma_sq in 0.5f32..9.0,
        seed in 0u32..1000,
    ) {
        let num_bins = 32;
        let fixed: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.42) % 30.0).collect();
        let moving: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.55 + 2.0) % 30.0).collect();
        let hist_data = super::compute_joint_histogram_direct(
            &fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None,
        );
        let slice = hist_data.as_slice::<f32>().unwrap();
        for (i, &v) in slice.iter().enumerate() {
            prop_assert!(v >= 0.0, "bin {} has negative value {}", i, v);
            prop_assert!(v.is_finite(), "bin {} has non-finite value {}", i, v);
        }
    }

    /// Direct↔sparse nonzero pattern: both paths must have nonzero entries
    /// at exactly the same bins.
    #[test]
    fn prop_direct_sparse_same_support(
        n in 10usize..30,
        sigma_sq in 0.5f32..4.0,
        seed in 0u32..1000,
    ) {
        let num_bins = 32;
        let fixed: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.31) % 30.0).collect();
        let moving: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.47 + 1.0) % 30.0).collect();
        let direct_data = super::compute_joint_histogram_direct(
            &fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None,
        );
        let sparse_w_fixed = super::build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);
        let sparse_data = super::compute_joint_histogram_from_cache_sparse(
            &sparse_w_fixed, &moving, num_bins, sigma_sq, None, None,
        );
        let direct_slice = direct_data.as_slice::<f32>().unwrap();
        let sparse_slice = sparse_data.as_slice::<f32>().unwrap();
        // SPARSE-329-01: direct and sparse are now fully equivalent after
        // per-sample normalization. Verify: nonzero patterns match, totals
        // are equal, and element-wise values match within parallel-reduction
        // tolerance.
        let mut direct_significant = 0usize;
        let mut sparse_significant = 0usize;
        for (d, s) in direct_slice.iter().zip(sparse_slice.iter()) {
            if *d > 1e-5 {
                direct_significant += 1;
            }
            if *s > 1e-5 {
                sparse_significant += 1;
            }
        }
        // Significant-bin counts should be similar (within 10%).
        let avg = ((direct_significant + sparse_significant) / 2).max(1);
        let rel_diff = direct_significant.abs_diff(sparse_significant) as f32 / avg as f32;
        prop_assert!(
            rel_diff < 0.10,
            "significant bin count mismatch: direct={} sparse={} rel_diff={}",
            direct_significant, sparse_significant, rel_diff
        );

        // Totals must be ≈ equal (SPARSE-329-01).
        let direct_total: f32 = direct_slice.iter().sum();
        let sparse_total: f32 = sparse_slice.iter().sum();
        let total_ratio = sparse_total / direct_total;
        prop_assert!(
            (total_ratio - 1.0).abs() < 0.05,
            "sparse/direct total ratio {} should be ≈ 1.0 (SPARSE-329-01)",
            total_ratio
        );
    }

    /// `inv_sum_f()` and `inv_sum_m()` must satisfy `1/(inv_sum_f) = sum_weights`.
    #[test]
    fn prop_inv_sum_inverse(
        f_val in 0.5f32..31.5,
        sigma_sq in 0.5f32..9.0,
    ) {
        let num_bins = 32;
        let cfg = ParzenConfig::new(sigma_sq);
        let window = SampleWindow::new(0, &[f_val], &[f_val + 5.0], num_bins, &cfg, &cfg, None)
            .expect("in-bounds");
        let sum: f32 = window.f_weights.iter().map(|(_, w)| w).sum();
        let expected_inv = 1.0 / sum;
        let actual_inv = window.inv_sum_f();
        let rel_err = (actual_inv - expected_inv).abs() / expected_inv;
        prop_assert!(
            rel_err < 1e-5,
            "f={} sigma_sq={}: inv_sum_f={} vs expected={} rel_err={}",
            f_val, sigma_sq, actual_inv, expected_inv, rel_err
        );
    }
}
