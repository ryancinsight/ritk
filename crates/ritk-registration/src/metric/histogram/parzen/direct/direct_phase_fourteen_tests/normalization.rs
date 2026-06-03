//! SPARSE-329-01: Full joint normalization in sparse path.

use super::super::sample::SampleWindow;
use super::super::types::ParzenConfig;
use super::super::*;

#[test]
fn sparse_full_normalization_total_equals_n() {
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

    assert!(
        (sum - n as f32).abs() < n as f32 * 0.1,
        "sparse normalized histogram sum={sum}, expected≈{n}"
    );
}

#[test]
fn sparse_full_normalization_boundary_and_interior_equal() {
    let num_bins = 32;
    let sigma_sq = 1.0;

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
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 5;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 5.0 + 3.0) % 30.0).collect();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, None);

    for (i, (entries, inv_sum_f)) in sparse_w_fixed.iter().enumerate() {
        if !entries.is_empty() {
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
    let num_bins = 32;
    let sigma_sq = 1.0;
    let n = 5;
    let fixed: Vec<f32> = (0..n).map(|i| i as f32 * 5.0).collect();
    let oob: Vec<f32> = vec![1.0, 1.0, 0.0, 1.0, 0.0];

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed, num_bins, sigma_sq, Some(&oob));

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

    assert!(
        !sparse_w_fixed[0].0.is_empty(),
        "in-bounds sample 0 should have entries"
    );
    assert!(
        sparse_w_fixed[0].1 > 0.0,
        "in-bounds sample 0 should have positive inv_sum_f"
    );
}
