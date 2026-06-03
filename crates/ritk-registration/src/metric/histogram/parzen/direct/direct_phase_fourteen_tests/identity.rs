//! SPARSE-329-01: Direct↔sparse numerical identity tests.

use super::super::*;

#[test]
fn direct_sparse_numerically_identical() {
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

    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let diff = (*d - *s).abs();
        let rel = diff / d.abs().max(1e-10);
        assert!(
            rel < 1e-3 || diff < 1e-7,
            "bin {i}: direct={d}, sparse={s}, diff={diff}, rel={rel}"
        );
    }

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

#[test]
fn direct_sparse_different_sigma_per_axis() {
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
