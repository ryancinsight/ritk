use super::*;

// ─── Phase Four property tests (TEST-316-05) ─────────────────────────────────

#[cfg(feature = "direct-parzen")]
#[test]
fn sparse_w_fixed_deterministic() {
    // TEST-316-05: Verify that building the sparse W_fixed^T cache twice from
    // the same input produces identical results (determinism under parallel
    // construction via `par_iter_mut`).
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &dev);

    let n = 500;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.51) % 255.0).collect();
    let fixed_tensor = Tensor::<B, 1>::from_floats(fixed.as_slice(), &dev);

    let num_bins = hist.num_bins;
    let sigma_sq_fix = direct::ParzenConfig::from_intensity_sigma(
        hist.parzen_sigma,
        hist.min_intensity,
        hist.max_intensity,
        num_bins,
    )
    .sigma_sq(); // SSOT-319-02
    let fixed_norm = dispatch::normalize_and_extract(
        &fixed_tensor,
        hist.min_intensity,
        hist.max_intensity,
        num_bins,
    );

    // Build the sparse cache twice
    let sparse1 =
        direct::build_sparse_w_fixed_transposed(&fixed_norm, num_bins, sigma_sq_fix, None);
    let sparse2 =
        direct::build_sparse_w_fixed_transposed(&fixed_norm, num_bins, sigma_sq_fix, None);

    // They must be identical
    assert_eq!(
        sparse1.len(),
        sparse2.len(),
        "sparse cache lengths must match"
    );
    for (i, (a, b)) in sparse1.iter().zip(sparse2.iter()).enumerate() {
        // SPARSE-329-01: cache entry is (entries, inv_sum_f) tuple
        assert_eq!(a.0.len(), b.0.len(), "sample {i}: entry count mismatch");
        assert!(
            (a.1 - b.1).abs() < 1e-10,
            "sample {i}: inv_sum_f mismatch a={} b={}",
            a.1,
            b.1
        );
        for (j, (ea, eb)) in a.0.iter().zip(b.0.iter()).enumerate() {
            assert_eq!(ea.bin, eb.bin, "sample {i} entry {j}: bin mismatch");
            let diff = (ea.weight - eb.weight).abs();
            assert!(diff < 1e-10, "sample {i} entry {j}: weight diff={diff}");
        }
    }
}

#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_non_negative_all_entries() {
    // TEST-316-05: All entries in the joint histogram must be non-negative
    // (weights are exp() values, always ≥ 0).
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);

    let n = 200;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 1.27 + 3.0) % 255.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.83 + 7.0) % 255.0).collect();
    let fixed_tensor = Tensor::<B, 1>::from_floats(fixed.as_slice(), &dev);
    let moving_tensor = Tensor::<B, 1>::from_floats(moving.as_slice(), &dev);

    let h = hist.compute_joint_histogram_dispatch(&fixed_tensor, &moving_tensor, None);
    let data = h.into_data();
    let slice = data.as_slice::<f32>().unwrap();

    for (i, &v) in slice.iter().enumerate() {
        assert!(v >= 0.0, "histogram entry {i} is negative: {v}");
    }
}

#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_marginals_sum_correctly() {
    // TEST-316-05: The marginal distributions obtained by summing the joint
    // histogram along each axis must be non-negative and have consistent
    // total weight. Summing along the fixed axis gives the fixed marginal;
    // summing along the moving axis gives the moving marginal. Both must
    // sum to the same total (since each sample contributes equally to both).
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);

    let fixed =
        Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0, 80.0, 210.0, 40.0], &dev);
    let moving =
        Tensor::<B, 1>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0, 90.0, 215.0, 35.0], &dev);

    let h = hist.compute_joint_histogram_dispatch(&fixed, &moving, None);
    let data = h.into_data();
    let slice = data.as_slice::<f32>().unwrap();

    let num_bins = 16;

    // Fixed marginal: sum along moving axis (each row)
    let mut fixed_marginal = vec![0.0f32; num_bins];
    // Moving marginal: sum along fixed axis (each column)
    let mut moving_marginal = vec![0.0f32; num_bins];

    for a in 0..num_bins {
        for b in 0..num_bins {
            let v = slice[a * num_bins + b];
            fixed_marginal[a] += v;
            moving_marginal[b] += v;
        }
    }

    // Both marginals must sum to the same total
    let sum_fixed: f32 = fixed_marginal.iter().sum();
    let sum_moving: f32 = moving_marginal.iter().sum();
    let diff = (sum_fixed - sum_moving).abs();
    assert!(
        diff < 1e-4,
        "marginal sums must match: fixed={sum_fixed}, moving={sum_moving}, diff={diff}"
    );

    // All marginal entries must be non-negative
    for (i, &v) in fixed_marginal.iter().enumerate() {
        assert!(v >= 0.0, "fixed marginal entry {i} is negative: {v}");
    }
    for (i, &v) in moving_marginal.iter().enumerate() {
        assert!(v >= 0.0, "moving marginal entry {i} is negative: {v}");
    }
}
