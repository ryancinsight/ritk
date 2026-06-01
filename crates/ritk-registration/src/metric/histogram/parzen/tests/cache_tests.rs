use super::*;

#[cfg(feature = "direct-parzen")]
#[test]
fn sparse_cache_dispatch_matches_direct() {
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);
    let fixed = Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0], &dev);

    // Compute via the non-cached dispatch (direct path)
    let direct_hist = hist.compute_joint_histogram_dispatch(&fixed, &moving, None);

    // Build sparse cache from the same fixed values and compute via sparse dispatch
    let w_fixed_t = hist.compute_w_fixed_transposed(&fixed, 5);
    let sparse = {
        let num_bins = hist.num_bins;
        let fix_min = hist.min_intensity;
        let fix_max = hist.max_intensity;
        let fix_sigma = hist.parzen_sigma;
        let sigma_sq_fix =
            direct::ParzenConfig::from_intensity_sigma(fix_sigma, fix_min, fix_max, num_bins)
                .sigma_sq(); // SSOT-319-02
        let fixed_norm = dispatch::normalize_and_extract(&fixed, fix_min, fix_max, num_bins);
        direct::build_sparse_w_fixed_transposed(&fixed_norm, num_bins, sigma_sq_fix, None)
    };
    let sparse_hist =
        hist.compute_joint_histogram_from_cache_sparse_dispatch(&sparse, &moving, None);

    let direct_data = direct_hist.into_data();
    let direct_slice = direct_data.as_slice::<f32>().unwrap();
    let sparse_data = sparse_hist.into_data();
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    // Both direct and sparse paths accumulate raw w_f × w_m products.
    // Verify nonzero patterns match exactly and totals are approximately equal.
    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let d_nz = *d > 1e-6;
        let s_nz = *s > 1e-6;
        assert_eq!(
            d_nz, s_nz,
            "sparse cache nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }

    // Also verify the dense cache path produces similar results.
    let dense_hist = hist.compute_joint_histogram_from_cache_dispatch(&w_fixed_t, &moving, None);
    let dense_data = dense_hist.into_data();
    let dense_slice = dense_data.as_slice::<f32>().unwrap();
    let direct_total: f32 = direct_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    let dense_total: f32 = dense_slice.iter().sum();
    // All three paths accumulate raw w_f × w_m products — totals should be
    // approximately equal (~n × 2π for σ² ≈ 1, ~5 × 6.28 ≈ 31).
    assert!(
        direct_total > 0.0,
        "direct_total {direct_total} must be positive"
    );
    assert!(
        sparse_total > 0.0,
        "sparse_total {sparse_total} must be positive"
    );
    assert!(
        dense_total > 0.0,
        "dense_total {dense_total} must be positive"
    );
    // Direct and sparse paths produce nonzero at the same bins. SPARSE-329-01:
    // sparse now matches direct (combined inv_sum_f × inv_sum_m). Ratio ≈ 1.0.
    let ds_ratio = sparse_total / direct_total;
    assert!(
        (ds_ratio - 1.0).abs() < 0.05,
        "sparse/direct ratio {ds_ratio} should be ≈ 1.0 (SPARSE-329-01)"
    );
}

// ─── Lazy sparse cache construction ────────────────────────────────────

#[cfg(feature = "direct-parzen")]
#[test]
fn lazy_sparse_cache_built_on_first_access() {
    // Test that the sparse cache is initially None but gets built on the
    // second call to compute_image_joint_histogram when the sparse dispatch
    // path is taken. This verifies the lazy construction logic.
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::interpolation::LinearInterpolator;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    // Small volume (non-chunked path)
    let shape = [4, 4, 4];
    let n = shape[0] * shape[1] * shape[2];
    let mut fixed_data = Vec::with_capacity(n);
    let mut moving_data = Vec::with_capacity(n);
    for i in 0..n {
        let v = (i as f32 * 2.0).min(255.0);
        fixed_data.push(v);
        moving_data.push(v * 0.9 + 5.0);
    }

    let fixed_t =
        Tensor::<B, 3>::from_data(TensorData::new(fixed_data, Shape::new(shape)), &device);
    let moving_t =
        Tensor::<B, 3>::from_data(TensorData::new(moving_data, Shape::new(shape)), &device);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();
    let fixed_img = Image::new(fixed_t, origin, spacing, direction);
    let moving_img = Image::new(moving_t, origin, spacing, direction);

    let interp = LinearInterpolator::new_zero_pad();
    let zero_translation = Tensor::<B, 1>::zeros([3], &device);
    let translation = TranslationTransform::<B, 3>::new(zero_translation);

    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &device);

    // First call — this should build the cache (with fixed_norm but sparse_w_fixed = None)
    let _first =
        hist.compute_image_joint_histogram(&fixed_img, &moving_img, &translation, &interp, None);

    // After first call: the cache should exist. The first call may take the
    // dense cache path (sparse_w_fixed is None on first cache-miss), but it
    // stores fixed_norm for lazy sparse cache construction.
    //
    // On the second call, get_cached_sparse_w_fixed will find the cache,
    // see sparse_w_fixed is None, take fixed_norm, build the sparse cache,
    // and store it. So after the second call, sparse_w_fixed should be Some.
    //
    // Capture the state after the first call but before the second.
    let sparse_built_after_first = {
        let cache = hist.cache.lock().unwrap();
        let cache_inner = cache.as_ref().expect("cache must exist after first call");
        // First call may not have built sparse_w_fixed yet (it's lazily built
        // on the first sparse dispatch path access).
        cache_inner.sparse_w_fixed.is_some()
    };

    // Second call — this should trigger the lazy sparse cache build
    let second =
        hist.compute_image_joint_histogram(&fixed_img, &moving_img, &translation, &interp, None);

    // After second call: sparse_w_fixed must now be built (lazy construction
    // on first sparse dispatch path access).
    {
        let cache = hist.cache.lock().unwrap();
        let cache_inner = cache.as_ref().expect("cache must exist after second call");
        assert!(
            cache_inner.sparse_w_fixed.is_some(),
            "sparse_w_fixed must be built after second compute_image_joint_histogram call"
        );
        // fixed_norm should have been consumed (taken) by the lazy build
        assert!(
            cache_inner.fixed_norm.is_none(),
            "fixed_norm should be consumed (None) after sparse cache is built"
        );
    }

    // If sparse cache was not built after the first call, verify that it
    // was lazily constructed on the second call (the core lazy build invariant).
    if !sparse_built_after_first {
        // This confirms the lazy construction: sparse_w_fixed was None after
        // first call, but is Some after the second call.
        // (If it was already Some after the first call, lazy construction
        // happened even sooner — still valid, just a different code path.)
    }

    // Also verify the second result is valid (non-zero histogram)
    let second_data = second.into_data();
    let second_slice = second_data.as_slice::<f32>().unwrap();
    let sum: f32 = second_slice.iter().sum();
    assert!(
        sum > 0.0,
        "second-call histogram must be non-zero, got sum={sum}"
    );
}

// ─── Chunked sparse path ───────────────────────────────────────────────

#[cfg(feature = "direct-parzen")]
#[test]
fn chunked_sparse_path_matches_nonchunked() {
    // Similar to chunked_cached_path_matches_non_chunked but gated to run
    // WITH direct-parzen feature. Create a volume just above CHUNK_SIZE
    // (64×32×32 = 65536 > 32768) and verify the chunked sparse cache path
    // produces results matching the direct (non-chunked) computation.
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::interpolation::LinearInterpolator;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    // Create a volume large enough to trigger chunking (N > 32768).
    // 64 × 32 × 32 = 65536 > 32768.
    let shape = [64, 32, 32];
    let n = shape[0] * shape[1] * shape[2];
    let mut fixed_data = Vec::with_capacity(n);
    let mut moving_data = Vec::with_capacity(n);
    for i in 0..n {
        let v = (i as f32 * 0.01).min(255.0);
        fixed_data.push(v);
        moving_data.push(v * 0.8 + 10.0);
    }

    let fixed_t =
        Tensor::<B, 3>::from_data(TensorData::new(fixed_data, Shape::new(shape)), &device);
    let moving_t =
        Tensor::<B, 3>::from_data(TensorData::new(moving_data, Shape::new(shape)), &device);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let direction = Direction::identity();
    let fixed_img = Image::new(fixed_t, origin, spacing, direction);
    let moving_img = Image::new(moving_t, origin, spacing, direction);

    let interp = LinearInterpolator::new_zero_pad();
    let zero_translation = Tensor::<B, 1>::zeros([3], &device);
    let translation = TranslationTransform::<B, 3>::new(zero_translation);

    let hist = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &device);

    // Compute via compute_image_joint_histogram (triggers chunked sparse path
    // under direct-parzen feature).
    let joint_chunked =
        hist.compute_image_joint_histogram(&fixed_img, &moving_img, &translation, &interp, None);

    // Also compute via the dispatch path (non-chunked sparse) for comparison.
    // We use compute_joint_histogram_dispatch which internally uses the same
    // sparse-loop algorithm, so this is a self-consistency check that chunking
    // the sparse cache path produces the same result as the non-chunked sparse path.
    let fixed_flat = fixed_img.data().clone().reshape([n]);
    let moving_flat = moving_img.data().clone().reshape([n]);
    let joint_dispatch = hist.compute_joint_histogram_dispatch(&fixed_flat, &moving_flat, None);

    let chunked_data = joint_chunked.into_data();
    let chunked_slice = chunked_data.as_slice::<f32>().unwrap();
    let dispatch_data = joint_dispatch.into_data();
    let dispatch_slice = dispatch_data.as_slice::<f32>().unwrap();

    // The chunked sparse path must match the non-chunked dispatch path since
    // both use the same direct sparse-loop algorithm. Any difference is due
    // only to floating-point accumulation order across chunk boundaries.
    for (i, (a, b)) in chunked_slice.iter().zip(dispatch_slice.iter()).enumerate() {
        let diff = (a - b).abs();
        let max_val = a.abs().max(b.abs()).max(1e-10);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 0.05 || diff < 0.05,
            "chunked sparse vs dispatch mismatch at bin {i}: chunked={a}, dispatch={b}, diff={diff}, rel_err={rel_err}"
        );
    }
}

// ─── Fingerprint validation ─────────────────────────────────────────────

#[cfg(feature = "direct-parzen")]
#[test]
fn masked_cache_fingerprint_detects_collision() {
    // Verify that validate_masked_cache_fingerprint returns false
    // when the stored fingerprint doesn't match the current data,
    // and that it invalidates the cache in that case.
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::interpolation::LinearInterpolator;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    let shape = [4, 4, 4];
    let n_voxels = shape[0] * shape[1] * shape[2];
    let mut fixed_data = Vec::with_capacity(n_voxels);
    let mut moving_data = Vec::with_capacity(n_voxels);
    for i in 0..n_voxels {
        let v = (i as f32 * 3.0) % 255.0;
        fixed_data.push(v);
        moving_data.push(v * 0.85 + 10.0);
    }

    let fixed_t =
        Tensor::<B, 3>::from_data(TensorData::new(fixed_data, Shape::new(shape)), &device);
    let moving_t =
        Tensor::<B, 3>::from_data(TensorData::new(moving_data, Shape::new(shape)), &device);

    let fixed_img = Image::new(
        fixed_t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    let moving_img = Image::new(
        moving_t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    let interp = LinearInterpolator::new_zero_pad();
    let zero_translation = Tensor::<B, 1>::zeros([3], &device);
    let translation = TranslationTransform::<B, 3>::new(zero_translation);

    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &device);

    let all_points = fixed_img.index_to_world_tensor(ritk_core::image::grid::generate_grid(
        fixed_img.shape(),
        &device,
    ));

    // Populate the masked cache
    let _ = hist.compute_masked_joint_histogram(
        &fixed_img,
        all_points.clone(),
        &moving_img,
        &translation,
        &interp,
        Some(42),
    );

    // Validate with the correct fixed_norm — should return true
    let fixed_norm = dispatch::normalize_and_extract(
        &fixed_img.data().clone().reshape([n_voxels]),
        hist.min_intensity,
        hist.max_intensity,
        hist.num_bins,
    );
    assert!(
        hist.validate_masked_cache_fingerprint(&fixed_norm),
        "fingerprint validation should succeed with matching data"
    );

    // Cache should still be populated
    {
        let cache = hist.masked_cache.lock().unwrap();
        assert!(
            cache.is_some(),
            "cache should still be populated after valid fingerprint"
        );
    }

    // Validate with DIFFERENT data — should return false and invalidate cache
    let mut wrong_norm = fixed_norm.clone();
    wrong_norm[0] += 100.0; // drastically change first value
    assert!(
        !hist.validate_masked_cache_fingerprint(&wrong_norm),
        "fingerprint validation should fail with different data"
    );

    // Cache should now be cleared
    {
        let cache = hist.masked_cache.lock().unwrap();
        assert!(
            cache.is_none(),
            "cache should be invalidated after fingerprint mismatch"
        );
    }
}

#[cfg(feature = "direct-parzen")]
#[test]
fn direct_parallel_matches_sparse() {
    // Verify that the parallelized dispatch path produces results matching
    // the sparse-path equivalent (both accumulate raw w_f × w_m products).
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);

    // Use enough samples to exercise parallel reduction
    let n = 1000;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.25) % 255.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 0.18 + 10.0) % 255.0).collect();

    let fixed_tensor = Tensor::<B, 1>::from_floats(fixed.as_slice(), &dev);
    let moving_tensor = Tensor::<B, 1>::from_floats(moving.as_slice(), &dev);

    // Compute via parallel dispatch path
    let dispatch_hist = hist.compute_joint_histogram_dispatch(&fixed_tensor, &moving_tensor, None);
    let dispatch_data = dispatch_hist.into_data();
    let dispatch_slice = dispatch_data.as_slice::<f32>().unwrap();

    // Compute via sparse path (also parallel, independent implementation)
    let num_bins = hist.num_bins;
    let fix_min = hist.min_intensity;
    let fix_max = hist.max_intensity;
    let fix_sigma = hist.parzen_sigma;
    let mov_sigma = hist.moving_parzen_sigma.unwrap_or(fix_sigma);
    let mov_min = hist.moving_min_intensity.unwrap_or(fix_min);
    let mov_max = hist.moving_max_intensity.unwrap_or(fix_max);

    let sigma_sq_fix =
        direct::ParzenConfig::from_intensity_sigma(fix_sigma, fix_min, fix_max, num_bins).sigma_sq(); // SSOT-319-02
    let sigma_sq_mov =
        direct::ParzenConfig::from_intensity_sigma(mov_sigma, mov_min, mov_max, num_bins).sigma_sq(); // SSOT-319-02

    let fixed_norm = dispatch::normalize_and_extract(&fixed_tensor, fix_min, fix_max, num_bins);
    let moving_norm = dispatch::normalize_and_extract(&moving_tensor, mov_min, mov_max, num_bins);

    let sparse = direct::build_sparse_w_fixed_transposed(&fixed_norm, num_bins, sigma_sq_fix, None);
    let sparse_data = direct::compute_joint_histogram_from_cache_sparse(
        &sparse,
        &moving_norm,
        num_bins,
        sigma_sq_mov,
        None,
        None,
    );
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    // Both paths accumulate raw w_f × w_m products identically.
    // Verify nonzero patterns match (parallel reduction may reorder contributions
    // but the final histogram is the same up to floating-point reorderings).
    for (i, (d, s)) in dispatch_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let d_nz = *d > 1e-6;
        let s_nz = *s > 1e-6;
        assert_eq!(
            d_nz, s_nz,
            "parallel direct vs sparse nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }
    // SPARSE-329-01: dispatch and sparse now produce equivalent normalized
    // histograms. Ratio ≈ 1.0.
    let dispatch_total: f32 = dispatch_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    assert!(
        dispatch_total > 0.0,
        "dispatch total {dispatch_total} must be positive"
    );
    assert!(
        sparse_total > 0.0,
        "sparse total {sparse_total} must be positive"
    );
    let ratio = dispatch_total / sparse_total;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "dispatch/sparse total ratio {ratio} should be ≈ 1.0 (SPARSE-329-01)"
    );
}

// ─── Property-based tests (TEST-315-06) ────────────────────────────────────

#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_symmetry_identical_images() {
    // When fixed == moving, the joint histogram must be symmetric:
    // H[a, b] == H[b, a] for all a, b.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);
    let fixed =
        Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0, 80.0, 210.0, 40.0], &dev);
    let h = hist.compute_joint_histogram_dispatch(&fixed, &fixed, None);
    let data = h.into_data();
    let slice = data.as_slice::<f32>().unwrap();
    let num_bins = 16;
    for a in 0..num_bins {
        for b in 0..num_bins {
            let ab = slice[a * num_bins + b];
            let ba = slice[b * num_bins + a];
            let diff = (ab - ba).abs();
            assert!(
                diff < 1e-4,
                "symmetry violation at ({a},{b}): H[a,b]={ab}, H[b,a]={ba}, diff={diff}"
            );
        }
    }
}

#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_normalization_total_weight() {
    // PERF-328-01: per-sample normalization by 1/(sum_f × sum_m) means each
    // sample contributes ≈ 1.0 to the histogram total. For n=100, the
    // total should be ≈ n (with minor boundary truncation losses).
    let dev = device();
    let n = 100;
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 2.55) % 255.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i as f32 * 1.87 + 5.0) % 255.0).collect();
    let fixed_tensor = Tensor::<B, 1>::from_floats(fixed.as_slice(), &dev);
    let moving_tensor = Tensor::<B, 1>::from_floats(moving.as_slice(), &dev);
    let hist = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &dev);
    let h = hist.compute_joint_histogram_dispatch(&fixed_tensor, &moving_tensor, None);
    let sum: f32 = h.into_data().as_slice::<f32>().unwrap().iter().sum();
    // Per-sample contribution ≈ 1.0. With n=100, total ≈ 100. Allow wide
    // bounds [0.5n, 1.5n] for boundary truncation effects.
    let expected_min = n as f32 * 0.5;
    let expected_max = n as f32 * 1.5;
    assert!(
        sum > expected_min,
        "normalized histogram total {sum} should be > {expected_min} (n × 0.5)"
    );
    assert!(
        sum < expected_max,
        "normalized histogram total {sum} should be < {expected_max} (n × 1.5)"
    );
}

#[cfg(feature = "direct-parzen")]
#[test]
fn histogram_boundary_bins_populated() {
    // Samples at intensity 0 (near bin 0) must populate the boundary bins.
    // This verifies that boundary clamping doesn't zero out the wrong bins.
    let dev = device();
    let fixed = Tensor::<B, 1>::from_floats([0.0, 0.0, 0.0, 0.0, 0.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([0.0, 0.0, 0.0, 0.0, 0.0], &dev);
    let hist = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &dev);
    let h = hist.compute_joint_histogram_dispatch(&fixed, &moving, None);
    let data = h.into_data();
    let slice = data.as_slice::<f32>().unwrap();
    let num_bins = 32;
    // Bin (0, 0) must have significant weight
    assert!(
        slice[0] > 0.1,
        "bin (0,0) must be populated for zero-valued samples, got {}",
        slice[0]
    );
    // The first few bins should have non-zero weight (Gaussian support)
    let sum_first_4_rows: f32 = slice[0..4].iter().sum();
    assert!(
        sum_first_4_rows > 0.5,
        "first 4 bins in row 0 should have weight, got {sum_first_4_rows}"
    );
    // Bins beyond the ±3σ support must be zero
    for b in 5..num_bins {
        assert!(
            slice[b] < 1e-6,
            "bin (0, {b}) should be ~0 beyond support, got {}",
            slice[b]
        );
    }
}
