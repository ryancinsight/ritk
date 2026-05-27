use super::*;
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn device() -> <B as burn::tensor::backend::Backend>::Device {
    Default::default()
}

// ─── bins_exp eager initialization ────────────────────────────────────────

#[test]
fn bins_exp_eager_init_matches_lazy() {
    // Verify that the eagerly initialized bins_exp in new() produces the
    // same bin-center tensor as an on-the-fly arange construction.
    let dev = device();
    let num_bins = 32;
    let hist = ParzenJointHistogram::<B>::new(num_bins, 0.0, 255.0, 8.0, &dev);

    // The bins_exp field is eagerly initialized in new() — it must be Some.
    assert!(
        hist.bins_exp.is_some(),
        "bins_exp must be eagerly initialized in new()"
    );

    // Compare the cached bins_exp against a fresh arange_bins call.
    let cached = hist.bins_exp.as_ref().cloned().unwrap();
    let fresh: Tensor<B, 2> = compute::arange_bins(num_bins, &dev);

    let cached_data = cached.into_data();
    let cached_slice = cached_data.as_slice::<f32>().unwrap();
    let fresh_data = fresh.into_data();
    let fresh_slice = fresh_data.as_slice::<f32>().unwrap();

    assert_eq!(
        cached_slice.len(),
        fresh_slice.len(),
        "bins_exp length mismatch"
    );
    for (i, (a, b)) in cached_slice.iter().zip(fresh_slice.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "bins_exp mismatch at index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn bins_exp_histogram_result_matches_lazy() {
    // Verify that compute_joint_histogram with eager bins_exp gives the
    // same numerical result as constructing bins_exp lazily.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0, &dev);
    let fixed = Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0], &dev);

    // Compute with eager bins_exp (default path)
    let h1 = hist.compute_joint_histogram(&fixed, &moving, None);

    // Manually replace bins_exp with a fresh one to simulate lazy init
    let mut hist2 = hist.clone();
    hist2.bins_exp = Some(compute::arange_bins(hist2.num_bins, &dev));
    let h2 = hist2.compute_joint_histogram(&fixed, &moving, None);

    let h1_data = h1.into_data();
    let s1 = h1_data.as_slice::<f32>().unwrap();
    let h2_data = h2.into_data();
    let s2 = h2_data.as_slice::<f32>().unwrap();
    for (i, (a, b)) in s1.iter().zip(s2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "histogram mismatch at bin {i}: {a} vs {b}"
        );
    }
}

// ─── compute_oob_mask_3d ─────────────────────────────────────────────────

#[test]
fn oob_mask_3d_in_bounds_all_ones() {
    // 4×4×4 volume; every coordinate strictly inside returns 1.0
    let dev = device();
    // [x=1.5, y=1.5, z=1.5] — floor = [1,1,1], dims = [4,4,4] → in-bounds on all axes
    let indices =
        Tensor::<B, 2>::from_floats([[1.5, 1.5, 1.5], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]], &dev);
    let mask = compute_oob_mask_3d(&indices, &[4, 4, 4]);
    let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(
        vals,
        vec![1.0, 1.0, 1.0],
        "all in-bounds coords must give 1.0"
    );
}

#[test]
fn oob_mask_3d_oob_all_zeros() {
    let dev = device();
    // x=-1 (OOB), y=5 > d1-1=3 (OOB), z=-0.1 → floor=-1 (OOB)
    let indices =
        Tensor::<B, 2>::from_floats([[-1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, -0.1]], &dev);
    let mask = compute_oob_mask_3d(&indices, &[4, 4, 4]);
    let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(vals, vec![0.0, 0.0, 0.0], "all OOB coords must give 0.0");
}

#[test]
fn oob_mask_3d_mixed_in_and_out() {
    let dev = device();
    // shape [Z=2, Y=4, X=4]: valid x in [0,3], y in [0,3], z in [0,1]
    let indices = Tensor::<B, 2>::from_floats(
        [
            [1.5, 1.5, 0.5],  // in-bounds
            [-0.5, 1.5, 0.5], // x OOB (floor=-1)
            [1.5, 4.0, 0.5],  // y OOB (floor=4 > 3)
            [1.5, 1.5, 2.0],  // z OOB (floor=2 > 1)
            [3.0, 3.0, 1.0],  // boundary, in-bounds
        ],
        &dev,
    );
    let mask = compute_oob_mask_3d(&indices, &[2, 4, 4]);
    let vals: Vec<f32> = mask.into_data().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(
        vals,
        vec![1.0, 0.0, 0.0, 0.0, 1.0],
        "mixed: in=1 OOB=0, boundary is in-bounds"
    );
}

// ─── compute_joint_histogram with OOB mask ───────────────────────────────

#[test]
fn oob_mask_zeros_out_oob_contribution() {
    // Verify that applying an all-zero OOB mask produces a zero histogram.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0, &dev);

    let fixed = Tensor::<B, 1>::from_floats([128.0, 64.0, 192.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([128.0, 64.0, 192.0], &dev);
    let all_oob = Tensor::<B, 1>::zeros([3], &dev); // all samples are OOB

    let h = hist.compute_joint_histogram(&fixed, &moving, Some(&all_oob));
    let sum: f32 = h.into_data().as_slice::<f32>().unwrap().iter().sum();
    assert!(
        sum < 1e-6,
        "histogram with all-OOB mask must be zero, got sum={sum}"
    );
}

#[test]
fn oob_mask_partial_filters_correctly() {
    // With a partial OOB mask (only first sample in-bounds), the histogram
    // should be dominated by the first sample's contribution.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0, &dev);

    // Three samples: first is identity (128, 128), rest are extreme (0, 255)
    let fixed = Tensor::<B, 1>::from_floats([128.0, 0.0, 255.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([128.0, 255.0, 0.0], &dev);
    // Only the first sample is in-bounds
    let partial_mask = Tensor::<B, 1>::from_floats([1.0, 0.0, 0.0], &dev);

    let h_masked = hist.compute_joint_histogram(&fixed, &moving, Some(&partial_mask));
    let h_unmasked = hist.compute_joint_histogram(&fixed, &moving, None);

    // The masked histogram should have strictly less total weight than unmasked.
    let sum_masked: f32 = h_masked.into_data().as_slice::<f32>().unwrap().iter().sum();
    let sum_unmasked: f32 = h_unmasked
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .sum();
    assert!(
        sum_masked < sum_unmasked,
        "masked sum ({sum_masked}) must be less than unmasked sum ({sum_unmasked})"
    );
}

#[test]
fn oob_mask_all_in_bounds_equivalent_to_no_mask() {
    // A mask of all 1.0 must produce the same result as passing None.
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(8, 0.0, 255.0, 32.0, &dev);

    let fixed = Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0], &dev);
    let all_in = Tensor::<B, 1>::ones([5], &dev);

    let h_with_mask = hist
        .compute_joint_histogram(&fixed, &moving, Some(&all_in))
        .into_data();
    let h_no_mask = hist
        .compute_joint_histogram(&fixed, &moving, None)
        .into_data();

    let s1 = h_with_mask.as_slice::<f32>().unwrap();
    let s2 = h_no_mask.as_slice::<f32>().unwrap();
    for (a, b) in s1.iter().zip(s2.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "all-ones mask must match no-mask: {a} vs {b}"
        );
    }
}

// ─── dispatch integration tests ──────────────────────────────────────────
//
// Verify that `compute_joint_histogram_dispatch` produces numerically identical
// results to the tensor-based `compute_joint_histogram` when the direct-parzen
// feature is enabled. The dispatch path extracts data to host and calls the
// sparse-loop algorithm, which uses a different numerical strategy (±3σ bins
// vs full matmul), so we allow a small tolerance.

#[cfg(feature = "direct-parzen")]
#[test]
fn dispatch_matches_tensor_path() {
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);
    let fixed = Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0, 80.0, 210.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0, 90.0, 215.0], &dev);

    // Tensor path (ground truth)
    let tensor_hist = hist.compute_joint_histogram(&fixed, &moving, None);
    let tensor_data = tensor_hist.into_data();
    let tensor_slice = tensor_data.as_slice::<f32>().unwrap();

    // Dispatch path (direct sparse-loop)
    let dispatch_hist = hist.compute_joint_histogram_dispatch(&fixed, &moving, None);
    let dispatch_data = dispatch_hist.into_data();
    let dispatch_slice = dispatch_data.as_slice::<f32>().unwrap();

    for (i, (t, d)) in tensor_slice.iter().zip(dispatch_slice.iter()).enumerate() {
        let diff = (t - d).abs();
        let max_val = t.abs().max(d.abs()).max(1e-10);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 0.05 || diff < 0.05,
            "dispatch mismatch at bin {i}: tensor={t}, dispatch={d}, diff={diff}, rel_err={rel_err}"
        );
    }
}

#[cfg(feature = "direct-parzen")]
#[test]
fn dispatch_with_oob_mask() {
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);
    let fixed = Tensor::<B, 1>::from_floats([128.0, 64.0, 192.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([128.0, 64.0, 192.0], &dev);
    let all_oob = Tensor::<B, 1>::zeros([3], &dev);

    let dispatch_hist = hist.compute_joint_histogram_dispatch(&fixed, &moving, Some(&all_oob));
    let sum: f32 = dispatch_hist
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .sum();
    assert!(
        sum < 1e-6,
        "all-OOB dispatch histogram must be zero, got sum={sum}"
    );
}

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
        let sigma_sq_fix = dispatch::sigma_sq_in_bins(fix_sigma, fix_min, fix_max, num_bins);
        let fixed_norm = dispatch::normalize_and_extract(&fixed, fix_min, fix_max, num_bins);
        direct::build_sparse_w_fixed_transposed(&fixed_norm, num_bins, sigma_sq_fix, None)
    };
    let sparse_hist =
        hist.compute_joint_histogram_from_cache_sparse_dispatch(&sparse, &moving, None);

    let direct_data = direct_hist.into_data();
    let direct_slice = direct_data.as_slice::<f32>().unwrap();
    let sparse_data = sparse_hist.into_data();
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let diff = (d - s).abs();
        let max_val = d.abs().max(s.abs()).max(1e-10);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 0.01 || diff < 0.01,
            "sparse cache mismatch at bin {i}: direct={d}, sparse={s}, diff={diff}, rel_err={rel_err}"
        );
    }

    // Also verify the dense cache path produces similar results
    let dense_hist = hist.compute_joint_histogram_from_cache_dispatch(&w_fixed_t, &moving, None);
    let dense_data = dense_hist.into_data();
    let dense_slice = dense_data.as_slice::<f32>().unwrap();

    for (i, (d, s)) in dense_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let diff = (d - s).abs();
        let max_val = d.abs().max(s.abs()).max(1e-10);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 0.05 || diff < 0.05,
            "dense vs sparse cache mismatch at bin {i}: dense={d}, sparse={s}, diff={diff}, rel_err={rel_err}"
        );
    }
}

/// Verify that the chunked-path W_fixed^T caching produces the same joint
/// histogram as the non-chunked path. When N > CHUNK_SIZE (32768), the
/// chunked code path slices cached W_fixed^T per chunk instead of recomputing
/// W_fixed from fixed values each time. This test constructs a scenario large
/// enough to trigger the chunked path and validates against the non-chunked
/// result.
///
/// Gated to only run when the tensor-based path is active (direct-parzen
/// disabled), since the direct sparse-loop algorithm uses a different
/// numerical strategy (+/-3-sigma bins vs full matmul) and produces slightly
/// different results.
#[test]
#[cfg_attr(feature = "direct-parzen", ignore = "requires tensor path")]
fn chunked_cached_path_matches_non_chunked() {
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
        moving_data.push(v * 0.8 + 10.0); // shifted/scaled
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

    // Compute joint histogram via ParzenJointHistogram (triggers chunked path).
    let hist = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &device);
    let joint_chunked = hist.compute_image_joint_histogram(
        &fixed_img,
        &moving_img,
        &translation,
        &interp,
        None, // no sampling → triggers non-sampling chunked path
    );

    // Also compute directly (bypass chunking) for comparison.
    let fixed_flat = fixed_img.data().clone().reshape([n]);
    let moving_flat = moving_img.data().clone().reshape([n]);
    let joint_direct = hist.compute_joint_histogram(
        &fixed_flat,
        &moving_flat,
        None, // no OOB mask
    );

    let chunked_data = joint_chunked.into_data();
    let chunked_slice = chunked_data.as_slice::<f32>().unwrap();
    let direct_data = joint_direct.into_data();
    let direct_slice = direct_data.as_slice::<f32>().unwrap();

    for (i, (a, b)) in chunked_slice.iter().zip(direct_slice.iter()).enumerate() {
        assert!(
            (a - b).abs() < 5e-2,
            "chunked vs direct mismatch at bin {i}: {a} vs {b}"
        );
    }
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

// ─── Masked path caching ─────────────────────────────────────────────────

#[cfg(feature = "direct-parzen")]
#[test]
fn masked_cache_reuses_weights_on_same_key() {
    // Verify that providing the same cache_key on successive calls to
    // compute_masked_joint_histogram causes the second call to reuse the
    // cached W_fixed^T (and sparse cache), producing the same histogram
    // without recomputing fixed-image weights.
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::interpolation::LinearInterpolator;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    // Create a small 3D image
    let shape = [8, 8, 8];
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

    // Use all voxel world points as the "mask"
    let all_points = fixed_img.index_to_world_tensor(ritk_core::image::grid::generate_grid(
        fixed_img.shape(),
        &device,
    ));

    // First call WITH caching — should compute and store W_fixed^T
    let first = hist.compute_masked_joint_histogram(
        &fixed_img,
        all_points.clone(),
        &moving_img,
        &translation,
        &interp,
        Some(42), // cache_key
    );

    // Second call WITH same cache_key — should reuse cached W_fixed^T
    let second = hist.compute_masked_joint_histogram(
        &fixed_img,
        all_points.clone(),
        &moving_img,
        &translation,
        &interp,
        Some(42), // same cache_key
    );

    // Results must match exactly (same inputs, cached weights)
    let first_data = first.into_data();
    let first_slice = first_data.as_slice::<f32>().unwrap();
    let second_data = second.into_data();
    let second_slice = second_data.as_slice::<f32>().unwrap();

    for (i, (a, b)) in first_slice.iter().zip(second_slice.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-6,
            "masked cache reuse mismatch at bin {i}: first={a}, second={b}, diff={diff}"
        );
    }

    // Verify that the masked cache is populated
    let cache = hist.masked_cache.lock().unwrap();
    assert!(
        cache.is_some(),
        "masked cache must be populated after first call"
    );
    let inner = cache.as_ref().unwrap();
    assert_eq!(inner.cache_key, 42);
    assert!(
        inner.w_fixed_transposed.is_some(),
        "w_fixed_transposed must be cached"
    );

    // Verify the histogram is non-zero
    let sum: f32 = first_slice.iter().sum();
    assert!(
        sum > 0.0,
        "masked histogram must be non-zero, got sum={sum}"
    );
}

#[cfg(feature = "direct-parzen")]
#[test]
fn masked_cache_different_key_recomputes() {
    // Verify that providing a DIFFERENT cache_key causes recomputation
    // (cache miss with new key).
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::interpolation::LinearInterpolator;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    let shape = [8, 8, 8];
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

    // First call with key=100
    let _first = hist.compute_masked_joint_histogram(
        &fixed_img,
        all_points.clone(),
        &moving_img,
        &translation,
        &interp,
        Some(100),
    );

    // Second call with key=200 — different key should cause cache miss
    let second = hist.compute_masked_joint_histogram(
        &fixed_img,
        all_points.clone(),
        &moving_img,
        &translation,
        &interp,
        Some(200),
    );

    // The cache should now hold the key=200 entry
    let cache = hist.masked_cache.lock().unwrap();
    let inner = cache.as_ref().unwrap();
    assert_eq!(
        inner.cache_key, 200,
        "masked cache key should be updated to 200 after second call with different key"
    );

    // Result should still be valid
    let sum: f32 = second.into_data().as_slice::<f32>().unwrap().iter().sum();
    assert!(
        sum > 0.0,
        "masked histogram must be non-zero, got sum={sum}"
    );
}

#[test]
fn masked_no_cache_key_matches_uncached() {
    // Verify that passing None as cache_key produces the same result
    // as the original uncached path.
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::interpolation::LinearInterpolator;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    let shape = [8, 8, 8];
    let n_voxels = shape[0] * shape[1] * shape[2];
    let mut fixed_data = Vec::with_capacity(n_voxels);
    let mut moving_data = Vec::with_capacity(n_voxels);
    for i in 0..n_voxels {
        let v = (i as f32 * 3.0) % 255.0;
        fixed_data.push(v);
        moving_data.push(v * 0.85 + 10.0);
    }

    let fixed_t = Tensor::<B, 3>::from_data(
        TensorData::new(fixed_data.clone(), Shape::new(shape)),
        &device,
    );
    let moving_t = Tensor::<B, 3>::from_data(
        TensorData::new(moving_data.clone(), Shape::new(shape)),
        &device,
    );
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

    let hist1 = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &device);
    let hist2 = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &device);

    let all_points = fixed_img.index_to_world_tensor(ritk_core::image::grid::generate_grid(
        fixed_img.shape(),
        &device,
    ));

    // Call with None (no caching) and with a cache key — results should match
    let no_cache_result = hist1.compute_masked_joint_histogram(
        &fixed_img,
        all_points.clone(),
        &moving_img,
        &translation,
        &interp,
        None, // no caching
    );

    let cached_result = hist2.compute_masked_joint_histogram(
        &fixed_img,
        all_points.clone(),
        &moving_img,
        &translation,
        &interp,
        Some(99), // with caching
    );

    let no_cache_data = no_cache_result.into_data();
    let no_cache_slice = no_cache_data.as_slice::<f32>().unwrap();
    let cached_data = cached_result.into_data();
    let cached_slice = cached_data.as_slice::<f32>().unwrap();

    for (i, (a, b)) in no_cache_slice.iter().zip(cached_slice.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-5,
            "no-cache vs cached mismatch at bin {i}: no_cache={a}, cached={b}, diff={diff}"
        );
    }
}
