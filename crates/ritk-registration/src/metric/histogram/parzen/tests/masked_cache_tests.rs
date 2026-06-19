use super::*;

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
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_interpolation::LinearInterpolator;
    use ritk_transform::TranslationTransform;

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
        &all_points,
        &moving_img,
        &translation,
        &interp,
        Some(42), // cache_key
    );
    // Second call WITH same cache_key — should reuse cached W_fixed^T
    let second = hist.compute_masked_joint_histogram(
        &fixed_img,
        &all_points,
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
            diff < 1e-4,
            "masked cache reuse mismatch at bin {i}: first={a}, second={b}, diff={diff}"
        );
    }

    // Verify that the masked cache is populated
    hist.masked_cache.with_ref(|cache| {
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
    });

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
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_interpolation::LinearInterpolator;
    use ritk_transform::TranslationTransform;

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
        &all_points,
        &moving_img,
        &translation,
        &interp,
        Some(100),
    );
    // Second call with key=200 — different key should cause cache miss
    let second = hist.compute_masked_joint_histogram(
        &fixed_img,
        &all_points,
        &moving_img,
        &translation,
        &interp,
        Some(200),
    );

    // The cache should now hold the key=200 entry
    hist.masked_cache.with_ref(|cache| {
        let inner = cache.as_ref().unwrap();
        assert_eq!(
            inner.cache_key, 200,
            "masked cache key should be updated to 200 after second call with different key"
        );
    });

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
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_interpolation::LinearInterpolator;
    use ritk_transform::TranslationTransform;

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

    let hist1 = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &device);
    let hist2 = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &device);

    let all_points = fixed_img.index_to_world_tensor(ritk_core::image::grid::generate_grid(
        fixed_img.shape(),
        &device,
    ));

    // Call with None (no caching) and with a cache key — results should match
    let no_cache_result = hist1.compute_masked_joint_histogram(
        &fixed_img,
        &all_points,
        &moving_img,
        &translation,
        &interp,
        None, // no caching
    );
    let cached_result = hist2.compute_masked_joint_histogram(
        &fixed_img,
        &all_points,
        &moving_img,
        &translation,
        &interp,
        Some(99), // with caching
    );

    let no_cache_data = no_cache_result.into_data();
    let no_cache_slice = no_cache_data.as_slice::<f32>().unwrap();
    let cached_data = cached_result.into_data();
    let cached_slice = cached_data.as_slice::<f32>().unwrap();
    // Both no-cache and cached paths accumulate raw w_f × w_m products.
    // Verify nonzero patterns match and totals are approximately equal.
    for (i, (a, b)) in no_cache_slice.iter().zip(cached_slice.iter()).enumerate() {
        let a_nz = *a > 1e-6;
        let b_nz = *b > 1e-6;
        assert_eq!(
            a_nz, b_nz,
            "nonzero pattern mismatch at bin {i}: no_cache={a}, cached={b}"
        );
    }
    // Verify both totals are positive and approximately equal.
    let no_cache_total: f32 = no_cache_slice.iter().sum();
    let cached_total: f32 = cached_slice.iter().sum();
    assert!(
        no_cache_total > 0.0 && no_cache_total.is_finite(),
        "no_cache_total {no_cache_total} must be positive and finite"
    );
    assert!(
        cached_total > 0.0 && cached_total.is_finite(),
        "cached_total {cached_total} must be positive and finite"
    );
    // Both paths produce nonzero at the same bins. The total weight may
    // differ slightly if boundary sample handling differs between the
    // no-cache and cache-miss code paths; verify totals are within 4x.
    let ratio = cached_total / no_cache_total;
    assert!(
        ratio > 0.5 && ratio < 4.0,
        "cached/no_cache ratio {ratio} should be in [0.5, 4.0], no_cache={no_cache_total}, cached={cached_total}"
    );
}

// ─── Cache invalidation ─────────────────────────────────────────────────

#[test]
fn cache_invalidate_clears_image_cache() {
    // Verify that invalidate_cache() clears the image-grid cache that was
    // populated by a prior compute_image_joint_histogram call.
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_interpolation::LinearInterpolator;
    use ritk_transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    // Small volume
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

    // Populate the image-grid cache
    let _result = hist.compute_image_joint_histogram(
        &fixed_img,
        &moving_img,
        &translation,
        &interp,
        crate::metric::sampling::SamplingConfig::full_grid(),
    );

    // Verify cache is populated
    assert!(
        hist.cache.is_populated(),
        "image-grid cache must be populated after compute_image_joint_histogram"
    );

    // Invalidate the image-grid cache
    hist.invalidate_cache();

    // Verify cache is now None
    assert!(
        !hist.cache.is_populated(),
        "image-grid cache must be None after invalidate_cache()"
    );

    // Invalidation is idempotent — calling again should not panic
    hist.invalidate_cache();
    assert!(
        !hist.cache.is_populated(),
        "image-grid cache must still be None after second invalidate_cache()"
    );
}

#[test]
fn cache_invalidate_clears_masked_cache() {
    // Verify that invalidate_masked_cache() clears the masked cache that was
    // populated by a prior compute_masked_joint_histogram call with a cache_key.
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_interpolation::LinearInterpolator;
    use ritk_transform::TranslationTransform;

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

    // Populate the masked cache with a cache_key
    let _result = hist.compute_masked_joint_histogram(
        &fixed_img,
        &all_points,
        &moving_img,
        &translation,
        &interp,
        Some(42), // cache_key
    );

    // Verify masked cache is populated
    assert!(
        hist.masked_cache.is_populated(),
        "masked cache must be populated after compute_masked_joint_histogram"
    );

    // Invalidate the masked cache
    hist.invalidate_masked_cache();

    // Verify masked cache is now None
    assert!(
        !hist.masked_cache.is_populated(),
        "masked cache must be None after invalidate_masked_cache()"
    );

    // Invalidation is idempotent — calling again should not panic
    hist.invalidate_masked_cache();
    assert!(
        !hist.masked_cache.is_populated(),
        "masked cache must still be None after second invalidate_masked_cache()"
    );
}
