//! Fingerprint validation test for masked cache.

use super::super::*;

#[cfg(feature = "direct-parzen")]
#[test]
fn masked_cache_fingerprint_detects_collision() {
    use ritk_image::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_interpolation::LinearInterpolator;
    use ritk_transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();

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

    let _ = hist.compute_masked_joint_histogram(
        &fixed_img,
        &all_points,
        &moving_img,
        &translation,
        &interp,
        Some(42),
    );

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

    assert!(
        hist.masked_cache.is_populated(),
        "cache should still be populated after valid fingerprint"
    );

    // `normalize_and_extract` returns `Cow<[f32]>`; materialize an owned copy
    // to mutate it for the fingerprint-mismatch assertion.
    let mut wrong_norm = fixed_norm.to_vec();
    wrong_norm[0] += 100.0;
    assert!(
        !hist.validate_masked_cache_fingerprint(&wrong_norm),
        "fingerprint validation should fail with different data"
    );

    assert!(
        !hist.masked_cache.is_populated(),
        "cache should be invalidated after fingerprint mismatch"
    );
}
