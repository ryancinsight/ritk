//! Lazy sparse cache construction test.

use super::super::*;

#[cfg(feature = "direct-parzen")]
#[test]
fn lazy_sparse_cache_built_on_first_access() {
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::interpolation::LinearInterpolator;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

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

    let _first =
        hist.compute_image_joint_histogram(&fixed_img, &moving_img, &translation, &interp, None);

    let _sparse_built_after_first = {
        let cache = hist.cache.lock().unwrap();
        let cache_inner = cache.as_ref().expect("cache must exist after first call");
        cache_inner.sparse_w_fixed.is_some()
    };

    let second =
        hist.compute_image_joint_histogram(&fixed_img, &moving_img, &translation, &interp, None);

    {
        let cache = hist.cache.lock().unwrap();
        let cache_inner = cache.as_ref().expect("cache must exist after second call");
        assert!(
            cache_inner.sparse_w_fixed.is_some(),
            "sparse_w_fixed must be built after second compute_image_joint_histogram call"
        );
        assert!(
            cache_inner.fixed_norm.is_none(),
            "fixed_norm should be consumed (None) after sparse cache is built"
        );
    }

    let second_data = second.into_data();
    let second_slice = second_data.as_slice::<f32>().unwrap();
    let sum: f32 = second_slice.iter().sum();
    assert!(
        sum > 0.0,
        "second-call histogram must be non-zero, got sum={sum}"
    );
}
