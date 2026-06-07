//! Cache integration tests for Parzen joint histogram cache dispatch.

#![allow(clippy::needless_range_loop)]

use super::super::*;

#[cfg(feature = "direct-parzen")]
#[test]
fn sparse_cache_dispatch_matches_direct() {
    let dev = device();
    let hist = ParzenJointHistogram::<B>::new(16, 0.0, 255.0, 255.0 / 16.0, &dev);
    let fixed = Tensor::<B, 1>::from_floats([50.0, 128.0, 200.0, 30.0, 175.0], &dev);
    let moving = Tensor::<B, 1>::from_floats([60.0, 130.0, 195.0, 25.0, 180.0], &dev);

    let direct_hist = hist.compute_joint_histogram_dispatch(&fixed, &moving, None);

    let w_fixed_t = hist.compute_w_fixed_transposed(&fixed, 5);
    let sparse = {
        let num_bins = hist.num_bins;
        let fix_min = hist.min_intensity;
        let fix_max = hist.max_intensity;
        let fix_sigma = hist.parzen_sigma;
        let sigma_sq_fix =
            direct::ParzenConfig::from_intensity_sigma(fix_sigma, fix_min, fix_max, num_bins)
                .sigma_sq();
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
        let d_nz = *d > 1e-6;
        let s_nz = *s > 1e-6;
        assert_eq!(
            d_nz, s_nz,
            "sparse cache nonzero pattern mismatch at bin {i}: direct={d}, sparse={s}"
        );
    }

    let dense_hist = hist.compute_joint_histogram_from_cache_dispatch(&w_fixed_t, &moving, None);
    let dense_data = dense_hist.into_data();
    let dense_slice = dense_data.as_slice::<f32>().unwrap();
    let direct_total: f32 = direct_slice.iter().sum();
    let sparse_total: f32 = sparse_slice.iter().sum();
    let dense_total: f32 = dense_slice.iter().sum();
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
    let ds_ratio = sparse_total / direct_total;
    assert!(
        (ds_ratio - 1.0).abs() < 0.05,
        "sparse/direct ratio {ds_ratio} should be ≈ 1.0 (SPARSE-329-01)"
    );
}

#[cfg(feature = "direct-parzen")]
#[test]
fn chunked_sparse_path_matches_nonchunked() {
    use burn::tensor::{Shape, TensorData};
    use ritk_core::image::Image;
    use ritk_core::interpolation::LinearInterpolator;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_core::transform::TranslationTransform;

    type B = burn_ndarray::NdArray<f32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

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

    let joint_chunked =
        hist.compute_image_joint_histogram(&fixed_img, &moving_img, &translation, &interp, None);

    let fixed_flat = fixed_img.data().clone().reshape([n]);
    let moving_flat = moving_img.data().clone().reshape([n]);
    let joint_dispatch = hist.compute_joint_histogram_dispatch(&fixed_flat, &moving_flat, None);

    let chunked_data = joint_chunked.into_data();
    let chunked_slice = chunked_data.as_slice::<f32>().unwrap();
    let dispatch_data = joint_dispatch.into_data();
    let dispatch_slice = dispatch_data.as_slice::<f32>().unwrap();

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
