//! Criterion benchmarks for the direct (sparse-loop) Parzen histogram computation.
//!
//! Compares the tensor-based dense path vs. the direct NdArray-optimized path
//! to quantify the speedup from avoiding full `[N, num_bins]` weight matrices.

use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{Transform, TranslationTransform};
use ritk_registration::metric::histogram::{
    build_sparse_w_fixed_transposed, compute_joint_histogram_direct,
    compute_joint_histogram_from_cache_sparse, ParzenJointHistogram, SparseWFixedT,
};

type B = NdArray<f32>;

fn create_test_image(device: &<B as burn::tensor::backend::Backend>::Device) -> Image<B, 3> {
    let n = 32 * 32 * 32;
    let data: Vec<f32> = (0..n).map(|i| i as f32 % 256.0).collect();
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new([32, 32, 32])), device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn bench_parzen_direct(c: &mut Criterion) {
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let fixed = create_test_image(&device);
    let moving = create_test_image(&device);
    let transform = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &device));
    let interpolator = LinearInterpolator::new_zero_pad();
    let histogram = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &device);

    let mut group = c.benchmark_group("ParzenHistogram");

    // 1. Tensor-based joint histogram (current production path — end-to-end)
    group.bench_function("tensor_joint_histogram_32cubed", |b| {
        b.iter(|| {
            black_box(histogram.compute_image_joint_histogram(
                black_box(&fixed),
                black_box(&moving),
                black_box(&transform),
                black_box(&interpolator),
                None,
            ))
        })
    });

    // Prepare normalized values for the direct benchmark
    let fixed_flat = fixed.data().clone().reshape([32 * 32 * 32]);
    let moving_points = transform.transform_points(fixed.index_to_world_tensor(
        ritk_core::image::grid::generate_grid(fixed.shape(), &device),
    ));
    let moving_indices = moving.world_to_index_tensor(moving_points);
    let moving_values = interpolator.interpolate(moving.data(), moving_indices);

    let num_bins = 32;
    let num_bins_f = (num_bins - 1) as f32;
    let fix_scale = num_bins_f / 255.0;
    let mov_scale = num_bins_f / 255.0;
    let fixed_norm = (fixed_flat.clone() * fix_scale).clamp(0.0, num_bins_f);
    let moving_norm = (moving_values.clone() * mov_scale).clamp(0.0, num_bins_f);
    let fixed_norm_data = fixed_norm.into_data();
    let fixed_norm_slice = fixed_norm_data.as_slice::<f32>().unwrap().to_vec();
    let moving_norm_data = moving_norm.into_data();
    let moving_norm_slice = moving_norm_data.as_slice::<f32>().unwrap().to_vec();
    let sigma_sq = 1.0_f32; // sigma_in_bins = 1 for Mattes MI

    // 2. Direct computation (no weight matrices, sample-by-sample accumulation)
    group.bench_function("direct_joint_histogram_32cubed", |b| {
        b.iter(|| {
            let result = compute_joint_histogram_direct(
                black_box(&fixed_norm_slice),
                black_box(&moving_norm_slice),
                num_bins,
                sigma_sq,
                sigma_sq,
                None,
                None,
            );
            black_box(result)
        })
    });

    // 3. Tensor-based joint histogram (values-only path, no image/spatial ops)
    group.bench_function("tensor_values_only_32cubed", |b| {
        let fixed_vals = fixed.data().clone().reshape([32 * 32 * 32]);
        let mov_vals = moving_values.clone();
        b.iter(|| {
            black_box(histogram.compute_joint_histogram(
                black_box(&fixed_vals),
                black_box(&mov_vals),
                None,
            ))
        })
    });

    // 4. Sparse W_fixed^T cache path — build the sparse cache ONCE, then
    // benchmark only the per-iteration histogram computation.
    // This models the CMA-ES hot loop where the fixed image doesn't change.
    let sparse_cache: SparseWFixedT =
        build_sparse_w_fixed_transposed(&fixed_norm_slice, num_bins, sigma_sq, None);
    group.bench_function("direct_sparse_cache_joint_histogram_32cubed", |b| {
        b.iter(|| {
            let result = compute_joint_histogram_from_cache_sparse(
                black_box(&sparse_cache),
                black_box(&moving_norm_slice),
                num_bins,
                sigma_sq,
                None,
                None,
            );
            black_box(result)
        })
    });

    // 5. Production dispatch path — `compute_joint_histogram_dispatch` is
    // pub(crate) so we call `compute_joint_histogram` (the public API that
    // dispatch delegates to). When the `direct-parzen` feature is enabled,
    // dispatch extracts normalized data and calls `compute_joint_histogram_direct`
    // instead; this benchmark measures the tensor-based path for comparison.
    let fixed_vals_1d = fixed.data().clone().reshape([32 * 32 * 32]);
    let moving_vals_1d = moving_values.clone();
    group.bench_function("dispatch_joint_histogram_32cubed", |b| {
        b.iter(|| {
            black_box(histogram.compute_joint_histogram(
                black_box(&fixed_vals_1d),
                black_box(&moving_vals_1d),
                None,
            ))
        })
    });

    // 6. Sparse cache build time — measures the one-time cost of constructing
    // the sparse W_fixed^T from normalized fixed values. In production, this
    // is now lazy (built on first CMA-ES iteration via `fixed_norm`), so this
    // benchmark quantifies the overhead of that first-iteration lazy build.
    group.bench_function("build_sparse_cache_32cubed", |b| {
        b.iter(|| {
            let result = build_sparse_w_fixed_transposed(
                black_box(&fixed_norm_slice),
                num_bins,
                sigma_sq,
                None,
            );
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_parzen_direct);
criterion_main!(benches);
