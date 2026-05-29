//! Criterion benchmarks for the full MI forward-pass pipeline.
//!
//! Measures the hot-path components used by CMA-ES and RSGD registration:
//!   1. MI forward pass (Mattes, full volume)
//!   2. MI forward pass with 20% stochastic sampling
//!   3. Joint histogram computation only
//!   4. Transform + interpolation chain only

use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ritk_core::image::grid;
use ritk_core::image::Image;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{Transform, TranslationTransform};
use ritk_registration::metric::histogram::ParzenJointHistogram;
use ritk_registration::metric::{Metric, MutualInformation};

type B = NdArray<f32>;

/// Create a 32³ ramp test image with intensity range [0, 255].
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

fn bench_registration_pipeline(c: &mut Criterion) {
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();

    let fixed = create_test_image(&device);
    let moving = create_test_image(&device);
    let transform = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &device));
    let interpolator = LinearInterpolator::new_zero_pad();

    let metric = MutualInformation::<B>::new_mattes(32, 0.0, 255.0, &device);
    let metric_sampled =
        MutualInformation::<B>::new_mattes(32, 0.0, 255.0, &device).with_sampling(0.20);

    let histogram = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 8.0, &device);

    let mut group = c.benchmark_group("RegistrationPipeline");

    // 1. Full MI forward pass (Mattes, 32³ dense)
    group.bench_function("mi_forward_mattes_32cubed", |b| {
        b.iter(|| {
            black_box(Metric::<B, 3>::forward(
                &metric,
                black_box(&fixed),
                black_box(&moving),
                black_box(&transform),
            ))
        })
    });

    // 2. MI forward pass with 20% stochastic sampling
    group.bench_function("mi_forward_mattes_32cubed_sampled_20pct", |b| {
        b.iter(|| {
            black_box(Metric::<B, 3>::forward(
                &metric_sampled,
                black_box(&fixed),
                black_box(&moving),
                black_box(&transform),
            ))
        })
    });

    // 3. Joint histogram only (compute_image_joint_histogram)
    group.bench_function("joint_histogram_32cubed", |b| {
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

    // 4. Transform + interpolation chain only
    group.bench_function("transform_interpolate_32cubed", |b| {
        let fixed_shape = fixed.shape();
        let fixed_indices = grid::generate_grid(fixed_shape, &device);

        b.iter(|| {
            let fixed_points = fixed.index_to_world_tensor(black_box(fixed_indices.clone()));
            let moving_points = transform.transform_points(black_box(fixed_points));
            let moving_indices = moving.world_to_index_tensor(black_box(moving_points));
            black_box(interpolator.interpolate::<3>(moving.data(), black_box(moving_indices)))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_registration_pipeline);
criterion_main!(benches);
