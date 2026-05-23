//! Criterion benchmarks for Parzen joint histogram (Mutual Information) computation.
//!
//! Benchmarks the core hot-path used by MutualInformation metrics during registration.

use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use criterion::{criterion_group, criterion_main, Criterion};
use ritk_registration::metric::histogram::ParzenJointHistogram;

type B = NdArray<f32>;

/// Benchmark Parzen joint histogram with 1000 intensity sample pairs, 32 bins.
fn bench_parzen_joint_histogram(c: &mut Criterion) {
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let n = 1000usize;

    // Deterministic intensity samples spanning [0, 255] — no RNG needed.
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i % 256) as f32).collect();
    // Offset pattern to create a non-trivial joint distribution.
    let moving_vec: Vec<f32> = (0..n).map(|i| ((i * 3 + 17) % 256) as f32).collect();

    let fixed = Tensor::<B, 1>::from_data(TensorData::new(fixed_vec, [n]), &device);
    let moving = Tensor::<B, 1>::from_data(TensorData::new(moving_vec, [n]), &device);

    // 32 bins, intensity range [0, 255], sigma = 8.0 (roughly one bin-width).
    let histogram = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 8.0, &device);

    let mut group = c.benchmark_group("MutualInformation");

    group.bench_function("parzen_joint_histogram_1000pts_32bins", |b| {
        b.iter(|| histogram.compute_joint_histogram(&fixed, &moving, None));
    });

    // Also benchmark with independent moving range (elastix-style binning).
    let histogram_sep =
        let histogram_sep = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 8.0, &device).with_separate_moving_range(0.0, 255.0);

    group.bench_function(
        "parzen_joint_histogram_1000pts_32bins_separate_range",
        |b| {
            b.iter(|| histogram_sep.compute_joint_histogram(&fixed, &moving, None));
        },
    );

    group.finish();
}

criterion_group!(benches, bench_parzen_joint_histogram);
criterion_main!(benches);
