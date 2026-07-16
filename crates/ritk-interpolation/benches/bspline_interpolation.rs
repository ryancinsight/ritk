//! Criterion benchmarks for BSpline interpolation hot-path.
//!
//! Covers the Sprint 294 regression scenario: 1000 random points on a 64³ volume.

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ritk_image::tensor::{Tensor, TensorData};
use ritk_interpolation::{BSplineInterpolator, Interpolator};

type B = SequentialBackend;

/// 3-D benchmark: 1000 points on a 64×64×64 f32 volume.
/// This is the primary Sprint 294 regression scenario (~850× speedup).
fn bench_bspline_3d(c: &mut Criterion) {
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let n = 64usize;
    let n_pts = 1000usize;

    // Build a 64³ volume with gradient values.
    let data_vec: Vec<f32> = (0..n * n * n).map(|i| i as f32).collect();
    let data = Tensor::<B, 3>::from_data((data_vec, [n, n, n]), &device);

    // Interior query points: cycle through the [1, 62] interior to avoid OOB.
    let pts: Vec<f32> = (0..n_pts)
        .flat_map(|p| {
            let c = (p % (n - 2) + 1) as f32;
            [c, c, c]
        })
        .collect();
    let indices = Tensor::<B, 2>::from_data((pts, [n_pts, 3]), &device);

    let interp = BSplineInterpolator::new();

    let mut group = c.benchmark_group("BSpline");
    group.bench_with_input(BenchmarkId::new("3d_64³_1000pts", ""), &(), |b, _| {
        b.iter(|| interp.interpolate(&data, indices.clone()));
    });
    group.finish();
}

/// 2-D benchmark: 1000 points on a 64×64 f32 image.
fn bench_bspline_2d(c: &mut Criterion) {
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let n = 64usize;
    let n_pts = 1000usize;

    // Build a 64² image with gradient values.
    let data_vec: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
    let data = Tensor::<B, 2>::from_data((data_vec, [n, n]), &device);

    // Interior query points.
    let pts: Vec<f32> = (0..n_pts)
        .flat_map(|p| {
            let c = (p % (n - 2) + 1) as f32;
            [c, c]
        })
        .collect();
    let indices = Tensor::<B, 2>::from_data((pts, [n_pts, 2]), &device);

    let interp = BSplineInterpolator::new();

    let mut group = c.benchmark_group("BSpline");
    group.bench_with_input(BenchmarkId::new("2d_64²_1000pts", ""), &(), |b, _| {
        b.iter(|| interp.interpolate(&data, indices.clone()));
    });
    group.finish();
}

criterion_group!(benches, bench_bspline_3d, bench_bspline_2d);
criterion_main!(benches);
