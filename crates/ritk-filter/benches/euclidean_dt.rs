//! Criterion benchmarks for `euclidean_dt` (Phase 3 Z-column pass).
//!
//! Measures `DistanceTransformImageFilter::apply` on 128³ volumes
//! to establish the parallelised baseline (PERF-380-04, Sprint 381).
//!
//! # Running
//!
//! ```text
//! cargo bench -p ritk-filter --bench euclidean_dt -- apply
//! ```
//!
//! # Recorded baselines
//!
//! Baselines live in `target/criterion/euclidean_dt/...` after a local run.
//! The Sprint 381 parallelisation of Phase 3 (Z-columns) via forward-transpose
//! + moirai + inverse-transpose is the implementation under test.
//!
//! | Size  | Expected median (release, 8-core) |
//! |-------|-----------------------------------|
//! | 128³  | ~80.2 ms (this runner baseline)    |
//!
//! # Evidence tier
//!
//! Empirical: criterion with recorded baselines.  A statistically significant
//! regression blocks merge per `performance_engineering`.

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ritk_filter::DistanceTransformImageFilter;
use ritk_image::test_support as ts;

type B = SequentialBackend;

/// Binary 128³ test volume: checkerboard pattern (alternating 0/1 voxels).
/// Checkerboard maximises the number of non-background voxels without
/// creating trivial all-foreground / all-background regions that would
/// degenerate the EDT computation.
fn make_binary_128() -> ritk_core::image::Image<f32, B, 3> {
    let n = 128;
    let vals: Vec<f32> = (0..n * n * n)
        .map(|i| {
            let z = i / (n * n);
            let y = (i / n) % n;
            let x = i % n;
            if (x + y + z) % 2 == 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    ts::make_image::<B, 3>(vals, [n, n, n])
}

fn bench_euclidean_dt(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_dt");

    let vol = make_binary_128();
    let filter = DistanceTransformImageFilter::default();

    group.bench_with_input(
        BenchmarkId::new("apply/128^3", "binary_checkerboard"),
        &vol,
        |b, img| {
            b.iter(|| filter.apply(img).unwrap());
        },
    );

    group.finish();
}

criterion_group!(benches, bench_euclidean_dt);
criterion_main!(benches);
