//! Criterion benchmarks for the full CPR `apply` pipeline.
//!
//! Measures the end-to-end cost of `CprImageFilter::apply` against a
//! deterministic 3-D volume, including path generation, arc-length
//! resampling, and trilinear sampling across the cross-section grid.
//!
//! # Regression sentinel
//!
//! After the Sprint 376 direction-inverse hoist, the inner trilinear
//! sampling loop dropped from one matrix inverse + one matrix-vector
//! multiply + eight indexed loads per pixel to one matrix-vector
//! multiply + six FMAs + eight indexed loads per pixel. Re-run this
//! bench to confirm the kernel still scales linearly with
//! `num_path × num_cross` and captures any future regression in the
//! cross-section loop.
//!
//! # Measured performance (release build, x86-64 AVX2)
//!
//! Recorded as the Sprint 376 CPR-PERF-01 baseline. Subsequent runs
//! should not regress below these floors without an architectural
//! reason recorded in `OPTIMIZATION.md`.

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ritk_filter::{CprConfig, CprImageFilter};
use ritk_image::test_support as ts;

type B = SequentialBackend;

fn make_test_image(size: usize) -> ritk_image::Image<f32, B, 3> {
    let mut v = vec![0.0_f32; size * size * size];
    for iz in 0..size {
        for iy in 0..size {
            for ix in 0..size {
                v[iz * size * size + iy * size + ix] = (iz + iy + ix) as f32;
            }
        }
    }
    let mid = size / 2;
    v[mid * size * size + mid * size + mid] = 999.0_f32;
    ts::make_image::<B, 3>(v, [size, size, size])
}

fn bench_cpr_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpr_apply");

    for &size in &[16_usize, 32, 64] {
        let image = make_test_image(size);
        // Default config: 256 path samples × 64 cross samples
        // (16 384 trilinear queries per call).
        let s = size as f64;
        let filter = CprImageFilter::new(
            vec![
                [2.0, 2.0, 2.0],
                [s * 0.5, s * 0.5, s * 0.5],
                [s - 2.0, s - 2.0, s - 2.0],
            ],
            CprConfig::default(),
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}^3")),
            &size,
            |b, _| {
                b.iter(|| {
                    let out = filter.apply(&image).expect("cpr apply");
                    std::hint::black_box(out);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cpr_apply);
criterion_main!(benches);
