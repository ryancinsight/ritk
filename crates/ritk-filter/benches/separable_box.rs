//! Criterion benchmarks for `separable_box_3d` (GrayscaleDilate/Erode).
//!
//! Measures `GrayscaleDilateFilter::apply` on 128³ volumes at two radii
//! to establish the parallelised baseline (PERF-380-05, Sprint 381).
//!
//! # Running
//!
//! ```text
//! cargo bench -p ritk-filter --bench separable_box -- apply
//! ```
//!
//! # Recorded baselines
//!
//! Baselines live in `target/criterion/separable_box_3d/...` after a local
//! run. The Sprint 381 parallelisation of the X/Y/Z passes via moirai is
//! the implementation under test; compare against any future serial
//! regression to verify the speedup holds.
//!
//! | Size  | r | Expected median (release, 8-core) |
//! |-------|---|-----------------------------------|
//! | 128³  | 2 | ~61.8 ms (this runner baseline)    |
//! | 128³  | 5 | ~63.8 ms (this runner baseline)    |
//!
//! # Evidence tier
//!
//! Empirical: criterion with recorded baselines.  A statistically
//! significant regression (criterion confidence interval) blocks merge
//! per `performance_engineering`.

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ritk_filter::GrayscaleDilation;
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

/// Deterministic 128³ test volume: ramp + sine modulation.
fn make_volume_128() -> Image<f32, B, 3> {
    let n = 128;
    let mut vals = Vec::with_capacity(n * n * n);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let phase = (iz + iy + ix) as f32 * 0.03;
                vals.push((iz as f32) * 0.5 + (iy as f32) * 0.3 + phase.sin());
            }
        }
    }
    Image::from_flat(
        vals,
        [n, n, n],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .expect("benchmark fixture dimensions match its data length")
}

fn bench_separable_box(c: &mut Criterion) {
    let mut group = c.benchmark_group("separable_box_3d");

    let vol = make_volume_128();

    for &radius in &[2_usize, 5_usize] {
        let filter = GrayscaleDilation::new(radius);
        group.bench_with_input(BenchmarkId::new("apply/128^3", radius), &vol, |b, img| {
            b.iter(|| filter.apply_native(img, &B::default()).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_separable_box);
criterion_main!(benches);
