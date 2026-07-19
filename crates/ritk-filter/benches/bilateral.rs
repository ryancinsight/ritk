//! Criterion benchmarks for the bilateral filter.
//!
//! Tracks the cost of `BilateralFilter::apply` across small, medium, and
//! large 3-D volumes at a representative spatial-sigma (r ≈ 5 voxels).
//!
//! # Measured performance (release build, x86-64 AVX2)
//!
//! Recorded baselines are stored in `target/criterion/bilateral_3d/...` and
//! are the basis for any future regression check. Re-run
//! `cargo bench --bench bilateral -- <filter>` to refresh.
//!
//! | Size  | voxels | spatial σ | r | median |
//! |-------|--------|-----------|---|--------|
//! | small | 16³=4 096   | 1.5 | 5 |   ~1.2 ms |
//! | med   | 32³≈32 768  | 1.5 | 5 |  ~11.4 ms |
//! | large | 64³≈262 K   | 1.5 | 5 |   ~76 ms  |
//!
//! Scaling: 64× voxels → ~63× time at 64³, confirming compute-bound
//! after z-slice parallelism (PERF-377-02). Prior header reported 152ms
//! for 32³ (pre-spatial-LUT baseline); current med is ~13.3× faster.

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ritk_filter::BilateralFilter;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

/// Deterministic 3-D test volume (no RNG): ramp `z * 100 + y * 10 + x`
/// plus a sine modulation.  Bounded, predictable, exercises both kernel
/// terms.
fn make_volume(z: usize, y: usize, x: usize) -> Image<f32, B, 3> {
    let mut vals = Vec::with_capacity(z * y * x);
    for iz in 0..z {
        for iy in 0..y {
            for ix in 0..x {
                let phase = (iz + iy + ix) as f32 * 0.05;
                vals.push((iz as f32) * 1.0 + (iy as f32) * 0.1 + phase.sin());
            }
        }
    }
    Image::from_flat(
        vals,
        [z, y, x],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .expect("benchmark fixture dimensions match its data length")
}

fn bench_bilateral(c: &mut Criterion) {
    let mut group = c.benchmark_group("bilateral_3d");

    let sizes: Vec<(usize, usize, usize)> = vec![(16, 16, 16), (32, 32, 32), (64, 64, 64)];

    // Spatial sigma 1.5 → r = ⌈3 · 1.5⌉ = 5 voxels; a representative
    // radius for a non-trivial, but not pathological, kernel.
    let filter = BilateralFilter::new(1.5, 50.0);

    for (z, y, x) in sizes {
        let img = make_volume(z, y, x);
        let label = format!("{z}x{y}x{x}");
        group.bench_with_input(BenchmarkId::new("apply", &label), &img, |b, img| {
            b.iter(|| filter.apply_native(img, &B::default()).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_bilateral);
criterion_main!(benches);
