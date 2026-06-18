//! Criterion benchmarks for the median filter.
//!
//! Tracks the cost of `MedianFilter::apply` across small, medium, and
//! large 3-D volumes at a representative radius (r = 2 voxels, which gives
//! a 125-voxel neighbourhood).
//!
//! # Running
//!
//! ```text
//! cargo bench -p ritk-filter --bench median -- apply/16x16x16
//! cargo bench -p ritk-filter --bench median -- apply/32x32x32
//! cargo bench -p ritk-filter --bench median -- apply/64x64x64
//! ```
//!
//! # Recorded baselines
//!
//! Baselines live in `target/criterion/median_3d/...` after a local run.
//! Compare against prior sessions for regression checks
//! (`perf_engineering`: criterion regression blocks merge).
//!
//! # Purpose for Sprint 377 carry-forward
//!
//! The Huang sliding-histogram optimisation (PERF-377-01 full, deferred
//! to next agent who owns `median.rs`) reduces the per-voxel cost from
//! `O(r³)` to `O(r²)` by maintaining a 2-D running column histogram as
//! the X window slides. The *measured* baseline from this bench is the
//! threshold that an alternative implementation must beat before it can
//! be considered a win — empirical-tier evidence per the evidence
//! hierarchy.

use burn_ndarray::NdArray;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ritk_core::image::Image;
use ritk_filter::MedianFilter;
use ritk_image::test_support as ts;

type B = NdArray<f32>;

/// Deterministic 3-D test volume (no RNG): ramp `z + y/10 + x/100`
/// plus a sine modulation. Bounded, predictable, independent of RNG
/// seeding so bench runs are bitwise-comparable across sessions.
fn make_volume(z: usize, y: usize, x: usize) -> Image<B, 3> {
    let mut vals = Vec::with_capacity(z * y * x);
    for iz in 0..z {
        for iy in 0..y {
            for ix in 0..x {
                let phase = (iz + iy + ix) as f32 * 0.05;
                vals.push((iz as f32) * 1.0 + (iy as f32) * 0.1 + phase.sin());
            }
        }
    }
    ts::make_image::<B, 3>(vals, [z, y, x])
}

fn bench_median(c: &mut Criterion) {
    let mut group = c.benchmark_group("median_3d");

    let sizes: Vec<(usize, usize, usize)> = vec![(16, 16, 16), (32, 32, 32), (64, 64, 64)];

    // Radius 2 → (2r+1)³ = 125-voxel neighbourhood. Comparison point
    // for any sliding-histogram alternative: O(r²) = 25 ops/voxel vs
    // O(r³) = 125 ops/voxel; theoretical max speedup ≈ 5×.
    let filter = MedianFilter::new(2);

    for (z, y, x) in sizes {
        let img = make_volume(z, y, x);
        let label = format!("{z}x{y}x{x}");
        group.bench_with_input(BenchmarkId::new("apply", &label), &img, |b, img| {
            b.iter(|| filter.apply(img).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_median);
criterion_main!(benches);
