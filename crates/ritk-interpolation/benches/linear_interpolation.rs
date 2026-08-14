//! Criterion baseline for the linear interpolation hot path.
//!
//! # Why this exists
//!
//! PERF-694-01 moved this kernel's per-axis scratch off the heap, and
//! PERF-694-02 then found that the same commit had also repacked three flat
//! arrays into one struct — a change measured at **2x slower**. Both went in
//! without a baseline, so nothing would have caught the regression.
//!
//! # Methodology (performance_engineering)
//!
//! - Inputs are pinned: a deterministic gradient volume and a fixed sweep of
//!   fractional query coordinates, so successive runs measure the same work.
//! - The timed closure returns its result and takes its inputs through
//!   `black_box`, so the kernel cannot be const-folded away.
//! - Reported metric: median wall time for a full batch of points.
//! - Compare against a stored baseline with
//!   `cargo bench -p ritk-interpolation -- --baseline <name>`.
//!
//! # Recorded baseline
//!
//! Windows, release profile, quiet host (2026-08-14). Criterion medians:
//!
//! | Benchmark | Median |
//! | --- | --- |
//! | `linear_interpolation_3d/1000` | 27.98 us |
//! | `linear_interpolation_3d/100000` | 2.531 ms |
//! | `linear_interpolation_bounds/extend` | 1.617 ms |
//! | `linear_interpolation_bounds/zero_pad` | 696.0 us |
//!
//! Two things worth noting from that first pass. Per-point cost is flat from
//! 1e3 to 1e5 points (28.0 ns against 25.3 ns), so nothing falls off a cache
//! cliff across that range. And `ZeroPad` is 2.3x faster than `Extend` on the
//! half-out-of-bounds batch, because it returns before the eight-corner loop
//! rather than clamping into it — so overhanging a resample is cheaper under
//! `ZeroPad`, not merely different.
//!
//! # Interpreting a delta here
//!
//! This kernel is allocator-sensitive, and PERF-694-02 recorded heap timings
//! spanning 3.3 ms to 31.5 ms across runs of *identical* code depending on
//! allocator state. Criterion's resampling absorbs much of that, but a result
//! within a few percent is not evidence of anything. Treat a change below
//! roughly 20% as noise unless it reproduces across separate invocations.

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ritk_image::tensor::Tensor;
use ritk_interpolation::{Interpolator, LinearInterpolator};
use std::hint::black_box;

type B = SequentialBackend;

/// Deterministic gradient volume of `side^3` samples.
fn volume(side: usize) -> Tensor<f32, B> {
    let data: Vec<f32> = (0..side * side * side).map(|i| (i % 4096) as f32).collect();
    Tensor::<f32, B>::from_slice([side, side, side], &data)
}

/// `count` query points on a fixed fractional sweep of the interior.
///
/// Fractional coordinates matter: an integer coordinate collapses every corner
/// weight to 0 or 1, which is not the arithmetic the kernel runs in practice.
fn query_points(count: usize, side: usize) -> Tensor<f32, B> {
    let span = (side - 2) as f32;
    let coords: Vec<f32> = (0..count)
        .flat_map(|p| {
            let t = (p % 997) as f32 / 997.0;
            let base = 1.0 + t * span;
            [base, base * 0.5 + 1.0, base * 0.25 + 1.0]
        })
        .collect();
    Tensor::<f32, B>::from_slice([count, 3], &coords)
}

/// Batch size sweep on a fixed volume.
///
/// Resampling drives this kernel once per output voxel, so the per-point cost
/// is what matters; the sweep shows whether it stays flat as the batch grows
/// past cache residency.
fn bench_linear_3d(c: &mut Criterion) {
    const SIDE: usize = 64;
    let data = volume(SIDE);
    let interpolator = LinearInterpolator::new();

    let mut group = c.benchmark_group("linear_interpolation_3d");
    for &count in &[1_000usize, 100_000] {
        let indices = query_points(count, SIDE);
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                let result = interpolator.interpolate(black_box(&data), indices.clone());
                black_box(result)
            });
        });
    }
    group.finish();
}

/// Out-of-bounds handling on the same batch.
///
/// `ZeroPad` returns early for a coordinate outside the volume while `Extend`
/// clamps and runs the full corner loop, so the pair bounds how much the
/// policy costs when a resample overhangs its input.
fn bench_bounds_policy(c: &mut Criterion) {
    const SIDE: usize = 64;
    let data = volume(SIDE);
    // Half the points sit outside, so each policy's path is exercised.
    let coords: Vec<f32> = (0..50_000usize)
        .flat_map(|p| {
            let inside = p % 2 == 0;
            let base = if inside { 1.0 + (p % 60) as f32 } else { -5.0 };
            [base, base, base]
        })
        .collect();
    let indices = Tensor::<f32, B>::from_slice([50_000, 3], &coords);

    let mut group = c.benchmark_group("linear_interpolation_bounds");
    for (policy, name) in [
        (LinearInterpolator::new(), "extend"),
        (LinearInterpolator::new_zero_pad(), "zero_pad"),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                let result = policy.interpolate(black_box(&data), indices.clone());
                black_box(result)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_linear_3d, bench_bounds_policy);
criterion_main!(benches);
