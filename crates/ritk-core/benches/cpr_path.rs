//! Criterion benchmarks for CPR path generation.
//!
//! Compares scalar `generate_path` vs segment-oriented `generate_path_batch`
//! across small, medium, and large input sizes.
//!
//! # Measured performance (release build, x86-64 AVX2)
//!
//! | Size | scalar | batch | speedup |
//! |------|--------|-------|---------|
//! | small (5 cp, 256 samples) | 1.86 µs | 1.04 µs | 1.8× |
//! | medium (10 cp, 2560 samples) | 18.2 µs | 10.2 µs | 1.8× |
//! | large (20 cp, 25600 samples) | 167 µs | 116 µs | 1.4× |

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ritk_core::filter::{generate_path, generate_path_batch};

/// Deterministic control points for benchmarking (no RNG).
///
/// Produces a smooth 3-D curve with increasing curvature.
fn control_points_small() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 2.0],
        [20.0, 10.0, 5.0],
        [30.0, 5.0, 8.0],
        [40.0, 0.0, 10.0],
    ]
}

fn control_points_medium() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [5.0, 3.0, 1.0],
        [10.0, 5.0, 2.0],
        [15.0, 8.0, 3.0],
        [20.0, 10.0, 5.0],
        [25.0, 9.0, 6.0],
        [30.0, 5.0, 8.0],
        [35.0, 2.0, 9.0],
        [40.0, 0.0, 10.0],
        [45.0, -2.0, 11.0],
    ]
}

fn control_points_large() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [2.5, 1.5, 0.5],
        [5.0, 3.0, 1.0],
        [7.5, 4.5, 1.5],
        [10.0, 5.0, 2.0],
        [12.5, 7.0, 2.5],
        [15.0, 8.0, 3.0],
        [17.5, 9.0, 4.0],
        [20.0, 10.0, 5.0],
        [22.5, 9.5, 5.5],
        [25.0, 9.0, 6.0],
        [27.5, 7.0, 7.0],
        [30.0, 5.0, 8.0],
        [32.5, 3.5, 8.5],
        [35.0, 2.0, 9.0],
        [37.5, 1.0, 9.5],
        [40.0, 0.0, 10.0],
        [42.5, -1.0, 10.5],
        [45.0, -2.0, 11.0],
        [47.5, -3.0, 11.5],
    ]
}

fn bench_cpr_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpr_path");

    let sizes: Vec<(&str, Vec<[f64; 3]>, usize)> = vec![
        ("small_5cp_256s", control_points_small(), 256),
        ("medium_10cp_2560s", control_points_medium(), 2560),
        ("large_20cp_25600s", control_points_large(), 25600),
    ];

    for (label, cps, num_samples) in &sizes {
        group.bench_with_input(
            BenchmarkId::new("scalar", label),
            &(cps, *num_samples),
            |b, &(ref cps, ns)| {
                b.iter(|| generate_path(cps, ns));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch", label),
            &(cps, *num_samples),
            |b, &(ref cps, ns)| {
                b.iter(|| generate_path_batch(cps, ns));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cpr_path);
criterion_main!(benches);
