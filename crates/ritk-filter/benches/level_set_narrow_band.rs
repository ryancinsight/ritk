//! Criterion benchmarks for the SparseField narrow-band level-set solver.
//!
//! `AntiAliasBinaryImageFilter` carries the ITK SparseField narrow band as
//! `layers: Vec<Vec<usize>>` (`nl = ndim`, so `num = 2 * ndim + 1` rows) and
//! mutates it with front-inserts (`insert(0, ..)`) and front-removals
//! (`remove(0)`). This bench tracks `apply` on a spherical boundary so the
//! active band — and therefore the layer churn — scales with the *surface*,
//! not the volume, which is the regime the container actually sees.
//!
//! # Running
//!
//! ```text
//! cargo bench -p ritk-filter --bench level_set_narrow_band
//! cargo bench -p ritk-filter --bench level_set_narrow_band -- apply/32
//! ```
//!
//! # Why this fixture
//!
//! A ramp or a uniform volume produces no zero crossing, hence no active layer
//! and no narrow band at all — the container is never touched. A sphere gives a
//! large active surface with a single connected boundary, which is the worst
//! case for the `layers` bookkeeping while staying a valid 3-D input.

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ritk_filter::AntiAliasBinaryImageFilter;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

/// A binary sphere inside a cube: `1.0` inside radius `0.4 * n`, else `0.0`.
/// Deterministic (no RNG) so bench runs are bitwise-comparable across sessions.
fn make_blob(n: usize) -> Image<f32, B, 3> {
    let centre = (n as f32 - 1.0) / 2.0;
    let radius = 0.4 * n as f32;
    let r2 = radius * radius;
    let mut vals = Vec::with_capacity(n * n * n);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let dz = iz as f32 - centre;
                let dy = iy as f32 - centre;
                let dx = ix as f32 - centre;
                vals.push(if dz * dz + dy * dy + dx * dx <= r2 {
                    1.0
                } else {
                    0.0
                });
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

fn bench_level_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("anti_alias_binary");

    for n in [16usize, 24, 32] {
        let img = make_blob(n);
        group.bench_with_input(BenchmarkId::new("apply", n), &img, |b, img| {
            b.iter(|| AntiAliasBinaryImageFilter::default().apply(img));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_level_set);
criterion_main!(benches);
