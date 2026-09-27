//! Criterion benchmark for standard SLIC on a 3-D volume.
//!
//! Every Lloyd iteration rebuilds the grid-cell search index and then scans,
//! for each voxel, the candidate centers the index lists for its cell; this
//! bench times that loop end to end through the public filter.
//!
//! ```text
//! cargo bench -p ritk-segmentation --bench slic
//! ```
//!
//! Wall-clock results are valid only with the process pinned to one core and
//! the host's concurrent load recorded beside them.

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use ritk_image::Image;
use ritk_segmentation::{SlicConfig, SlicSuperpixelFilter};
use ritk_spatial::{Direction, Point, Spacing};
use std::hint::black_box;

const EXTENT: [usize; 3] = [48, 48, 32];
const SUPERPIXELS: usize = 256;

/// Plateaus plus a sinusoidal ramp: deterministic and tie-rich.
fn volume() -> Image<f32, SequentialBackend, 3> {
    let len = EXTENT.iter().product::<usize>();
    let values = (0..len)
        .map(|i| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "fixture indices stay far below f32's exact-integer range"
            )]
            let (plateau, index) = (((i / 997) % 7) as f32, i as f32);
            plateau * 30.0 + (index * 0.013).sin() * 5.0
        })
        .collect();
    Image::from_flat(
        values,
        EXTENT,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .expect("invariant: fixture length is the product of its extent")
}

fn bench_slic(c: &mut Criterion) {
    let image = volume();
    let backend = SequentialBackend;
    let filter = SlicSuperpixelFilter::new(
        SlicConfig::new(SUPERPIXELS)
            .and_then(|config| config.with_max_iterations(5))
            .expect("invariant: nonzero count and iterations are valid"),
    );
    c.bench_function("slic_3d/48x48x32_k256", |bencher| {
        bencher.iter(|| {
            filter
                .apply_native(black_box(&image), &backend)
                .expect("invariant: fixture is finite and 3-D")
        });
    });
}

criterion_group!(benches, bench_slic);
criterion_main!(benches);
