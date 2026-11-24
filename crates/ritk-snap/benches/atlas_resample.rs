//! Throughput benchmarks for the atlas-overlay resample path.
//!
//! Measures [`resample_nearest`] (full-grid nearest-neighbour warp) and
//! [`downsample_half`] (block-mean coarse-grid construction) at the MNI152
//! grid used by `examples/atlas_overlay.rs`. Run before and after touching
//! the kernels in `ritk_snap::tools::atlas_overlay`:
//!
//! ```bash
//! cargo bench --offline -p ritk-snap --bench atlas_resample
//! ```

#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use leto::Array3;
use ritk_registration::AffineTransform;
use ritk_snap::tools::atlas_overlay::{downsample_half, resample_nearest};

/// MNI152 grid from `test_data/ants_example/mni152.nii.gz`.
const SHAPE: [usize; 3] = [215, 256, 207];

/// Deterministic synthetic volume: smooth blob plus high-frequency ripple so
/// both kernels touch representative value distributions.
fn synthetic_volume() -> Array3<f64> {
    let mut values = Vec::with_capacity(SHAPE[0] * SHAPE[1] * SHAPE[2]);
    for z in 0..SHAPE[0] {
        for y in 0..SHAPE[1] {
            for x in 0..SHAPE[2] {
                let dx = x as f64 - 128.0;
                let dy = y as f64 - 107.0;
                let dz = z as f64 - 103.0;
                let blob = (-(dx * dx + dy * dy + dz * dz) / 3600.0).exp();
                let ripple = 0.02 * (x as f64 * 1.3).sin() * (y as f64 * 0.7).sin();
                values.push((blob + ripple) * 255.0);
            }
        }
    }
    Array3::from_shape_vec(SHAPE, values).expect("benchmark shape is valid")
}

fn bench_atlas_resample(c: &mut Criterion) {
    let moving = synthetic_volume();
    let identity = AffineTransform::IDENTITY.0;

    let mut group = c.benchmark_group("atlas_resample");
    // Full-grid passes take seconds; keep the sample count low.
    group.sample_size(10);
    group.bench_function("resample_nearest_identity", |b| {
        b.iter(|| {
            resample_nearest(black_box(&moving), black_box(SHAPE), black_box(&identity))
                .expect("benchmark shapes are valid")
        })
    });
    group.bench_function("downsample_half", |b| {
        b.iter(|| downsample_half(black_box(&moving)).expect("benchmark shapes are valid"))
    });
    group.finish();
}

criterion_group!(benches, bench_atlas_resample);
criterion_main!(benches);
