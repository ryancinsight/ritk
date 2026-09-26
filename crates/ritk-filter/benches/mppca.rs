//! Criterion benchmark for MP-PCA denoising (`MpPcaDenoiser::denoise`).
//!
//! The workload is the `dwidenoise` geometry at reduced extent: 60 volumes,
//! so the derived window is the 4³ = 64-voxel cube and every window
//! decomposes a 60 × 60 Gram matrix, over a 16 × 16 × 10 image (2560
//! windows). Per-window cost is independent of the image size, so the
//! full-size time scales with the voxel count; a 64 × 64 × 40 series is 64
//! times this workload.
//!
//! # Running
//!
//! ```text
//! cargo bench -p ritk-filter --bench mppca
//! ```
//!
//! # Time budget
//!
//! One iteration is about 0.5 s on one core and under 0.1 s in parallel on
//! the 24-thread reference host; ten flat samples over a 5 s window keep the
//! benchmark near 10 s of the suite's 300 s bound.

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use ritk_filter::mppca::MpPcaDenoiser;
use std::hint::black_box;
use std::time::Duration;

const SHAPE: [usize; 3] = [16, 16, 10];
const VOLUMES: usize = 60;
const RANK: usize = 6;

/// An index as `f32`, exactly: benchmark indices stay below 2¹⁶.
fn exact(index: usize) -> f32 {
    f32::from(u16::try_from(index).expect("invariant: benchmark indices are below 2^16"))
}

/// Deterministic rank-6 series plus a xorshift noise field: smooth spatial
/// loadings times oscillating volume profiles, offset to a positive mean.
fn series() -> Vec<Vec<f32>> {
    let voxels: usize = SHAPE.iter().product();
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Top 16 bits → uniform [−0.5, 0.5): unit-scale noise, no RNG crate.
        f32::from(
            u16::try_from(state >> 48)
                .expect("invariant: a 64-bit state shifted by 48 fits 16 bits"),
        ) / 65_536.0
            - 0.5
    };
    (0..VOLUMES)
        .map(|d| {
            (0..voxels)
                .map(|v| {
                    let [z, y, x] = [
                        v / (SHAPE[1] * SHAPE[2]),
                        (v / SHAPE[2]) % SHAPE[1],
                        v % SHAPE[2],
                    ]
                    .map(exact);
                    let signal: f32 = (0..RANK)
                        .map(|k| {
                            let (k, d) = (exact(k), exact(d));
                            let loading = (0.1 * (k + 1.0) * (z + y + x) + k).sin();
                            (40.0 / (1.0 + k)) * loading * (0.1 * (k + 1.0) * d).cos()
                        })
                        .sum();
                    100.0 + signal + noise()
                })
                .collect()
        })
        .collect()
}

fn bench_mppca(c: &mut Criterion) {
    let volumes = series();
    let views: Vec<&[f32]> = volumes.iter().map(Vec::as_slice).collect();
    let denoiser = MpPcaDenoiser::default();
    let mut group = c.benchmark_group("mppca");
    group
        .sampling_mode(SamplingMode::Flat)
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));
    group.bench_function("denoise/16x16x10x60/f32", |b| {
        b.iter(|| {
            denoiser
                .denoise(black_box(SHAPE), black_box(&views))
                .expect("invariant: the synthetic series fits the derived window")
        });
    });
    group.finish();
}

criterion_group!(benches, bench_mppca);
criterion_main!(benches);
