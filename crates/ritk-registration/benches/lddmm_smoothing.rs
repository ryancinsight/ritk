//! Criterion benchmarks comparing CPU vs GPU field smoothing in LDDMM
//! registration.
//!
//! Measures wall-clock time for a single gradient-descent iteration on a
//! 256³ synthetic volume with 10 geodesic integration steps.
//!
//! # Running
//!
//! ```bash
//! cargo bench -p ritk-registration --bench lddmm_smoothing
//! ```
//!
//! The GPU benchmark requires a WGPU-compatible device (Vulkan, Metal,
//! or DX12).  On systems without GPU support the benchmark will still
//! compile but may panic at runtime.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::SeedableRng;

use ritk_registration::lddmm::{LddmmConfig, LddmmRegistration};
use ritk_registration::GpuFieldSmoother;
use ritk_spatial::Spacing;

type GpuBackend = Wgpu<f32, i32>;

/// Build a pair of synthetic 3-D images (fixed, moving) populated with
/// uniform random values in [0, 255].  Uses a fixed RNG seed so benchmark
/// runs are directly comparable.
fn build_synthetic_pair(n: usize) -> (Vec<f32>, Vec<f32>) {
    use rand::Rng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let fixed: Vec<f32> = (0..n).map(|_| rng.random_range(0.0..255.0)).collect();
    let moving: Vec<f32> = (0..n).map(|_| rng.random_range(0.0..255.0)).collect();
    (fixed, moving)
}

fn bench_lddmm_smoothing(c: &mut Criterion) {
    let dims = [256usize, 256, 256];
    let n = dims[0] * dims[1] * dims[2];
    let spacing = [1.0_f64; 3];

    let (fixed, moving) = build_synthetic_pair(n);

    let config = LddmmConfig {
        max_iterations: 1, // single gradient-descent iteration
        num_time_steps: 10,
        kernel_sigma: ritk_filter::GaussianSigma::new_unchecked(2.0),
        learning_rate: 0.1,
        regularization_weight: 1.0,
        convergence_threshold: 1e-5,
    };

    let mut group = c.benchmark_group("LDDMM_256cubed_10steps");
    // LDDMM on 256³ is heavy — keep sample size low to avoid
    // multi-minute benchmark runs.
    group.sample_size(10);

    // ── CPU path: CpuFieldSmoother (via register()) ──────────────────────
    group.bench_function("cpu_smoother", |b| {
        let reg = LddmmRegistration::new(config.clone());
        b.iter(|| {
            black_box(
                reg.register(
                    black_box(&fixed),
                    black_box(&moving),
                    black_box(dims),
                    black_box(spacing),
                )
                .unwrap(),
            )
        })
    });

    // ── GPU path: GpuFieldSmoother<Wgpu> (via register_with) ─────────────
    group.bench_function("gpu_smoother_wgpu", |b| {
        let device = WgpuDevice::default();
        let mut smoother =
            GpuFieldSmoother::<GpuBackend>::new(dims, Spacing::new([1.0, 1.0, 1.0]), 2.0, &device);
        let reg = LddmmRegistration::new(config.clone());
        b.iter(|| {
            black_box(
                reg.register_with(
                    black_box(&fixed),
                    black_box(&moving),
                    black_box(dims),
                    black_box(spacing),
                    black_box(&mut smoother),
                )
                .unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_lddmm_smoothing);
criterion_main!(benches);
