//! Benchmarks comparing BSpline displacement evaluation performance.
//!
//! Measures the speedup from the pre-computed basis cache + interior fast
//! path (Sprint 308).
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ritk_registration::bspline_ffd::basis::{
    evaluate_bspline_displacement_fast, init_control_grid, BasisCache,
};

fn bench_displacement_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bspline_displacement");

    // Small volume: 64x64x64, 4x4x4 control spacing → ~4K control points
    let dims_small = [64usize, 64, 64];
    let ctrl_spacing_small = [8.0_f64, 8.0, 8.0];
    let ctrl_dims_small = init_control_grid(dims_small, &ctrl_spacing_small);
    let cn_small = ctrl_dims_small[0] * ctrl_dims_small[1] * ctrl_dims_small[2];
    let cp_z_small = vec![0.5_f32; cn_small];
    let cp_y_small = vec![-0.3_f32; cn_small];
    let cp_x_small = vec![0.1_f32; cn_small];
    let cache_small = BasisCache::new(dims_small, &ctrl_spacing_small);

    group.bench_function("small_64cubed_8spacing", |b| {
        b.iter(|| {
            evaluate_bspline_displacement_fast(
                black_box(&cp_z_small),
                black_box(&cp_y_small),
                black_box(&cp_x_small),
                black_box(&ctrl_dims_small),
                black_box(dims_small),
                black_box(&cache_small),
            )
        })
    });

    // Medium volume: 128x128x128, 8x8x8 control spacing
    let dims_med = [128usize, 128, 128];
    let ctrl_spacing_med = [16.0_f64, 16.0, 16.0];
    let ctrl_dims_med = init_control_grid(dims_med, &ctrl_spacing_med);
    let cn_med = ctrl_dims_med[0] * ctrl_dims_med[1] * ctrl_dims_med[2];
    let cp_z_med = vec![0.5_f32; cn_med];
    let cp_y_med = vec![-0.3_f32; cn_med];
    let cp_x_med = vec![0.1_f32; cn_med];
    let cache_med = BasisCache::new(dims_med, &ctrl_spacing_med);

    group.bench_function("medium_128cubed_16spacing", |b| {
        b.iter(|| {
            evaluate_bspline_displacement_fast(
                black_box(&cp_z_med),
                black_box(&cp_y_med),
                black_box(&cp_x_med),
                black_box(&ctrl_dims_med),
                black_box(dims_med),
                black_box(&cache_med),
            )
        })
    });

    // Large volume: 256x256x256, 32x32x32 control spacing
    let dims_large = [256usize, 256, 256];
    let ctrl_spacing_large = [32.0_f64, 32.0, 32.0];
    let ctrl_dims_large = init_control_grid(dims_large, &ctrl_spacing_large);
    let cn_large = ctrl_dims_large[0] * ctrl_dims_large[1] * ctrl_dims_large[2];
    let cp_z_large = vec![0.5_f32; cn_large];
    let cp_y_large = vec![-0.3_f32; cn_large];
    let cp_x_large = vec![0.1_f32; cn_large];
    let cache_large = BasisCache::new(dims_large, &ctrl_spacing_large);

    group.bench_function("large_256cubed_32spacing", |b| {
        b.iter(|| {
            evaluate_bspline_displacement_fast(
                black_box(&cp_z_large),
                black_box(&cp_y_large),
                black_box(&cp_x_large),
                black_box(&ctrl_dims_large),
                black_box(dims_large),
                black_box(&cache_large),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_displacement_evaluation);
criterion_main!(benches);
