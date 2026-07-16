//! Criterion benchmarks for the direct (sparse-loop) Parzen histogram computation.
//!
//! Compares the tensor-based dense path vs. the direct NdArray-optimized path
//! to quantify the speedup from avoiding full `[N, num_bins]` weight matrices.
//!
//! Sprint 328: Added field-compaction benchmarks measuring the memory and
//! performance impact of `StackWeights.len` `usize → u8` (MEM-325-01) and
//! `SparseWFixedEntry.bin` `usize → u16` (PERF-326-02).

use coeus_core::SequentialBackend;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ritk_core::image::Image;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_interpolation::{Interpolator, LinearInterpolator};
use ritk_registration::metric::histogram::{
    build_sparse_w_fixed_transposed, compaction_sizes, compute_joint_histogram_direct,
    compute_joint_histogram_from_cache_sparse, ParzenJointHistogram, SparseWFixedEntry,
    SparseWFixedT,
};
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::{Transform, TranslationTransform};

type B = SequentialBackend;

fn create_test_image(device: &<B as ritk_image::tensor::Backend>::Device) -> Image<f32, B, 3> {
    let n = 32 * 32 * 32;
    let data: Vec<f32> = (0..n).map(|i| i as f32 % 256.0).collect();
    let tensor = Tensor::<B, 3>::from_data((data, ([32, 32, 32])), device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn bench_parzen_direct(c: &mut Criterion) {
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let fixed = create_test_image(&device);
    let moving = create_test_image(&device);
    let transform = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &device));
    let interpolator = LinearInterpolator::new_zero_pad();
    let histogram = ParzenJointHistogram::<B>::new(32, 0.0, 255.0, 255.0 / 32.0, &device);

    let mut group = c.benchmark_group("ParzenHistogram");

    // 1. Tensor-based joint histogram (current production path — end-to-end)
    group.bench_function("tensor_joint_histogram_32cubed", |b| {
        b.iter(|| {
            black_box(histogram.compute_image_joint_histogram(
                black_box(&fixed),
                black_box(&moving),
                black_box(&transform),
                black_box(&interpolator),
                ritk_registration::metric::SamplingConfig::full_grid(),
            ))
        })
    });

    // Prepare normalized values for the direct benchmark
    let fixed_flat = fixed.data().clone().reshape([32 * 32 * 32]);
    let moving_points = transform.transform_points(fixed.index_to_world_tensor(
        ritk_core::image::grid::generate_grid(fixed.shape(), &device),
    ));
    let moving_indices = moving.world_to_index_tensor(moving_points);
    let moving_values = interpolator.interpolate(moving.data(), moving_indices);

    let num_bins = 32;
    let num_bins_f = (num_bins - 1) as f32;
    let fix_scale = num_bins_f / 255.0;
    let mov_scale = num_bins_f / 255.0;
    let fixed_norm = (fixed_flat.clone() * fix_scale).clamp(0.0, num_bins_f);
    let moving_norm = (moving_values.clone() * mov_scale).clamp(0.0, num_bins_f);
    let fixed_norm_data = fixed_norm.into_data();
    let fixed_norm_slice = fixed_norm_data.as_slice::<f32>().unwrap().to_vec();
    let moving_norm_data = moving_norm.into_data();
    let moving_norm_slice = moving_norm_data.as_slice::<f32>().unwrap().to_vec();
    let sigma_sq = 1.0_f32; // sigma_in_bins = 1 for Mattes MI

    // 2. Direct computation (no weight matrices, sample-by-sample accumulation)
    group.bench_function("direct_joint_histogram_32cubed", |b| {
        b.iter(|| {
            let result = compute_joint_histogram_direct(
                black_box(&fixed_norm_slice),
                black_box(&moving_norm_slice),
                num_bins,
                sigma_sq,
                sigma_sq,
                None,
                None,
            );
            black_box(result)
        })
    });

    // 3. Tensor-based joint histogram (values-only path, no image/spatial ops)
    group.bench_function("tensor_values_only_32cubed", |b| {
        let fixed_vals = fixed.data().clone().reshape([32 * 32 * 32]);
        let mov_vals = moving_values.clone();
        b.iter(|| {
            black_box(histogram.compute_joint_histogram(
                black_box(&fixed_vals),
                black_box(&mov_vals),
                None,
            ))
        })
    });

    // 4. Sparse W_fixed^T cache path — build the sparse cache ONCE, then
    // benchmark only the per-iteration histogram computation.
    // This models the CMA-ES hot loop where the fixed image doesn't change.
    let sparse_cache: SparseWFixedT =
        build_sparse_w_fixed_transposed(&fixed_norm_slice, num_bins, sigma_sq, None);
    group.bench_function("direct_sparse_cache_joint_histogram_32cubed", |b| {
        b.iter(|| {
            let result = compute_joint_histogram_from_cache_sparse(
                black_box(&sparse_cache),
                black_box(&moving_norm_slice),
                num_bins,
                sigma_sq,
                None,
                None,
            );
            black_box(result)
        })
    });

    // 5. Production dispatch path — `compute_joint_histogram_dispatch` is
    // pub(crate) so we call `compute_joint_histogram` (the public API that
    // dispatch delegates to). When the `direct-parzen` feature is enabled,
    // dispatch extracts normalized data and calls `compute_joint_histogram_direct`
    // instead; this benchmark measures the tensor-based path for comparison.
    let fixed_vals_1d = fixed.data().clone().reshape([32 * 32 * 32]);
    let moving_vals_1d = moving_values.clone();
    group.bench_function("dispatch_joint_histogram_32cubed", |b| {
        b.iter(|| {
            black_box(histogram.compute_joint_histogram(
                black_box(&fixed_vals_1d),
                black_box(&moving_vals_1d),
                None,
            ))
        })
    });

    // 6. Sparse cache build time — measures the one-time cost of constructing
    // the sparse W_fixed^T from normalized fixed values. In production, this
    // is now lazy (built on first CMA-ES iteration via `fixed_norm`), so this
    // benchmark quantifies the overhead of that first-iteration lazy build.
    group.bench_function("build_sparse_cache_32cubed", |b| {
        b.iter(|| {
            let result = build_sparse_w_fixed_transposed(
                black_box(&fixed_norm_slice),
                num_bins,
                sigma_sq,
                None,
            );
            black_box(result)
        })
    });

    group.finish();
}

// ── Field-compaction memory & stress benchmarks (Sprint 328) ───────────────

/// Benchmark the memory footprint of direct-Parzen types after compaction.
///
/// Documents the `size_of` for key types to track size regressions:
/// - `StackWeights`: `usize → u8` saved ~8 bytes (MEM-325-01)
/// - `BinRange`: `usize → u16` saved 12 bytes (MEM-324-04)
/// - `SampleWindow`: cumulative compaction saved ~28 bytes production
/// - `SparseWFixedEntry`: `usize → u16` saved 8 bytes (PERF-326-02)
///
/// Also measures sparse cache total heap allocation for a 32³ volume
/// and build throughput at broad sigma to stress the compacted types.
fn bench_compaction_memory(c: &mut Criterion) {
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let fixed = create_test_image(&device);

    let fixed_flat = fixed.data().clone().reshape([32 * 32 * 32]);
    let num_bins = 32;
    let num_bins_f = (num_bins - 1) as f32;
    let fix_scale = num_bins_f / 255.0;
    let fixed_norm = (fixed_flat * fix_scale).clamp(0.0, num_bins_f);
    let fixed_norm_data = fixed_norm.into_data();
    let fixed_norm_slice = fixed_norm_data.as_slice::<f32>().unwrap().to_vec();
    let sigma_sq = 1.0_f32;

    let mut group = c.benchmark_group("ParzenCompactionMemory");

    // 1. Static sizeof measurements (runtime cost is zero; documents sizes)
    group.bench_function("sizeof_types", |b| {
        b.iter(|| {
            let sizes = compaction_sizes();
            black_box(sizes)
        })
    });

    // 2. Sparse cache total allocation (heap memory for the sparse W_fixed^T)
    //    With u16 compaction: ~1.75 KB for 32K samples × 7 entries × 8 bytes
    //    Without compaction:  ~3.5 KB for 32K samples × 7 entries × 16 bytes
    group.bench_function("sparse_cache_heap_bytes_32cubed", |b| {
        b.iter(|| {
            let cache = build_sparse_w_fixed_transposed(
                black_box(&fixed_norm_slice),
                num_bins,
                sigma_sq,
                None,
            );
            let entry_count: usize = cache.iter().map(|v| v.0.len()).sum();
            let heap_bytes = entry_count * std::mem::size_of::<SparseWFixedEntry>()
                + cache.len() * std::mem::size_of::<(Vec<SparseWFixedEntry>, f32)>();
            black_box((cache, heap_bytes))
        })
    });

    // 3. Sparse cache build throughput with broad sigma (σ²=4.0, ~13 entries/sample).
    //    More entries per sample means more `SparseWFixedEntry::new()` calls,
    //    stressing the u16 construction path and memory allocation.
    let broad_sigma_sq = 4.0_f32;
    group.bench_function("build_sparse_cache_sigma4_32cubed", |b| {
        b.iter(|| {
            let result = build_sparse_w_fixed_transposed(
                black_box(&fixed_norm_slice),
                num_bins,
                broad_sigma_sq,
                None,
            );
            black_box(result)
        })
    });

    group.finish();
}

// ── Broad-sigma & large-volume stress benchmarks (Sprint 328) ────────────

/// Stress-test the direct and sparse Parzen paths at broad sigma (σ²=4.0)
/// and larger volumes (64³).
///
/// These benchmarks exercise the compacted types under conditions that
/// maximize their benefits:
/// - Broad sigma → more `StackWeights` entries (13 vs 7), stressing `u8` len
/// - Broad sigma → more `SparseWFixedEntry` values per sample (~13 vs ~7),
///   stressing `u16` bin and cache locality
/// - Large volume (64³ = 262K samples) → ~3.4M sparse entries, where `u16`
///   compaction saves ~27 MB of heap (3.4M × 8 bytes)
///
/// These are stress/regression benchmarks, not A/B comparisons — the
/// compactions are compile-time changes with no pre-compaction baseline.
fn bench_parzen_broad_sigma(c: &mut Criterion) {
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let fixed = create_test_image(&device);
    let moving = create_test_image(&device);
    let transform = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &device));
    let interpolator = LinearInterpolator::new_zero_pad();

    let fixed_flat = fixed.data().clone().reshape([32 * 32 * 32]);
    let moving_points = transform.transform_points(fixed.index_to_world_tensor(
        ritk_core::image::grid::generate_grid(fixed.shape(), &device),
    ));
    let moving_indices = moving.world_to_index_tensor(moving_points);
    let moving_values = interpolator.interpolate(moving.data(), moving_indices);

    let num_bins = 32;
    let num_bins_f = (num_bins - 1) as f32;
    let fix_scale = num_bins_f / 255.0;
    let mov_scale = num_bins_f / 255.0;
    let fixed_norm = (fixed_flat * fix_scale).clamp(0.0, num_bins_f);
    let moving_norm = (moving_values * mov_scale).clamp(0.0, num_bins_f);
    let fixed_norm_data = fixed_norm.into_data();
    let fixed_norm_slice = fixed_norm_data.as_slice::<f32>().unwrap().to_vec();
    let moving_norm_data = moving_norm.into_data();
    let moving_norm_slice = moving_norm_data.as_slice::<f32>().unwrap().to_vec();

    let mut group = c.benchmark_group("ParzenBroadSigma");

    // 1. Direct path with broad sigma (σ²=4.0, half_width=6, 13 bins)
    //    Larger StackWeights (u8 len=13) exercises the compacted struct.
    let broad_sigma_sq = 4.0_f32;
    group.bench_function("direct_sigma4_32cubed", |b| {
        b.iter(|| {
            let result = compute_joint_histogram_direct(
                black_box(&fixed_norm_slice),
                black_box(&moving_norm_slice),
                num_bins,
                broad_sigma_sq,
                broad_sigma_sq,
                None,
                None,
            );
            black_box(result)
        })
    });

    // 2. Sparse path with broad sigma (σ²=4.0) — more entries per sample
    //    means more SparseWFixedEntry values, stressing the u16 compaction.
    let broad_cache: SparseWFixedT =
        build_sparse_w_fixed_transposed(&fixed_norm_slice, num_bins, broad_sigma_sq, None);
    group.bench_function("sparse_cache_sigma4_32cubed", |b| {
        b.iter(|| {
            let result = compute_joint_histogram_from_cache_sparse(
                black_box(&broad_cache),
                black_box(&moving_norm_slice),
                num_bins,
                broad_sigma_sq,
                None,
                None,
            );
            black_box(result)
        })
    });

    // 3. Larger volume (64³) — stresses memory bandwidth of compacted types.
    //    With 64³ = 262K samples, the sparse cache is ~14 KB with u16 (was ~28 KB).
    let large_n = 64 * 64 * 64;
    let large_data: Vec<f32> = (0..large_n).map(|i| i as f32 % 256.0).collect();
    let large_tensor = Tensor::<B, 3>::from_data(
        (large_data, ([64, 64, 64])),
        &device,
    );
    let large_flat = large_tensor.reshape([large_n]);
    let large_fix_scale = num_bins_f / 255.0;
    let large_fixed_norm = (large_flat * large_fix_scale).clamp(0.0, num_bins_f);
    let large_fixed_data = large_fixed_norm.into_data();
    let large_fixed_slice: Vec<f32> = large_fixed_data.as_slice::<f32>().unwrap().to_vec();

    // Build the 64³ cache once (one-time cost), then benchmark the hot loop.
    let large_cache: SparseWFixedT =
        build_sparse_w_fixed_transposed(&large_fixed_slice, num_bins, broad_sigma_sq, None);
    let large_mov_data: Vec<f32> = (0..large_n).map(|i| (i * 3 + 7) as f32 % 256.0).collect();
    let large_moving_tensor = Tensor::<B, 1>::from_data(
        (large_mov_data, ([large_n])),
        &device,
    );
    let large_mov_scale = num_bins_f / 255.0;
    let large_moving_norm = (large_moving_tensor * large_mov_scale).clamp(0.0, num_bins_f);
    let large_moving_data = large_moving_norm.into_data();
    let large_moving_slice: Vec<f32> = large_moving_data.as_slice::<f32>().unwrap().to_vec();

    group.bench_function("sparse_cache_large64cubed", |b| {
        b.iter(|| {
            let result = compute_joint_histogram_from_cache_sparse(
                black_box(&large_cache),
                black_box(&large_moving_slice),
                num_bins,
                broad_sigma_sq,
                None,
                None,
            );
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_parzen_direct,
    bench_compaction_memory,
    bench_parzen_broad_sigma
);
criterion_main!(benches);
