# RITK Performance Optimization Guide

This document tracks performance characteristics, known bottlenecks, and optimization opportunities across the RITK codebase.

## Current State (v0.50.60)

### Test Suite Performance
| Package | Tests | Time (approx) | Status |
|--------|-------|--------------|--------|
| ritk-core | 1395 | ~8s | ✅ All passing |
| ritk-registration | 300 | ~15s | ✅ All passing |
| ritk-io | ~308 | ~30s | ✅ All passing |

### Known Optimizations Already Implemented

1. **CMA-ES Optimizer (`ritk-registration/src/optimizer/cma_es/`)**
   - ✅ Zero inner-loop allocation design
   - ✅ Statically mapped arrays for populations and covariance matrices
   - ✅ Flat `Vec<f64>` storage for spatial locality
   - Autodiff stripping in CMA-ES loop (Sprint 290) - 2-5x speedup

2. **Histogram Computation (`ritk-registration/src/metric/histogram.rs`)**
   - ✅ Fixed-image cache: `HistogramCache` stores `w_fixed_transposed` across iterations
   - ✅ Chunking for large datasets (CHUNK_SIZE = 32768)
   - ✅ Vectorized weight computation using broadcast operations

3. **Multi-Resolution Pyramid**
   - ✅ Coarse-to-fine registration reduces computation at full resolution
   - ✅ Configurable shrink factors per axis

---

## Identified Bottlenecks

### ✅ CLOSED — Sprint 293 (Completed 2026-05-23)

#### 1. B-Spline Interpolation Memory Allocations — FIXED
**Location:** `crates/ritk-core/src/interpolation/bspline.rs`

**Issue:**
```rust
// Old implementation — 64 full volume clones per point in 3D
let sample = data.clone().slice([
    xi as usize..xi as usize + 1,
    yi as usize..yi as usize + 1,
    zi as usize..zi as usize + 1,
]);
let sample_scalar = sample.reshape([1]);
result = result.add(sample_scalar.mul_scalar(weight));
```
- For N points on a volume of size V: O(64 × N × V) memory allocations
- For 1000 points on 64³ volume: ~64,000 volume clones, ~16.8 billion elements

**Fix (Sprint 293):**
```rust
// Pre-extract data once outside the loop
let volume_data = data.clone().to_data();
let volume_slice: &[f32] = volume_data.as_slice::<f32>().expect("...");

// Use direct stride indexing per sample — zero tensor allocations:
let idx = base0 + yi as usize * stride1 + zi as usize;
result += unsafe { *volume_slice.get_unchecked(idx) } * weight;
```

**Results:**
- Memory allocations: O(64×N×V) → O(1) per interpolation
- Performance: 1000-point 64³ BSpline in 0.051s (debug build)
- All 1398 ritk-core tests pass, all 306 ritk-registration tests pass
- Additional optimizations: `cubic_bspline` uses multiplication instead of `powi`, `#[inline]` attributes added
```

**Measured Improvement (debug, NdArray):** 33 s → **0.039 s** for 1000 pts on 64³ (**~850×**)

#### 2. Sequential Point Processing in Interpolators
**Location:** All interpolators (`bspline.rs`, `linear/`, `nearest.rs`)

**Issue:**
- The `interpolate` method loops over points sequentially
- Each point is processed independently
- No SIMD vectorization across points

**Impact:**
- Poor CPU cache utilization
- Missed opportunity for batch processing

**Status:** Partially addressed in Sprint 293 for BSpline (restructured loops with pre-computed base indices)

**Solution:**
- Refactor to process all points in batch using tensor operations
- Enable SIMD auto-vectorization

**Estimated Improvement:** 4-8x speedup from SIMD, better cache locality

**Note:** Sprint 293 implemented loop restructuring for BSpline with pre-computed strides, reducing nested conditionals. Full batch tensor processing remains as future work.

---

### 🟡 MEDIUM PRIORITY

#### 3. Tensor Data Movement Between CPU/GPU
**Location:** Various - any code using `to_data()` or `.into_data()`

**Issue:**
- Burn tensors may be on GPU (via wgpu backend)
- Frequent `.to_data()` calls force CPU-GPU sync
- B-spline interpolation does this in hot loop

**Impact:**
- GPU backend: Significant performance penalty
- CPU backend: Minimal impact

**Solution:**
- Keep data on device when possible
- Use device-native operations
- Batch sync operations

#### 4. Histogram Chunking Overhead
**Location:** `histogram.rs`

**Issue:**
- Chunking (CHUNK_SIZE = 32768) adds loop overhead
- Each chunk creates new intermediate tensors

**Impact:**
- Slight overhead for small datasets that fit in one chunk
- Necessary for large datasets to avoid dispatch limits

**Solution:**
- Auto-tune chunk size based on dataset size
- Consider fused operations to reduce intermediate allocations

---

## Optimization Roadmap

### Sprint 294 (was 293): Interpolator Optimization ✅ COMPLETE
**Goal:** 10x reduction in B-spline interpolation memory allocations

| Task | Priority | Complexity | Result |
|------|----------|------------|--------|
| Refactor B-spline to use flat data + direct indexing | High | Medium | ✅ **~850× speedup** |
| Refactor B-spline to batch-process points | High | High | ⏳ Deferred to Sprint 295 |
| Add `#[ignore]` timing regression test | Medium | Low | ✅ Added |
| Add Criterion benchmarks | Medium | Low | ⏳ Deferred |

**Actual speedup:** 33 s → 0.039 s for 1000 pts on 64³ (debug, NdArray backend)

### Sprint 295: Registration Pipeline Optimization ✅ PARTIAL
**Goal:** Reduce registration time by 30%

| Task | Priority | Complexity | Status | Estimated Impact |
|------|----------|------------|--------|------------------|
| Cache W_fixed^T in chunked histogram path | High | Medium | ✅ Done | 2× per MI eval for N > 32768 |
| Fuse transform + interpolation | Medium | High | ⏳ Deferred | Reduce intermediate tensors |
| Optimize MI metric inner loop | Medium | Medium | ⏳ Deferred | 10-20% faster |
| Parallelize multi-start | Low | High | ⏳ Deferred | Linear speedup with cores |

**Achieved:** W_fixed^T caching eliminates O(N×bins) recomputation per CMA-ES iteration in the chunked path. DRY extraction via `compute_w_fixed_transposed`.

### Sprint 300: MI Inner-Loop Optimization ✅ COMPLETE

**Goal:** 10–20% MI evaluation speedup via kernel dispatch elimination and arithmetic simplification

| Task | Priority | Complexity | Status | Estimated Impact |
|------|----------|------------|--------|------------------|
| `powf_scalar(2.0)` → `diff * diff` | High | Low | ✅ Done | 8–12% MI speedup |
| Pre-computed `bins_exp` tensor | High | Low | ✅ Done | 3–5% MI speedup |
| `encode_us` stack allocation + `#[inline]` | Medium | Low | ✅ Done | Minor DIMSE speedup |
| Fuse transform + interpolation | Medium | High | ⏳ Deferred | Reduce intermediate tensors |
| Eliminate `w_fixed_t.clone().slice()` per chunk | Medium | Medium | ⏳ Deferred | 5–10% for large volumes |
| Parallelize multi-start | Low | High | ⏳ Deferred | Linear speedup with cores |

**Achieved:** Combined 11–17% MI evaluation speedup from `powf_scalar` elimination + `bins_exp` caching. Both are numerically safe (identical floating-point results for finite inputs).

**Bug fix:** Restored sampling-path `if use_sampling` branch in `compute_image_joint_histogram` (broken during Sprint 295 partition).

### Sprint 303: Parzen Weight Matrix Optimization ✅ COMPLETE

**Goal:** 3-5x MI evaluation speedup by eliminating the dominant Parzen weight matrix bottleneck

**Context (Sprint 302 retrospective):** Benchmark profiling revealed that Parzen weight matrix construction (`W_fixed`, `W_moving`) accounts for ~86% of total MI forward time, with transform+interpolate at only ~14%. The dense `[N, num_bins]` weight matrix requires O(N x num_bins) exp() calls and multiple intermediate tensor allocations.

| Task | Priority | Complexity | Status | Result |
|------|----------|------------|--------|--------|
| Direct NdArray joint histogram (no weight matrices) | High | Medium | Done | ~6x speedup (22ms to 4ms) |
| Sparse scatter-based Parzen kernel | High | High | Done (archived) | Autodiff-incompatible, kept for future custom kernel |
| Sparse W_fixed^T cache (Vec<Vec<(usize, f32)>>) | High | Medium | Done | ~2.5-3x over dense cache path |
| Eliminate w_fixed_t.clone().slice() per chunk | Medium | Medium | Done | 4MB→56 bytes per sample for chunk slicing |
| `compute_parzen_weights` extraction | Medium | Low | Done | Code quality: named method replaces closure |
| `direct-parzen` feature flag | Medium | Low | Done | Default-on, opt-out for GPU/autodiff |
| Dispatch integration (compute_image.rs) | High | Low | Done | Sparse→Dense→Full dispatch chain |
| Parzen-direct benchmark suite | Medium | Low | Done | 5 benches comparing tensor vs direct vs sparse |
| Direct computation + OOB mask tests | Medium | Low | Done | 3 integration tests |
| Dispatch integration tests | Medium | Low | Done | 3 tests: dispatch matches tensor, OOB, sparse cache |

**Benchmark Results (release build, NdArray backend, 32-cubed volume):**

| Method | Time | Speedup vs Tensor | Sprint |
|--------|------|-------------------|--------|
| Tensor-based joint histogram | ~22 ms | 1.0x | 295 |
| Direct sparse-loop histogram | ~4 ms | ~5.5x | 303 |
| Direct sparse-loop (OPT-1..5) | 1.40 ms | 7.2× | 311 |
| Sparse W_fixed^T cache (CMA-ES hot loop) | 1.00 ms | 10.1× | 303 |
| Sparse cache + OPT-1..5 | 1.00 ms | 10.1× | 311 |
| Tensor-based (values only) | ~21 ms | 1.0x | 295 |

**Architecture:** Three-tier dispatch chain in `compute_image.rs`:
1. **Sparse cache hit** → `compute_joint_histogram_from_cache_sparse_dispatch` (CMA-ES, ~7 non-zero fixed bins per sample, contiguous memory)
2. **Dense cache hit** → `compute_joint_histogram_from_cache_dispatch` (autodiff-safe, tensor matmul preserves gradient tape for RSGD)
3. **Cache miss** → `compute_joint_histogram_dispatch` (full computation from both fixed and moving values)

**Key findings:**
- The direct computation path avoids all `[N, num_bins]` intermediate allocations
- Only computes exp() for bins within +/-3-sigma (~7 bins vs 32 for Mattes MI)
- Accumulates directly into `[num_bins, num_bins]` histogram
- The sparse W_fixed^T cache eliminates strided memory access (128 KB stride) and 78% of zero-weight iterations
- Limitation: Direct/sparse paths are only for NdArray backend without autodiff
- Burn's autodiff scatter backward doesn't support expand-scatter pattern

**Files added/modified:**

| File | Lines | Status |
|------|-------|--------|
| `parzen/direct.rs` | 520 | New + sparse cache functions |
| `parzen/sparse.rs` | 406 | New - Archived sparse scatter kernel (unused, kept for future) |
| `parzen/dispatch.rs` | 200 | Modified - Added sparse dispatch method |
| `parzen/compute_image.rs` | 400 | Modified - Sparse cache integration, helper refactors |
| `parzen/compute.rs` | 236 | Modified - Extracted `compute_parzen_weights` method |
| `histogram/cache.rs` | 30 | Modified - Added `sparse_w_fixed` field |
| `histogram/mod.rs` | 12 | Modified - Added sparse exports |
| `benches/parzen_direct.rs` | 160 | Modified - Added sparse and dispatch benches |
| `Cargo.toml` | - | Added `direct-parzen` feature flag |

### Sprint 311: Parzen Direct Inner-Loop Optimization + Lazy Sparse Cache ✅ COMPLETE

**Goal:** Optimize the hot accumulation loops in the direct Parzen histogram path and reduce peak memory during cache construction.

| Task | Priority | Complexity | Status | Result |
|------|----------|------------|--------|--------|
| OPT-1: Row base pointers for histogram row access | High | Low | ✅ Done | Replaces multiply with pointer add in inner loop |
| OPT-2: Hoist moving exp() out of fixed-weight loop | High | Low | ✅ Done | 49 → 14 exp() calls per sample (7×7 window) |
| OPT-3: Unchecked histogram access | High | Low | ✅ Done | Removes bounds check from hottest path |
| OPT-4: StackWeights for sparse cache path | High | Low | ✅ Done | Same OPT-2 hoisting + stack allocation for CMA-ES hot loop |
| OPT-5: Stack-allocated `[f32; 7]` moving weights | Medium | Low | ✅ Done | Zero heap allocation per sample |
| MEM-311-01: Lazy sparse cache construction | High | Medium | ✅ Done | Peak memory ~6.5 MB → ~4.1 MB |
| Cache-matching deduplication | Low | Low | ✅ Done | `cache_matches_image()` shared helper |
| Masked-path caching TODO | Low | Low | ✅ Done | Documented future optimization |

**Estimated combined inner-loop speedup**: ~15–25% over the pre-Sprint-311 direct path (OPT-1: pointer add vs multiply, OPT-2: 3.5× fewer exp() calls, OPT-3: branch elimination, OPT-4/5: same for sparse path). The sparse cache path (CMA-ES hot loop) was already ~15× faster than the original tensor path; these optimizations further widen the gap.

**Memory improvement**: Lazy sparse cache defers the ~2 MB sparse allocation from cache-construction time to first CMA-ES iteration. Peak memory during the initial cache-miss call drops from ~6.5 MB (dense + sparse) to ~4.1 MB (dense + 128 KB `fixed_norm` Vec).

### Sprint 312: Parzen Benchmark Verification + Cache Dispatch Hardening + Memory + Architecture ✅ COMPLETE

**Goal:** Verify actual benchmark numbers (Sprint 311 had estimates), parallelize sparse cache build, implement masked-path caching, and fix remaining compiler warnings.

Key deliverables:

1. **PERF-312-01: Parallel sparse cache build with rayon** — `build_sparse_w_fixed_transposed` now uses `rayon::par_iter_mut` to compute each sample's sparse entries in parallel. On a 32³ volume, the one-time lazy build cost dropped from 3.94 ms → 2.33 ms (41% improvement). Combined with `Vec::with_capacity(7)` pre-allocation (MEM-312-01), this eliminates repeated re-allocations for the typical ~7 non-zero entries per sample.

2. **MEM-312-01: Vec pre-allocation for sparse cache entries** — `Vec::new()` → `Vec::with_capacity(7)` in `build_sparse_w_fixed_transposed`, avoiding ~7 push-induced re-allocations per sample across N=32K samples.

3. **FIX-312-01: Eliminated 5 `non_snake_case` warnings** — Replaced `if/else { None }` pattern with `.then(|| { ... }).flatten()` in `compute_image.rs`, and `None =>` match arm with `_ =>` in `compute.rs`.

4. **ARCH-312-01: Masked-path caching with caller-supplied cache key** — `compute_masked_joint_histogram` now accepts `cache_key: Option<u64>`. When `Some(key)`, fixed-image Parzen weights are cached and reused across calls with the same key. The sparse W_fixed^T cache is also lazily built for derivative-free backends (CMA-ES). This closes the TODO-311-01 gap. New `MaskedHistogramCache<B>` struct mirrors `HistogramCache<B>` but uses `cache_key: u64` instead of image spatial metadata matching.

5. **STR-312-01: `masked.rs` → `masked/mod.rs` + `masked/masked_chunked.rs`** — Extracted chunked helper methods into a submodule to stay under the 500-line structural limit.

**Verified benchmark results** (release mode, 32³ volume, NdArray backend):

| Path | Time | Speedup vs tensor |
|------|------|-------------------|
| `tensor_joint_histogram_32cubed` (end-to-end) | 10.14 ms | 1.0× |
| `direct_joint_histogram_32cubed` | 1.40 ms | **7.2×** |
| `direct_sparse_cache_joint_histogram_32cubed` | 1.00 ms | **10.1×** |
| `build_sparse_cache_32cubed` (one-time) | 2.33 ms | — |

**Files changed:** `direct/mod.rs`, `compute_image.rs`, `compute.rs`, `cache.rs`, `masked/mod.rs`, `masked/masked_chunked.rs`, `parzen/mod.rs`, `parzen/dispatch.rs`, `mutual_information.rs`, `tests.rs`, `CHANGELOG.md`

### Sprint 295: Memory Efficiency
**Goal:** Reduce peak memory usage by 40%

| Task | Priority | Complexity | Estimated Impact |
|------|----------|------------|------------------|
| Zero-copy image access where possible | High | High | Significant |
| Reuse tensor buffers across iterations | Medium | Medium | Moderate |
| Streaming processing for large volumes | Medium | High | Significant for large data |
| Profile and tune chunk sizes | Low | Low | Small |

---

## Architecture Decisions

### Why Parzen Uses Sparse W_fixed^T Cache (Vec<Vec<(usize, f32)>>)

The Parzen joint histogram computation builds a fixed-image weight matrix
`W_fixed^T [num_bins, N]` that is constant across all registration iterations.
Originally this was stored as a dense tensor, but ~78% of entries are zero
(only ~7 bins within ±3σ are non-zero per sample for 32 bins with σ ≈ 1 bin-width).

The sparse representation stores only the ~7 non-zero `(bin_index, weight)` pairs
per sample. This provides two independent performance wins:

1. **Eliminates 78% of inner-loop iterations** and the `if w_f > 0.0` branch
   (which was poorly predicted at 22% taken rate)
2. **Converts strided memory access to contiguous**: The dense cache accessed
   `w_fixed_transposed[a * n + i]` with stride = N (up to 128 KB), causing
   L1/L2 cache misses on every read. The sparse entries are packed into
   ~56 bytes per sample (1 L1 cache line).

Chose `Vec<Vec<(usize, f32)>>` over flat `Vec<(usize, f32)>` + offsets because:
- Slicing for the chunked path is trivial: `sparse[start..end].to_vec()`
- No two-pass counting required during construction
- Each inner Vec is tiny (~56 bytes), so the 24-byte Vec header overhead is negligible

The sparse cache is only used for derivative-free backends (CMA-ES with
`B::InnerBackend`). The dense tensor cache is retained for the autodiff path
where `into_data()` would sever the gradient tape.

### Why the Sparse Cache is Built Lazily (Sprint 311)

The sparse W_fixed^T cache was originally built eagerly on the first cache-miss call alongside the dense `w_fixed_transposed` tensor. This meant peak memory during cache construction held both representations simultaneously (~6.5 MB for a 32³ volume). However, RSGD (autodiff path) never uses the sparse cache — it always takes the tensor matmul path via `compute_joint_histogram_from_cache_dispatch`.

The lazy construction stores only the normalized `fixed_norm` Vec (~128 KB for N=32K) during the initial cache-miss. The sparse cache (~2 MB) is built on-demand the first time `get_cached_sparse_w_fixed` is called with the sparse dispatch path (CMA-ES first iteration). After construction, `fixed_norm` is consumed (set to `None`) to free the ~128 KB.

This reduces peak memory from ~6.5 MB to ~4.1 MB during the critical cache-construction phase, with no performance cost — the sparse cache is built once and then reused for all subsequent CMA-ES iterations.

### Why Direct Parzen Uses Row Base Pointers + Stack Weights (Sprint 311)

The direct Parzen histogram inner loops accumulate into a `[num_bins, num_bins]` histogram in row-major order. Five micro-optimizations were applied:

1. **Row base pointers (OPT-1)**: Pre-compute `Vec<*mut f32>` where `ptrs[a] = &mut histogram[a * num_bins]`. The inner loop uses `ptr.add(b)` instead of `histogram[a * num_bins + b]`, replacing a multiply with an add.

2. **Hoisted moving exp() (OPT-2)**: In `compute_joint_histogram_direct`, `w_m` depends only on the moving bin `b`, not on the fixed bin `a`. Pre-computing all moving weights before the fixed-weight loop eliminates `(f_range - 1) * m_range` redundant exp() calls per sample. For a 7×7 window: 49 → 14 exp() calls.

3. **Unchecked access (OPT-3)**: Bin indices are clamped to `[0, num_bins-1]` before the loop, so `get_unchecked_mut` is safe and eliminates a bounds check.

4. **Sparse path hoisting (OPT-4)**: Same OPT-2 optimization applied to `compute_joint_histogram_from_cache_sparse`, the CMA-ES hot loop.

5. **Stack-allocated weights (OPT-5)**: The pre-computed moving weights span at most `2 * half_width + 1 = 7` entries. `StackWeights { weights: [f32; 7], len: usize }` avoids heap allocation entirely — the 28-byte array fits in a single cache line.

### Why the Masked Path Uses a Caller-Supplied Cache Key (ARCH-312-01)

The image-grid path caches via `(shape, origin, spacing, direction)` matching, but the masked path receives arbitrary world points, so that cache key doesn't apply. Three possible cache key strategies were identified in TODO-311-01:

1. **Hash of the world-point set** — exact but expensive to compute (O(N) hash on every call)
2. **Caller-supplied generation counter** — cheap, but requires API change (**chosen**)
3. **Pointer/identity check** if the same Tensor handle is reused — fragile, breaks if caller clones

Strategy (2) was chosen because it's the most practical: the CMA-ES optimizer already has a generation counter, and the caller can provide it as a simple integer. The overhead is O(1) per call. The only risk is that a caller providing a stale key would reuse outdated cached weights, but this is prevented by the `n` (point count) check that catches mismatched point sets.

### Why LinearInterpolator Uses Flat Data + Gather
The linear interpolator (`crates/ritk-core/src/interpolation/linear/`) uses a highly optimized pattern:

1. Pre-flatten the entire volume: `data.clone().reshape([total_voxels])`
2. Pre-compute flat indices for all 8 corner voxels
3. Use `gather` operations to sample all corners at once
4. Apply weights using vectorized operations

This pattern:
- ✅ Minimizes allocations (one flatten vs 8 slices)
- ✅ Enables batch processing of all points
- ✅ Works well with autodiff backends
- ✅ Maintains numerical stability

**Recommendation:** Apply this pattern to B-spline and other interpolators.

### Why CMA-ES Uses Flat Vec<f64>
The CMA-ES optimizer uses flat `Vec<f64>` arrays for all state:

- Population samples: flat vector
- Covariance matrix: flat vector with manual indexing
- Cholesky factor: flat vector

This design:
- ✅ Zero allocations in inner loop
- ✅ Perfect spatial locality
- ✅ Predictable memory access patterns
- ✅ Easy to reason about performance

**Recommendation:** Maintain this design. Extend to other optimizers.

---

## Profiling Data

### B-Spline Interpolation (Before Optimization)
Test: 1000 points on 64³ volume (NdArray backend)
- Original: ~33 seconds (with `clone().slice()` pattern)
- Expected after optimization: <1 second

### Registration Pipeline
| Phase | Time | % of Total |
|-------|------|------------|
| CMA-ES (coarse level) | ~10s | 67% |
| MI Metric Evaluation | ~3s | 20% |
| RSGD Refinement | ~2s | 13% |

**Note:** Times are approximate and vary by hardware.

---

## Measurement Methodology

### Benchmark Setup
```rust
use std::time::Instant;

let start = Instant::now();
// Operation to measure
let duration = start.elapsed();
```

### CI Integration
Add to GitHub Actions:
```yaml
- name: Run benchmarks
  run: cargo bench --workspace
```

### Regression Tests
All performance-critical code should have regression tests:
```rust
#[test]
fn test_performance_regression() {
    let start = Instant::now();
    // ... operation ...
    let duration = start.elapsed();
    assert!(duration.as_secs_f32() < TIMEOUT);
}
```

---

## References

- [Burn Tensor Operations](https://docs.rs/burn/latest/burn/tensor/)
- [Rust Performance Book](https://doc.rust-lang.org/1.70.0/book/performance.html)
- [CMA-ES Original Paper](https://www.researchgate.net/publication/221220513_Completely_Derandomized_Self-Adaptation_in_Evolution_Strategies)
