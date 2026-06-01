# RITK Performance Optimization Guide

This document tracks performance characteristics, known bottlenecks, and
optimization opportunities across the RITK codebase.

## Current State (v0.50.91)

### Test Suite Performance

| Package | Tests | Time (approx) | Status |
|--------|-------|--------------|--------|
| ritk-core | 1395 | ~8s | ✅ All passing |
| ritk-registration | 499 | ~16s | ✅ All passing (`--features direct-parzen`) |
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
   - ✅ Exp-ratchet in `StackWeights::new` (Sprint 319) - ~3× faster per-axis weight computation
   - ✅ Lock-free `HistogramPool::checkout` (Sprint 319) - reduced contention under rayon

3. **Multi-Resolution Pyramid**
   - ✅ Coarse-to-fine registration reduces computation at full resolution
   - ✅ Configurable shrink factors per axis

---

## Sprint 320 — Parzen Direct Path Phase Seven

### DRY sigma² helpers (DRY-320-01)

`ParzenJointHistogram::fixed_sigma_cfg()` and `moving_sigma_cfg()` encapsulate
the repeated `ParzenConfig::from_intensity_sigma(self.parzen_sigma, ...)` pattern
that appeared at 8 call sites across `compute.rs`, `compute_image.rs`,
`masked/mod.rs`, and `dispatch.rs`. Each call site now uses
`self.fixed_sigma_cfg().sigma_sq` (1 line) instead of the 5-line inline pattern.

### ParzenConfig self-methods (ARCH-320-03, ARCH-320-06)

`ParzenConfig::bin_range(val, num_bins)` and `compute_weights(val, num_bins)`
encapsulate the `floor → BinRange::new → StackWeights::new` pattern that was
previously inlined at 4 call sites. `sum_weights()` provides the discrete
Gaussian weight sum for introspection and cross-validation.

### Clippy zero-warning (CLIPPY-320-03/04/05)

- `needless_range_loop`: `for slot in 0..len` → `for w in weights.iter_mut().take(len)`
- `int_plus_one`: `hi - lo + 1 <= C` → `hi - lo < C` (equivalent for usize)
- `doc_lazy_continuation`: 7 continuation lines indented with 3 spaces

---

## Sprint 319 — Parzen Direct Path Phase Six

### Exp-ratchet optimisation (PERF-319-04)

`StackWeights::new` now uses a FMA chain instead of N independent `exp()`
calls. Adjacent integer bins differ by exactly 1 in the `diff` value, so
the exponent changes by a constant increment with a constant second
difference:

```
exponent[0] = diff₀² × inv_2sigma_sq
Δ₀ = inv_2sigma_sq × (1 - 2 × diff₀)
exponent[k+1] = exponent[k] + Δ_k
Δ_{k+1} = Δ_k + 2 × inv_2sigma_sq
```

For a typical 7-bin window: 1× `exp()` + 6× FMA ≈ 3× faster than 7× `exp()`.
Drift bounded by ~15 ULP for the maximum 15-bin window, well within the
1e-4 test tolerance.

### Lock-free checkout (PERF-319-05)

`HistogramPool::checkout` drops the Mutex lock before zero-filling or
allocating. Previously, the lock was held during the entire
`fill(0.0)` / allocation, blocking other threads in the rayon fold.
New allocations skip the redundant `fill(0.0)` since `vec![0.0; N]`
already produces a zeroed buffer.

### STACK_WEIGHTS_CAPACITY increased (FIX-319-09)

From 16 (σ ≤ 4.5 bins) to 32 (σ ≤ 5.2 bins). The previous capacity
was insufficient for `sigma_sq ≥ 9.0` (σ = 3 bins → half_width = 9 →
range = 19 bins > 16). `StackWeights` is now 132 bytes (32×f32 + usize),
still `Copy`-safe and cache-friendly.

### SSOT completion (SSOT-319-01, SSOT-319-02)

All sigma² conversions across the Parzen subsystem now go through
`ParzenConfig::from_intensity_sigma`. The deprecated `sigma_sq_in_bins`
function has been removed entirely. 10+ call sites across `compute.rs`,
`compute_image.rs`, `masked/mod.rs`, `dispatch.rs`, and test files
consolidated to a single SSOT path.

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

### Sprint 313: Parzen Cache Dispatch Hardening + Parallel Hot Loop + Structural Compliance

✅ COMPLETE

**Goal:** Parallelize the CMA-ES sparse hot loop, add cache invalidation API, extract shared lazy-build logic, fix remaining clippy warnings, deprecate the dense cache path, and achieve structural compliance.

Key deliverables:

1. **PERF-313-01: Parallel sparse hot-loop histogram reduction** — `compute_joint_histogram_from_cache_sparse` now uses rayon `into_par_iter().fold().reduce()` with thread-local histograms. Each thread accumulates into its own `[num_bins × num_bins]` buffer, eliminating all synchronization from the hot loop. The final reduction sums thread-local results into the output histogram. This also removes `unsafe` pointer arithmetic from this path (safe indexing replaces OPT-1 row base pointers + unchecked writes within each thread-local buffer).

2. **FIX-313-01: Eliminated 4 remaining clippy warnings** — `needless_range_loop` on OPT-1 hot loop (suppressed: `a` used for both indexing and arithmetic), `single_range_in_vec_init` on Burn 1-D `.slice()` (suppressed: correct API), `doc_quote_line_without_gt_marker` in `sparse.rs` (escaped `\>`), `op_ref` in `lncc.rs` (removed unnecessary `&`).

3. **FIX-313-02: Deprecated `compute_joint_histogram_from_cache_direct`** — the dense-cache path is slower than the sparse path and only retained for test validation. Marked `#[deprecated(since = "0.50.75")]`.

4. **ARCH-313-01: Cache invalidation API** — `ParzenJointHistogram` now exposes `invalidate_cache()`, `invalidate_masked_cache()`, and `invalidate_all_caches()` for explicit cache clearing between registration stages, fixed-image switches, or memory reclamation.

5. **ARCH-313-02: Shared lazy-build logic** — `get_or_build_sparse_w_fixed` method on both `HistogramCache` and `MaskedHistogramCache` eliminates the duplicated lazy-build pattern previously inlined in `compute_image.rs` and `masked/mod.rs`.

6. **STR-313-01: `tests.rs` → `tests/` directory module** — the 1054-line test file was split into `tests/mod.rs` (338 lines), `tests/cache_tests.rs` (238 lines), and `tests/masked_cache_tests.rs` (457 lines), all compliant with the 500-line limit.

**Files changed:** `direct/mod.rs`, `compute.rs`, `sparse.rs`, `lncc.rs`, `cache.rs`, `parzen/mod.rs`, `compute_image.rs`, `masked/mod.rs`, `mod.rs` (histogram), `tests/` (new directory module), `CHANGELOG.md`

### Sprint 312: Parzen Benchmark Verification + Cache Dispatch Hardening + Memory + Architecture

✅ COMPLETE

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

### Why the Sparse Hot Loop Uses Parallel Reduction (PERF-313-01, Sprint 313)

`compute_joint_histogram_from_cache_sparse` is called on every CMA-ES iteration and was previously sequential. The loop iterates over N samples, each computing ~7 moving Parzen weights and accumulating into the `[num_bins, num_bins]` histogram.

The loop is not embarrassingly parallel because of the shared mutable histogram — concurrent writes would race. The standard parallel-reduction pattern solves this:

1. **Thread-local histograms**: Each rayon thread allocates its own `vec![0.0f32; num_bins * num_bins]` in the `fold` initializer. All writes within a thread are to its local buffer — no synchronization needed.
2. **Safe indexing**: With thread-local buffers, the `unsafe` pointer arithmetic (OPT-1 row base pointers + `get_unchecked_mut`) is replaced by safe `local_hist[row_base + m_lo + j]` indexing. The bounds check is negligible compared to the exp() calls.
3. **Deterministic reduction**: The `reduce` phase sums all thread-local histograms into the final result. This is a simple element-wise addition.

**Trade-off**: Parallel reduction changes the floating-point accumulation order, so results are no longer bit-identical to the sequential version. The differences are typically ~1e-5, well within the 1e-4 tolerance used in tests. This is inherent to any parallel summation over floats.

**Memory**: Each thread allocates `num_bins² × 4` bytes. For 32 bins, that's 4 KB per thread — negligible. The total temporary memory is `num_threads × 4 KB`.

The non-cached path (`compute_joint_histogram_direct`) was not parallelized because it's called only on the first CMA-ES iteration (or for RSGD) and is not on the hot loop. The deprecated dense-cache path (`compute_joint_histogram_from_cache_direct`) was also left sequential since it's deprecated.

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

### Sprint 316 (0.50.78) — Parzen Cache Dispatch Phase Four

#### MEM-316-01: `SampleWindow` precomputed bin ranges

New `SampleWindow` struct computes a sample's `(primary, lo, hi)` bin range once, replacing the repeated `floor/primary - hw/max(0)/min(num_bins-1)` calculation that was duplicated in both `compute_joint_histogram_direct` and `compute_joint_histogram_from_cache_sparse`. Returns `None` for OOB samples, eliminating the `if mask_val >= 0.5` branch from fold closures (FIX-316-07).

#### PERF-316-03: SIMD-aligned `StackWeights`

Weight array size rounded from `[f32; 7]` to `[f32; 8]` (32 bytes = one AVX2 `__m256` register). `STACK_WEIGHTS_CAPACITY = 8` constant introduced. The 8th slot is zero-filled padding that never participates in any computation, enabling the compiler to emit aligned `vmovaps` instead of `vmovups` when auto-vectorizing the inner weight loop.

#### ARCH-316-04: `BinRange` newtype

Replaces bare `(lo, hi)` pairs with a typed struct providing named fields `lo`/`hi`, plus `len()`, `is_empty()`, and `iter()` methods. Handles edge case where `primary > num_bins - 1` by collapsing the range to a single boundary bin. Zero runtime cost.

#### FIX-316-07: Branch-eliminated accumulate

The OOB mask check is now folded into `SampleWindow::new()` / `SampleWindow::new_moving_only()`, which return `Option`. Both fold closures simplified from:
```rust
let mask_val = match oob_mask { ... };
if mask_val >= 0.5 {
    let m_val = ...; let m_primary = ...; let m_lo = ...; let m_hi = ...;
    accumulate_sample(hist, num_bins, m_val, m_lo, m_hi, ...);
}
```
to:
```rust
if let Some((m_val, m_range)) = SampleWindow::new_moving_only(i, ...) {
    accumulate_sample(hist, num_bins, m_val, m_range, ...);
}
```

#### DOC-316-06: Module-level Safety and Examples

Added `# Safety` section (no `unsafe`, zero-filled padding, Mutex poison recovery) and `# Examples` section to `direct/mod.rs`.

#### TEST-316-05: 16 new tests

5 property tests: `sparse_w_fixed_deterministic`, `histogram_non_negative_all_entries`, `histogram_marginals_sum_correctly`, `bin_range_primary_exceeds_num_bins`, `bin_range_primary_negative`. Plus 11 unit tests: 6 `BinRange`, 5 `SampleWindow`, 2 SIMD-alignment.

### Sprint 318 (0.50.80) — Parzen Cache Direct Path Phase Five

#### FIX-318-01: MAX_PARZEN_BINS and STACK_WEIGHTS_CAPACITY increased
Increased from 7/8 to 15/16, supporting σ up to ~4.5 bins (half_width ≤ 7,
range ≤ 15). Previously, sigma_sq ≥ 4.0 caused a `debug_assert!` panic in
`StackWeights::new`. Capacity check is now a runtime `assert!` (memory
safety issue — must fire in release builds too).

#### SSOT-318-03: ParzenConfig::from_intensity_sigma
New SSOT constructor that converts intensity-space sigma to bin-index
sigma², deriving half_width and inv_2sigma_sq in one step. `sigma_sq_in_bins`
now delegates to this internally, eliminating duplicated computation across
6+ call sites in `dispatch.rs` and `compute_image.rs`.

#### SECURE-318-05: Input validation
`ParzenConfig::new` asserts `sigma_sq > 0.0` and `sigma_sq.is_finite()`;
all three public direct-path functions validate non-empty inputs, matching
lengths, `num_bins > 0`, and OOB mask length consistency.

#### ARCH-318-08: PartialEq on ParzenConfig
Enables `assert_eq!` in tests and value comparison.

#### TEST-318-06: 14 new tests
8 unit tests in `direct_types_tests.rs`, 6 property/edge-case tests in
`direct_property_tests.rs`. Includes broad-sigma StackWeights,
from_intensity_sigma SSOT verification, input validation panic tests,
single-bin histogram, and marginal consistency with OOB mask.

| File | Lines | Notes |
|------|-------|-------|
| `direct/types.rs` | 487 | +`ParzenConfig::from_intensity_sigma`, `PartialEq`, input validation, `MAX_PARZEN_BINS=15`, `STACK_WEIGHTS_CAPACITY=16` |
| `direct/mod.rs` | 421 | +input validation, +`ParzenConfig` re-export |
| `direct/direct_types_tests.rs` | 447 | +8 new tests |
| `direct/direct_property_tests.rs` | 451 | +6 new property tests |
| `dispatch.rs` | 220 | `sigma_sq_in_bins` delegates to `ParzenConfig`; dispatch methods use `from_intensity_sigma` |

### Sprint 317 (0.50.79) — Parzen Cache Dispatch Phase Four

#### ARCH-317-01: ParzenConfig + Monomorphized direct path
`ParzenConfig` groups per-axis σ², half-width, and `inv_2sigma_sq` into a single struct, replacing the scattered `compute_half_width_from_sigma_sq` / `-0.5 / sigma_sq` derivations. `SampleWindow` now pre-computes `StackWeights` for both axes, making the direct-path inner loop entirely heap-free — no `SparseWFixedEntry` construction per sample.

#### ARCH-317-04: DRY SampleWindow::mask_val
Shared inner OOB-filter method eliminates duplicated `match oob_mask` / `if mask_val < 0.5` blocks between `new` and `new_moving_only`.

#### SSOT-317-03: Canonical compute_half_width
Moved from `direct/mod.rs` to `direct/types.rs` with unified `sigma_sq` parameter. `sparse.rs` test module delegates when `direct-parzen` is enabled.

#### TEST-317-06: 13 new tests
7 property tests, 3 `ParzenConfig` unit tests, 2 `accumulate_sample` tests, 1 `compute_half_width` SSOT test.

| File | Lines | Notes |
|------|-------|-------|
| `direct/mod.rs` | 353 | Replaced `compute_half_width_from_sigma_sq` with `ParzenConfig`; split `accumulate_sample` into direct/sparse variants |
| `direct/types.rs` | 452 | Added `ParzenConfig`, `MIN_HALF_WIDTH`, `compute_half_width`, `SampleWindow` pre-computed weights |
| `direct/direct_tests.rs` | 340 | Property tests moved out to `direct_property_tests.rs` |
| `direct/direct_types_tests.rs` | 329 | Added `ParzenConfig`, `accumulate_sample`, SSOT tests |
| `direct/direct_property_tests.rs` | 275 | **New** — 7 property tests |
| `sparse.rs` | 418 | `compute_half_width` now takes `sigma_sq` for API consistency |

### Sprint 315 (0.50.77) — Parzen Cache Dispatch Phase Three

#### MEM-315-01: `StackWeights` derives `Copy`

The 32-byte `StackWeights` struct now derives `Copy`, enabling pass-by-value without overhead. An `iter()` method provides zero-cost iteration over active entries, eliminating manual indexing.

#### ARCH-315-03: `HistogramPool` struct

Extracted the duplicated `Mutex<Vec<Vec<f32>>>` pool logic from both `compute_joint_histogram_direct` and `compute_joint_histogram_from_cache_sparse` into a reusable `HistogramPool` struct with `new()`, `checkout()`, and `return_buffer()` methods. Single point of maintenance for buffer pool semantics.

#### PERF-315-02: `accumulate_sample` helper

Monomorphized fold body shared by both direct and sparse paths. Takes `impl IntoIterator<Item = SparseWFixedEntry>`, ensuring consistent optimization across both computation functions without code duplication.

#### ARCH-315-05: `SparseWFixedEntry` newtype

Replaces bare `(usize, f32)` tuples in `SparseWFixedT` with a typed struct providing named field access (`bin`, `weight`) and `Copy` semantics. Prevents accidental index/weight swaps at the type level.

#### FIX-315-04: `sparse.rs` dead code cleanup

Removed `#[allow(dead_code)]` from module declaration; gated test-only functions (`compute_sparse_parzen_weights`, `compute_half_width`, `MIN_HALF_WIDTH`) with `#[cfg(test)]`; removed entirely dead `compute_sparse_parzen_weights_transposed` wrapper.

#### TEST-315-06: 5 new property-based tests

`stack_weights_is_copy`, `accumulate_sample_direct_vs_sparse_weights`, `histogram_symmetry_identical_images`, `histogram_normalization_total_weight`, `histogram_boundary_bins_populated`.

## Sprint 328 (0.50.91) — Per-Sample Weight Normalization

### PERF-328-01: Per-sample weight normalization in direct path

`accumulate_sample_direct` now multiplies each sample's contribution by
`inv_sum_f × inv_sum_m`, where both factors are pre-computed in
`SampleWindow::new` via `ParzenConfig::compute_weights_with_inv_sum()`. The
per-sample total contribution to the histogram is therefore exactly `1.0`
(interior samples; boundary-truncated samples contribute slightly less due
to support clipping).

**Performance characteristic**: histogram total is now σ²-invariant. A
loss function computed from this histogram has a stable dynamic range
across σ hyperparameter sweeps, eliminating the prior `n × 2π` scale factor
that required downstream callers to compensate for σ dependence.

**Code change**: hot-loop body is

```rust
let inv_norm = window.inv_sum_f() * window.inv_sum_m();
for (fi, w_f) in window.f_weights.iter() {
    let row_base = (f_lo_u + fi) * num_bins;
    for (mj, w_m) in window.m_weights.iter() {
        hist[row_base + m_lo_u + mj] += w_f * w_m * inv_norm;
    }
}
```

The `inv_norm` scalar is hoisted out of both inner loops (PERF-327-02/03
already hoisted `f_lo_u` and `m_lo_u`; PERF-328-01 adds the third hoist).

### PERF-328-02: Sparse-path moving-axis normalization

`accumulate_sample_sparse` signature gained an `inv_sum_m: f32` parameter.
Callers (currently `compute_joint_histogram_from_cache_sparse`) pass the
combined `inv_sum_f × inv_sum_m` so the sparse path's per-sample
contribution matches the direct path's `1.0` after normalization.

**Code change**: sparse path is

```rust
fn accumulate_sample_sparse(
    hist: &mut [f32],
    num_bins: usize,
    m_range: BinRange,
    m_weights: &StackWeights,
    inv_sum_m: f32,
    fixed_weights: &[SparseWFixedEntry],
) {
    let m_lo_u = m_range.lo as usize;
    for entry in fixed_weights {
        let row_base = entry.bin as usize * num_bins;
        for (j, w_m) in m_weights.iter() {
            hist[row_base + m_lo_u + j] += entry.weight * w_m * inv_sum_m;
        }
    }
}
```

The `m_lo_u` hoist (PERF-327-03) is preserved. The fixed-axis `1/sum_f` is
folded into the caller's `inv_sum_m` argument via
`SampleWindow::new_moving_only`'s returned factor.

### ARCH-328-04/05: StackWeights/BinRange len() and is_empty() promoted to production

`StackWeights::len()` and `BinRange::len()` are now production-visible
(were `#[cfg(test)]`-gated in Sprints 322-325). Callers in `mod.rs` and
`sample.rs` use them in size-assertion regression tests.

### PERF-328-01: ParzenConfig::compute_weights_with_inv_sum() production API

`ParzenConfig::compute_weights_with_inv_sum(val, num_bins) -> (BinRange, StackWeights, f32)`
returns the bin range, weights, and `1/sum_weights` in one pass — avoiding
the duplicated `StackWeights::new()` work that would be required to compute
the inverse separately.

### SampleWindow memory footprint

`SampleWindow` now carries `inv_sum_f: f32` and `inv_sum_m: f32` (+8 bytes
production). Total production size: ~272 bytes (was ~280; the prior
estimate of ~266 in MEM-325-01 assumed 4 fewer bytes from alignment
padding, which the new fields absorb without growing the struct).

### Test count

Direct path: 168 tests (was 155 in 0.50.90; +13 in `direct_phase_thirteen_tests.rs`).
Total with `--features direct-parzen`: 499 (was 518 in 0.50.90; -19 from
consolidating `direct_phase_twelve_tests.rs` stale tests with the new
normalized expectations).

### Sprint 314 (0.50.76) — Parzen Cache Dispatch Hardening Phase Two

#### PERF-314-01: Parallel `compute_joint_histogram_direct`

The non-cached direct path (called on the first CMA-ES iteration before the sparse cache is built) was the last sequential computation function in the direct Parzen module. Now uses the same rayon `into_par_iter().fold().reduce()` pattern with thread-local histograms that was applied to `compute_joint_histogram_from_cache_sparse` in Sprint 313.

This also eliminates the last `unsafe` pointer arithmetic from the direct Parzen path. The OPT-1 `row_base_pointers` helper and OPT-3 unchecked writes have been replaced by safe indexing into thread-local buffers. Each thread accumulates into its own `[num_bins × num_bins]` buffer; the final reduction sums all thread-local results.

Trade-off: floating-point accumulation order changes, producing ~1e-5 differences vs. the sequential version (within 1e-4 test tolerance).

#### MEM-314-01: Thread-local histogram buffer pool

Both parallel functions (`compute_joint_histogram_direct` and `compute_joint_histogram_from_cache_sparse`) now use a `Mutex<Vec<Vec<f32>>>` pool to reuse thread-local histogram buffers across fold/reduce calls. This avoids repeated `vec![0.0f32; num_bins²]` allocation + zeroing for each fold/reduce invocation.

Pool mechanics:
- On fold initialization: check out a buffer from the pool (or allocate a new one), zero-fill it
- On reduction: sum the local buffer into the accumulator, return the local buffer to the pool
- The pool is scoped to a single function call, so buffers are dropped after the function returns

For 32 bins (1K histogram entries per buffer), this saves ~4 KB allocation per rayon worker per fold call. For 64 bins (4 KB), the savings scale quadratically.

#### ARCH-314-01: SparseWFixedCache trait

The duplicated `get_or_build_sparse_w_fixed` method on `HistogramCache` and `MaskedHistogramCache` was extracted into a `SparseWFixedCache` trait with a default implementation. Both structs implement the trait via accessor methods (`sparse_w_fixed()`, `sparse_w_fixed_mut()`, `take_fixed_norm()`), making the lazy-build logic a single point of maintenance.

#### ARCH-314-02: Cache key collision guard

`MaskedHistogramCache` now stores an optional `data_fingerprint: Option<f32>` — the sum of the first 256 normalized fixed-image values, computed at cache creation time. The public method `validate_masked_cache_fingerprint()` on `ParzenJointHistogram` compares this stored fingerprint against current data, invalidating the cache on mismatch. This provides probabilistic collision detection for partial key collisions.

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
