//! Direct NdArray joint histogram computation — avoids full `[N, num_bins]` weight matrices.
//!
//! On the NdArray (CPU) backend, the dominant cost of Parzen histogram computation is
//! building the `[N, num_bins]` weight matrices via broadcast subtraction, squaring,
//! and exp(). For 32³ = 32768 samples × 32 bins, this allocates and computes ~1M
//! elements per intermediate tensor, with 4 intermediates per weight matrix × 2
//! weight matrices = ~8M elements of temporary data (~32MB).
//!
//! This module provides a direct computation path that:
//! 1. Extracts the normalized intensity values as a flat `Vec<f32>`
//! 2. For each sample, computes the Gaussian weight only for bins within ±3σ
//! 3. Accumulates directly into the `[num_bins, num_bins]` joint histogram
//! 4. Returns the histogram as a Burn tensor
//!
//! This reduces exp() calls by ~4.5× (for 32 bins with σ ≈ 1 bin-width) and
//! eliminates all intermediate `[N, num_bins]` allocations, dramatically reducing
//! memory pressure and improving cache locality.
//!
//! Additionally, a **sparse W_fixed^T** cache path is provided. Instead of
//! storing the full `[num_bins, N]` dense weight matrix (~4MB for 32 bins × 32K
//! samples), each sample's non-zero fixed-image weights are stored as a short
//! `Vec<SparseWFixedEntry>` with ~7 entries. The sparse cache eliminates the
//! inner `0..num_bins` scan and the `if w_f > 0.0` branch in the hot-loop
//! variant, further improving cache locality and reducing memory.
//!
//! **Limitation**: This path is only available for the NdArray backend without
//! autodiff. For autodiff or GPU backends, the standard tensor-based path is used.
//!
//! # Architecture (Phase Seven — Sprint 320)
//!
//! ## DRY sigma² helpers (DRY-320-01)
//!
//! `ParzenJointHistogram::fixed_sigma_cfg()` and `moving_sigma_cfg()`
//! encapsulate the `ParzenConfig::from_intensity_sigma(self.parzen_sigma, ...)`
//! pattern that was repeated at 8 call sites. All callers now use these
//! 1-line helpers instead of the 5-line inline pattern.
//!
//! ## ParzenConfig self-methods (ARCH-320-03)
//!
//! `ParzenConfig::bin_range(val, num_bins)` and `compute_weights(val, num_bins)`
//! encapsulate the `floor → BinRange::new → StackWeights::new` derivation.
//! `SampleWindow::new`, `new_moving_only`, and `build_sparse_w_fixed_transposed`
//! now delegate to `compute_weights` instead of manually constructing bin
//! ranges and weights. This is SRP monomorphization: `ParzenConfig` owns
//! the weight computation, `SampleWindow` owns the per-sample context.
//!
//! ## ParzenConfig::sum_weights (ARCH-320-06)
//!
//! `ParzenConfig::sum_weights(val, num_bins)` provides the discrete Gaussian
//! weight sum for a normalized value. For interior values, this approximates
//! √(2πσ²). Useful for cross-validating the exp-ratchet and for future
//! weight-normalization features.
//!
//! # Architecture (Phase Six — Sprint 319)
//!
//! ## ParzenConfig (SRP / SSOT)
//!
//! [`ParzenConfig`] groups per-axis σ parameters and their derived values
//! (`half_width`, `inv_2sigma_sq`) in a single struct. Both
//! `compute_joint_histogram_direct` and `compute_joint_histogram_from_cache_sparse`
//! construct `ParzenConfig` once and pass it to `SampleWindow` — eliminating
//! the repeated `compute_half_width_from_sigma_sq` / `-0.5 / sigma_sq` derivation
//! that was scattered across both functions.
//!
//! `ParzenConfig::from_intensity_sigma` converts an intensity-space sigma
//! to bin-index sigma², deriving half_width and inv_2sigma_sq in one step.
//! The standalone `sigma_sq_in_bins` function in `dispatch.rs` delegates to
//! this, and `compute.rs` also delegates to it (SSOT-319-01), making
//! `ParzenConfig` the single source of truth for **all** sigma conversions
//! across both the dispatch and tensor paths.
//!
//! ## Exp-ratchet optimisation (PERF-319-04)
//!
//! `StackWeights::new` now uses an **exp-ratchet** technique instead of
//! computing `exp()` independently for each bin. Adjacent integer bins
//! differ by exactly 1 in the `diff` value, so the exponent changes by a
//! fixed increment with a constant second difference. This allows a FMA
//! chain: only the first entry calls `exp()`, and subsequent entries derive
//! their exponent via two additions per step. For the typical 7-bin window,
//! this reduces the cost from `7 × exp()` to `1 × exp() + 6 × fma`,
//! approximately 3× faster. Floating-point drift is bounded by ~15 ULP
//! for the maximum 15-bin window, well within the 1e-4 test tolerance.
//!
//! ## HistogramPool optimisation (PERF-319-05)
//!
//! `HistogramPool::checkout` now drops the Mutex lock before zero-filling
//! or allocating, reducing lock contention under rayon's parallel fold.
//! New allocations skip the redundant `fill(0.0)` since `vec![0.0; N]`
//! already produces a zeroed buffer.
//!
//! ## Monomorphized direct path (ARCH-317-01)
//!
//! `SampleWindow::new` now pre-computes `StackWeights` for **both** the fixed
//! and moving axes. The direct-path `accumulate_sample` consumes the
//! `SampleWindow` directly — no `SparseWFixedEntry` construction, no heap
//! allocation per sample. The sparse-cache path still uses
//! `SparseWFixedEntry` for the fixed axis (the weights come from the cache),
//! but also receives pre-computed `StackWeights` for the moving axis.
//!
//! ## Cross-iteration pool reuse (MEM-317-02)
//!
//! `HistogramPool` is stored on `ParzenJointHistogram` and reused across
//! CMA-ES iterations, amortising the initial `Vec` allocation. The per-call
//! `checkout()` still zero-fills the buffer (O(num_bins²)), but avoids
//! the allocation + deallocation cycle. `compute_joint_histogram_direct`
//! and `compute_joint_histogram_from_cache_sparse` accept an optional
//! `pool: Option<&HistogramPool>` parameter — when `None`, a local pool
//! is created per invocation (backward-compatible fallback).
//!
//! # Inner-loop optimizations
//!
//! - **Exp-ratchet (PERF-319-04):** `StackWeights::new` uses a FMA chain
//!   to compute adjacent-bin exponents incrementally instead of calling
//!   `exp()` per bin. Reduces exp() calls from N to 1 per axis per sample.
//! - **Hoisted moving exp() (OPT-2):** Pre-computed moving weights in
//!   `StackWeights` eliminate redundant exp() calls per sample.
//! - **Stack-allocated weights (OPT-5):** Fixed-size `[f32; 32]` SIMD-aligned
//!   array avoids heap allocation entirely. `StackWeights` is `Copy`.
//!   Supports σ up to ~5.2 bins (half_width ≤ 15, range ≤ 31 bins).
//! - **Precomputed bin ranges (MEM-316-01):** `SampleWindow` computes
//!   bin ranges once, avoiding repeated floor/clamp calculations.
//! - **Lock-free checkout (PERF-319-05):** `HistogramPool::checkout`
//!   drops the Mutex before zero-filling/allocating, reducing contention.
//! - **Parallel reduction (OPT-6):** Both paths use rayon
//!   `into_par_iter().fold().reduce()` with thread-local histograms.
//! - **Histogram pool (ARCH-315-03):** Reusable buffer pool with Mutex
//!   poison recovery.
//! - **Monomorphized fold body (PERF-315-02):** The `accumulate_sample`
//!   helper factors out the common histogram accumulation.
//! - **Branch-eliminated accumulate (FIX-316-07):** OOB check folded
//!   into `SampleWindow::new` / `new_moving_only` via `Option`.
//!
//! # Safety
//!
//! This module contains **no `unsafe` code**. All parallelism is provided by
//! rayon's safe abstractions (`into_par_iter`, `fold`, `reduce`). The
//! `HistogramPool` uses `Mutex` with poison recovery
//! (`unwrap_or_else(|e| e.into_inner())`) so a panic in one thread does not
//! propagate to others.
//!
//! `StackWeights` uses a fixed-size `[f32; 32]` array with zero-filled padding
//! — entries beyond `len` are always `0.0f32`, never uninitialized memory.
//!
//! # Examples
//!
//! ```ignore
//! use ritk_registration::metric::histogram::compute_joint_histogram_direct;
//!
//! let fixed = vec![15.3, 20.7, 10.1]; // normalized to [0, num_bins-1]
//! let moving = vec![12.0, 18.5, 8.0];
//! let num_bins = 32;
//! let sigma_sq = 1.0; // in bin-index units
//!
//! let hist_data = compute_joint_histogram_direct(
//!     &fixed, &moving, num_bins, sigma_sq, sigma_sq, None, None,
//! );
//! ```

pub(crate) mod pool;
pub(crate) mod sample;
pub(crate) mod types;

use burn::tensor::{Shape, TensorData};
use rayon::prelude::*;
use sample::SampleWindow;
use types::{BinRange, StackWeights};

// Re-export for sparse.rs delegation when direct-parzen is enabled.
pub use pool::HistogramPool;
pub use sample::{SparseWFixedEntry, SparseWFixedT};

#[cfg(test)]
pub(crate) use types::compute_half_width;
pub(crate) use types::ParzenConfig;

// ── Monomorphized fold body (PERF-315-02, ARCH-317-01) ─────────────────────

/// Accumulate a single sample's contribution into the joint histogram
/// (direct path — both axes pre-computed in `SampleWindow`).
///
/// This is the heap-free hot-loop body for `compute_joint_histogram_direct`.
/// Both fixed and moving weights are already in `StackWeights` form inside
/// the `SampleWindow`, so no `SparseWFixedEntry` construction occurs.
///
/// Returns the total weight contributed by this sample (the sum of all
/// `w_f * w_m` products). This enables per-sample validation in tests
/// without affecting the production hot-loop.
#[inline(always)]
fn accumulate_sample_direct(hist: &mut [f32], num_bins: usize, window: &SampleWindow) -> f32 {
    let mut total = 0.0f32;
    for (fi, w_f) in window.f_weights.iter() {
        let a = window.f_range.lo + fi;
        let row_base = a * num_bins;
        for (mj, w_m) in window.m_weights.iter() {
            let val = w_f * w_m;
            hist[row_base + window.m_range.lo + mj] += val;
            total += val;
        }
    }
    total
}

/// Accumulate a single sample's contribution into the joint histogram
/// (sparse-cache path — fixed weights from cache, moving pre-computed).
///
/// This is the hot-loop body for `compute_joint_histogram_from_cache_sparse`.
/// The fixed weights come from the sparse cache (as `SparseWFixedEntry`
/// iterators), and the moving weights are pre-computed in `StackWeights`.
#[inline(always)]
fn accumulate_sample_sparse(
    hist: &mut [f32],
    num_bins: usize,
    m_range: BinRange,
    m_weights: &StackWeights,
    fixed_weights: impl IntoIterator<Item = SparseWFixedEntry>,
) {
    for entry in fixed_weights {
        let row_base = entry.bin * num_bins;
        for (j, w_m) in m_weights.iter() {
            hist[row_base + m_range.lo + j] += entry.weight * w_m;
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Compute the joint histogram directly from normalized intensity values.
///
/// This is the hot-path optimization for the NdArray backend: instead of building
/// full `[N, num_bins]` Parzen weight matrices and multiplying them, we compute
/// the histogram by iterating over samples and accumulating each sample's
/// contribution directly into the `[num_bins, num_bins]` result.
///
/// Both fixed and moving Parzen weights are pre-computed as `StackWeights`
/// inside `SampleWindow`, making the inner loop entirely heap-free per sample.
/// This eliminates the `SparseWFixedEntry` construction that was previously
/// done per sample in the direct path.
///
/// Uses rayon parallel reduction (OPT-6): each thread accumulates into its own
/// thread-local histogram buffer, then all buffers are summed in the final
/// reduction phase. This eliminates synchronization from the hot loop —
/// no locks, no atomics, no `unsafe` pointer arithmetic.
///
/// # Arguments
/// * `fixed_norm` — Normalized fixed-image values `[N]` in `[0, num_bins - 1]`
/// * `moving_norm` — Normalized moving-image values `[N]` in `[0, num_bins - 1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_fix` — Fixed-image Parzen sigma² in bin-index units
/// * `sigma_sq_mov` — Moving-image Parzen sigma² in bin-index units
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = out-of-bounds)
///
/// # Returns
/// Joint histogram `[num_bins, num_bins]` as a TensorData object.
///
/// # Parallel accumulation trade-off
///
/// Floating-point accumulation order changes under parallel reduction,
/// producing ~1e-5 differences vs. the sequential version. This is within
/// the 1e-4 tolerance used by the test suite.
#[allow(private_interfaces)]
pub fn compute_joint_histogram_direct(
    fixed_norm: &[f32],
    moving_norm: &[f32],
    num_bins: usize,
    sigma_sq_fix: f32,
    sigma_sq_mov: f32,
    oob_mask: Option<&[f32]>,
    pool: Option<&HistogramPool>,
) -> TensorData {
    // Input validation
    assert!(!fixed_norm.is_empty(), "fixed_norm must not be empty");
    assert!(!moving_norm.is_empty(), "moving_norm must not be empty");
    assert_eq!(
        fixed_norm.len(),
        moving_norm.len(),
        "fixed_norm and moving_norm must have same length"
    );
    assert!(num_bins > 0, "num_bins must be > 0");
    if let Some(mask) = oob_mask {
        assert_eq!(
            mask.len(),
            fixed_norm.len(),
            "oob_mask length must match sample count"
        );
    }

    let n = fixed_norm.len();
    let fix_cfg = ParzenConfig::new(sigma_sq_fix);
    let mov_cfg = ParzenConfig::new(sigma_sq_mov);
    let local_pool_if_none;
    let pool: &HistogramPool = match pool {
        Some(p) => p,
        None => {
            local_pool_if_none = HistogramPool::new(num_bins * num_bins);
            &local_pool_if_none
        }
    };

    let histogram: Vec<f32> = (0..n)
        .into_par_iter()
        .fold(
            || pool.checkout(),
            |mut local_hist, i| {
                if let Some(window) = SampleWindow::new(
                    i,
                    fixed_norm,
                    moving_norm,
                    num_bins,
                    &fix_cfg,
                    &mov_cfg,
                    oob_mask,
                ) {
                    accumulate_sample_direct(&mut local_hist, num_bins, &window);
                }
                local_hist
            },
        )
        .reduce(
            || pool.checkout(),
            |mut acc, local| {
                for (dst, src) in acc.iter_mut().zip(local.iter()) {
                    *dst += src;
                }
                pool.return_buffer(local);
                acc
            },
        );

    TensorData::new(histogram, Shape::new([num_bins, num_bins]))
}

/// Build the sparse W_fixed^T cache from normalized fixed-image values.
///
/// For each sample, this computes the Gaussian Parzen weights only for bins
/// within ±3σ of the primary bin, storing `(bin_index, weight)` pairs where
/// the weight exceeds 1e-12. OOB samples receive an empty Vec.
///
/// # Arguments
/// * `fixed_norm` — Normalized fixed-image values `[N]` in `[0, num_bins - 1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_fix` — Fixed-image Parzen sigma² in bin-index units
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = out-of-bounds)
pub fn build_sparse_w_fixed_transposed(
    fixed_norm: &[f32],
    num_bins: usize,
    sigma_sq_fix: f32,
    oob_mask: Option<&[f32]>,
) -> SparseWFixedT {
    // Input validation
    assert!(!fixed_norm.is_empty(), "fixed_norm must not be empty");
    assert!(num_bins > 0, "num_bins must be > 0");
    if let Some(mask) = oob_mask {
        assert_eq!(
            mask.len(),
            fixed_norm.len(),
            "oob_mask length must match sample count"
        );
    }

    let n = fixed_norm.len();
    let fix_cfg = ParzenConfig::new(sigma_sq_fix);

    let mut entries: SparseWFixedT = (0..n).map(|_| Vec::with_capacity(7)).collect();
    entries.par_iter_mut().enumerate().for_each(|(i, entry)| {
        // OOB check — inline rather than SampleWindow because we only need
        // the fixed axis and build SparseWFixedEntry, not StackWeights.
        let mask_val = match oob_mask {
            Some(m) => m[i],
            None => 1.0,
        };
        if mask_val < 0.5 {
            return;
        }
        let f_val = fixed_norm[i];
        // ARCH-320-03: delegate to ParzenConfig::compute_weights
        let (f_range, f_weights) = fix_cfg.compute_weights(f_val, num_bins);
        for (j, w_f) in f_weights.iter() {
            if w_f > 1e-12 {
                entry.push(SparseWFixedEntry::new(f_range.lo + j, w_f));
            }
        }
    });
    entries
}

/// Compute the joint histogram from a sparse W_fixed^T cache and live moving values.
///
/// This is the sparse hot-loop variant used on every CMA-ES iteration after the
/// first. Only the moving-image weights are recomputed (as `StackWeights` inside
/// `SampleWindow`); the fixed-image weights are provided as a pre-computed sparse
/// cache. The inner loop iterates only over the ~7 non-zero entries per sample,
/// eliminating the full `0..num_bins` scan and the `if w_f > 0.0` branch required
/// by the dense cache path.
///
/// Uses rayon parallel reduction (OPT-6) with a thread-local histogram pool
/// to avoid repeated allocation + zeroing.
///
/// # Arguments
/// * `sparse_w_fixed` — Sparse fixed-image weights per sample (from `build_sparse_w_fixed_transposed`)
/// * `moving_norm` — Normalized moving-image values `[N]` in `[0, num_bins - 1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_mov` — Moving-image Parzen sigma² in bin-index units
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = out-of-bounds)
#[allow(private_interfaces)]
pub fn compute_joint_histogram_from_cache_sparse(
    sparse_w_fixed: &SparseWFixedT,
    moving_norm: &[f32],
    num_bins: usize,
    sigma_sq_mov: f32,
    oob_mask: Option<&[f32]>,
    pool: Option<&HistogramPool>,
) -> TensorData {
    // Input validation
    assert!(
        !sparse_w_fixed.is_empty(),
        "sparse_w_fixed must not be empty"
    );
    assert_eq!(
        sparse_w_fixed.len(),
        moving_norm.len(),
        "sparse_w_fixed and moving_norm must have same length"
    );
    assert!(num_bins > 0, "num_bins must be > 0");
    if let Some(mask) = oob_mask {
        assert_eq!(
            mask.len(),
            moving_norm.len(),
            "oob_mask length must match sample count"
        );
    }

    let n = sparse_w_fixed.len();
    let mov_cfg = ParzenConfig::new(sigma_sq_mov);
    let local_pool_if_none;
    let pool: &HistogramPool = match pool {
        Some(p) => p,
        None => {
            local_pool_if_none = HistogramPool::new(num_bins * num_bins);
            &local_pool_if_none
        }
    };

    let histogram: Vec<f32> = (0..n)
        .into_par_iter()
        .fold(
            || pool.checkout(),
            |mut local_hist, i| {
                if let Some((_m_val, m_range, m_weights)) =
                    SampleWindow::new_moving_only(i, moving_norm, num_bins, &mov_cfg, oob_mask)
                {
                    accumulate_sample_sparse(
                        &mut local_hist,
                        num_bins,
                        m_range,
                        &m_weights,
                        sparse_w_fixed[i].iter().copied(),
                    );
                }
                local_hist
            },
        )
        .reduce(
            || pool.checkout(),
            |mut acc, local| {
                for (dst, src) in acc.iter_mut().zip(local.iter()) {
                    *dst += src;
                }
                pool.return_buffer(local);
                acc
            },
        );

    TensorData::new(histogram, Shape::new([num_bins, num_bins]))
}

#[cfg(test)]
mod direct_phase_seven_tests;
#[cfg(test)]
mod direct_phase_six_tests;
#[cfg(test)]
mod direct_property_tests;
#[cfg(test)]
mod direct_tests;
#[cfg(test)]
mod direct_types_tests;
