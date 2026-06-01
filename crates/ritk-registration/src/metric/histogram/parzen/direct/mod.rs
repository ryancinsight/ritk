//! Direct NdArray joint histogram computation — avoids full `[N, num_bins]` weight matrices.
//!
//! On NdArray (CPU), building `[N, num_bins]` weight matrices dominates cost
//! (~8M temporaries, ~32MB for 32³×32). This module computes each sample's
//! Gaussian weights only within ±3σ and accumulates directly into the
//! `[num_bins, num_bins]` joint histogram, reducing exp() calls ~4.5× and
//! eliminating all `[N, num_bins]` allocations.
//!
//! A **sparse W_fixed^T** cache path stores only ~7 non-zero entries per sample
//! (vs. ~4MB dense), eliminating the inner `0..num_bins` scan and `if w_f > 0.0`
//! branch. **Limitation**: NdArray backend only, no autodiff.
//!
//! # Architecture (Phase Fourteen — Sprint 329)
//!
//! - **SPARSE-329-01:** Full joint normalization in sparse path — `inv_sum_f`
//! stored per-sample in `SparseWFixedT` alongside fixed entries; sparse path
//! computes `inv_norm = inv_sum_f × inv_sum_m`, matching the direct path.
//! Direct↔sparse histograms are now numerically identical (no more scale
//! difference from missing `inv_sum_f`).
//! - **PERF-329-02:** FMA-idiomatic inner loop — `hist[idx] += w_f * w_m * inv_norm`
//! is a canonical FMA pattern that LLVM auto-fuses into `vfmadd231ps`.
//!
//! ## Direct↔sparse parity (SPARSE-329-01)
//!
//! Both paths now apply the same full joint normalization:
//! - Direct: `inv_norm = inv_sum_f() × inv_sum_m()` from `SampleWindow`
//! - Sparse: `inv_norm = sparse_cache.inv_sum_f × inv_sum_m` from
//! `SparseWFixedT` + `SampleWindow::new_moving_only`
//!
//! This eliminates the Sprint 328 asymmetry where the sparse path only
//! normalized by `1/sum_m`, producing histograms scaled by `sum_f ≈ √(2πσ²)`.
//!
//! ## σ²-invariance property
//!
//! Per-sample contribution to histogram total is `w_f · w_m · 1/(sum_f · sum_m) = 1.0`
//! for interior samples (no boundary truncation). Boundary-truncated samples
//! contribute slightly less because `sum_f × sum_m` is smaller for clipped
//! windows. The histogram total equals the number of in-bounds samples,
//! regardless of σ². This eliminates a previously-implicit scale factor
//! `n × 2π` that the loss function and gradient had to compensate for.
//!
//! # Prior phases (S327–328)
//!
//! - **S328:** Per-sample normalization (PERF-328-01), `inv_sum_f`/`inv_sum_m` on
//! `SampleWindow`, `compute_weights_with_inv_sum()` API, sparse-path moving-axis
//! normalization only (partial — no `inv_sum_f` in sparse cache).
//! - **S326-327:** `SparseWFixedEntry.bin` u16, `extract_oob_mask()` DRY, hoisted
//! offsets, dead `total` removal, `validate_inputs()` SSOT.
//! - **S325:** `StackWeights.len` u8, `BinRange::new` assert, `merge_histograms` idiomatic.
//! - **S324:** `BinRange` u16, `accumulate_sample_sparse` monomorphized, `merge_histograms` extracted.
//! - **S319–323:** `ParzenConfig` SSOT, exp-ratchet, pool, `SampleWindow`, `StackWeightsIter`.
//! - **S319–320:** `ParzenConfig` SRP, exp-ratchet FMA chain, pool checkout.
//!
//! # Inner-loop optimizations
//!
//! - **Exp-ratchet (PERF-319-04):** FMA chain for adjacent exponents; 1 exp() per axis.
//! - **Hoisted moving exp() (OPT-2):** Pre-computed `StackWeights`.
//! - **Stack weights (OPT-5):** `[f32; 32]` SIMD-aligned, `Copy`, no heap. σ ≤ ~5.2 bins.
//! - **Precomputed bin ranges (MEM-316-01):** `SampleWindow` computes once.
//! - **Lock-free checkout (PERF-319-05):** Mutex dropped before zero-fill.
//! - **Parallel reduction (OPT-6):** rayon `fold().reduce()` with thread-local histograms.
//! - **Histogram pool (ARCH-315-03):** Reusable buffers, Mutex poison recovery.
//! - **Monomorphized fold (PERF-315-02):** `accumulate_sample` factors out common body.
//! - **Branch-eliminated (FIX-316-07):** OOB folded into `Option`-returning constructors.
//!
//! # Safety
//!
//! **No `unsafe` code.** Parallelism via rayon safe abstractions. `HistogramPool`
//! uses `Mutex` with poison recovery so panics don't propagate.
//!
//! `StackWeights`: zero-filled `[f32; 32]` — entries beyond `len` are `0.0`,
//! never uninit. `StackWeightsIter` uses safe `[]` indexing, always in-bounds.
//!
//! `BinRange` u16 (MEM-324-04): `as usize` lossless; `BinRange::new` asserts
//! `num_bins ≤ u16::MAX` (MEM-325-02). `StackWeights.len` u8 (MEM-325-01):
//! `as usize` lossless; max active = 31 < `u8::MAX`.
//!
//! PERF-328-01 / SPARSE-329-01: `inv_sum_f`/`inv_sum_m` = `1/sum_weights` where
//! sum > 0 for in-bounds samples (≥1 bin with Gaussian weight > 0). Division
//! is safe. OOB samples store `inv_sum_f = 0.0` in `SparseWFixedT`; they are
//! excluded by `SampleWindow::mask_val` so the zero value is never used.
//!
//! # Examples
//!
//! ```ignore
//! use ritk_registration::metric::histogram::compute_joint_histogram_direct;
//!
//! let fixed = vec![15.3, 20.7, 10.1];
//! let moving = vec![12.0, 18.5, 8.0];
//! let hist_data = compute_joint_histogram_direct(
//!     &fixed, &moving, 32, 1.0, 1.0, None, None,
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

/// Memory sizes of direct-Parzen types after field compaction.
///
/// Exposed for benchmark size-regression testing. The `u8`/`u16` field
/// compactions (MEM-325-01, MEM-324-04, PERF-326-02, MEM-328-03) significantly
/// reduced struct sizes, improving cache locality in the hot loops.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct CompactionSizes {
    /// `StackWeights` — ~128 bytes with `u8` len (was ~136 with `usize`).
    pub stack_weights: usize,
    /// `BinRange` — 4 bytes with `u16` fields (was 16 with `usize`).
    pub bin_range: usize,
    /// `ParzenConfig` — 24 bytes with `usize` half_width (u16 attempt reverted; see types.rs).
    pub parzen_config: usize,
    /// `SampleWindow` — ~272 bytes production with `u8` len, `u16` bin compactions (was ~304).
    pub sample_window: usize,
    /// `SparseWFixedEntry` — 8 bytes with `u16` bin (was 16 with `usize`).
    pub sparse_fixed_entry: usize,
}

/// Return `size_of` for key direct-Parzen types (benchmark regression guard).
#[doc(hidden)]
#[inline]
pub fn compaction_sizes() -> CompactionSizes {
    CompactionSizes {
        stack_weights: std::mem::size_of::<types::StackWeights>(),
        bin_range: std::mem::size_of::<types::BinRange>(),
        parzen_config: std::mem::size_of::<types::ParzenConfig>(),
        sample_window: std::mem::size_of::<sample::SampleWindow>(),
        sparse_fixed_entry: std::mem::size_of::<sample::SparseWFixedEntry>(),
    }
}

#[cfg(test)]
pub(crate) use types::compute_half_width;
pub(crate) use types::ParzenConfig;

// ── Monomorphized fold body (PERF-315-02, ARCH-317-01, PERF-327-02/04) ────

/// Accumulate one sample into the joint histogram (direct path).
///
/// Heap-free hot-loop body for `compute_joint_histogram_direct`.
/// Both fixed/moving weights are `StackWeights` in `SampleWindow` — no
/// `SparseWFixedEntry` construction.
///
/// Base offsets `f_lo_u`/`m_lo_u` hoisted out of loops (PERF-327-02).
/// Returns `()` (PERF-327-04); former `f32` return was only used by one test.
///
/// PERF-329-02: Inner loop `hist[idx] += w_f * w_m * inv_norm` is the canonical
/// FMA pattern that LLVM auto-fuses into `vfmadd231ps` on AVX2. The
/// three-multiply form `a * b * c + d` is recognized by both GCC and LLVM
/// without explicit `mul_add`. Explicit `mul_add` with hoisted `w_f * inv_norm`
/// was benchmarked to be ~8% slower (less register-efficient for the
/// 7-outer / 7-inner loop), so the original form is retained.
///
/// Both paths accumulate normalized `w_f × w_m × inv_norm` for
/// direct↔sparse parity (SPARSE-329-01).
#[inline(always)]
fn accumulate_sample_direct(hist: &mut [f32], num_bins: usize, window: &SampleWindow) {
    let f_lo_u = window.f_range().lo as usize; // PERF-327-02: hoisted
    let m_lo_u = window.m_range().lo as usize; // PERF-327-02: hoisted
    let inv_norm = window.inv_sum_f() * window.inv_sum_m(); // PERF-328-01
                                                            // The `hist[idx] += w_f * w_m * inv_norm` form is the canonical FMA
                                                            // pattern that LLVM auto-fuses into `vfmadd231ps` on AVX2 (PERF-329-02
                                                            // docs). Explicit `mul_add` form with hoisted `w_f * inv_norm` was
                                                            // benchmarked to be ~8% slower (less register-efficient for the
                                                            // 7-outer / 7-inner loop), so the original form is retained.
    for (fi, w_f) in window.f_weights.iter() {
        let row_base = (f_lo_u + fi) * num_bins;
        for (mj, w_m) in window.m_weights.iter() {
            hist[row_base + m_lo_u + mj] += w_f * w_m * inv_norm;
        }
    }
}

/// Accumulate one sample into the joint histogram (sparse-cache path).
///
/// Hot-loop body for `compute_joint_histogram_from_cache_sparse`.
/// Fixed weights from sparse cache (`SparseWFixedEntry`), moving from `StackWeights`.
///
/// `m_lo_u` hoisted out of inner loop (PERF-327-03). Both paths accumulate
/// normalized `w_f × w_m × inv_norm` for direct↔sparse parity (SPARSE-329-01).
///
/// SPARSE-329-01: `inv_sum_f` is now stored in the sparse cache alongside
/// the fixed entries, enabling full joint normalization. The `inv_norm`
/// parameter combines `inv_sum_f × inv_sum_m`, matching the direct path.
///
/// PERF-329-02: Same FMA-idiomatic form as `accumulate_sample_direct`.
#[inline(always)]
fn accumulate_sample_sparse(
    hist: &mut [f32],
    num_bins: usize,
    m_range: BinRange,
    m_weights: &StackWeights,
    inv_norm: f32,
    fixed_weights: &[SparseWFixedEntry],
) {
    let m_lo_u = m_range.lo as usize; // PERF-327-03: hoisted
                                      // Same as direct path: `+= w_f * w_m * inv_norm` is the canonical
                                      // FMA pattern LLVM fuses. See accumulate_sample_direct for the
                                      // PERF-329-02 benchmark note.
    for entry in fixed_weights {
        let row_base = entry.bin as usize * num_bins;
        for (j, w_m) in m_weights.iter() {
            hist[row_base + m_lo_u + j] += entry.weight * w_m * inv_norm;
        }
    }
}

/// Merge `src` into `dst` element-wise (PERF-324-05, PERF-325-03).
///
/// Extracted from rayon reduce closure for inlining/auto-vectorization.
/// PERF-325-03: `iter_mut().zip(iter())` is the idiomatic form LLVM
/// auto-vectorizes most reliably (AVX2 `vmovups`/`vaddps`/`vmovups`
/// when slice len is a multiple of 8).
#[inline(always)]
fn merge_histograms(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

// ── Input validation (DRY-327-05) ──────────────────────────────────────────

/// Validate shared inputs for direct-path histogram functions (DRY-327-05).
///
/// SSOT for common `num_bins > 0` and optional `oob_mask` length checks
/// previously duplicated across three public functions.
#[inline]
fn validate_inputs(num_bins: usize, sample_count: usize, oob_mask: Option<&[f32]>) {
    assert!(num_bins > 0, "num_bins must be > 0");
    if let Some(mask) = oob_mask {
        assert_eq!(
            mask.len(),
            sample_count,
            "oob_mask length must match sample count"
        );
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Compute the joint histogram directly from normalized intensity values.
///
/// NdArray hot-path: iterates samples, accumulates directly into
/// `[num_bins, num_bins]` instead of building full `[N, num_bins]` weight
/// matrices. Fixed/moving weights pre-computed as `StackWeights` in
/// `SampleWindow` — heap-free inner loop, no `SparseWFixedEntry`.
///
/// Rayon parallel reduction (OPT-6): thread-local histograms merged in
/// reduce phase — no locks, atomics, or `unsafe`.
///
/// # Arguments
/// * `fixed_norm` — Normalized fixed-image values `[N]` in `[0, num_bins-1]`
/// * `moving_norm` — Normalized moving-image values `[N]` in `[0, num_bins-1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_fix` — Fixed-image Parzen sigma² (bin-index units)
/// * `sigma_sq_mov` — Moving-image Parzen sigma² (bin-index units)
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = OOB)
///
/// # Returns
/// Joint histogram `[num_bins, num_bins]` as TensorData.
///
/// # Parallel accumulation trade-off
///
/// Float accumulation order differs under parallel reduction (~1e-5 vs
/// sequential), within the 1e-4 test tolerance.
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
    // Input validation (DRY-327-05)
    assert!(!fixed_norm.is_empty(), "fixed_norm must not be empty");
    assert!(!moving_norm.is_empty(), "moving_norm must not be empty");
    assert_eq!(
        fixed_norm.len(),
        moving_norm.len(),
        "fixed_norm and moving_norm must have same length"
    );
    validate_inputs(num_bins, fixed_norm.len(), oob_mask);

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
                merge_histograms(&mut acc, &local);
                pool.return_buffer(local);
                acc
            },
        );

    TensorData::new(histogram, Shape::new([num_bins, num_bins]))
}

/// Build the sparse W_fixed^T cache from normalized fixed-image values.
///
/// Per sample, computes Gaussian weights only within ±3σ, storing
/// `(bin_index, weight)` pairs where weight > 1e-12. OOB samples get empty Vec.
/// Each entry also stores `inv_sum_f = 1/sum_f` (SPARSE-329-01) so the
/// sparse path can apply full joint normalization matching the direct path.
///
/// # Arguments
/// * `fixed_norm` — Normalized fixed-image values `[N]` in `[0, num_bins-1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_fix` — Fixed-image Parzen sigma² (bin-index units)
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = OOB)
pub fn build_sparse_w_fixed_transposed(
    fixed_norm: &[f32],
    num_bins: usize,
    sigma_sq_fix: f32,
    oob_mask: Option<&[f32]>,
) -> SparseWFixedT {
    // Input validation (DRY-327-05)
    assert!(!fixed_norm.is_empty(), "fixed_norm must not be empty");
    validate_inputs(num_bins, fixed_norm.len(), oob_mask);

    let n = fixed_norm.len();
    let fix_cfg = ParzenConfig::new(sigma_sq_fix);

    // SPARSE-329-01: each element is (entries, inv_sum_f)
    let mut entries: SparseWFixedT = (0..n).map(|_| (Vec::with_capacity(7), 0.0f32)).collect();
    entries.par_iter_mut().enumerate().for_each(|(i, entry)| {
        // OOB check — reuse SampleWindow::mask_val (ARCH-321-04)
        if SampleWindow::mask_val(i, oob_mask).is_none() {
            return;
        }
        let f_val = fixed_norm[i];
        // SPARSE-329-01: compute weights and inv_sum in one pass
        let (f_range, f_weights, inv_sum_f) = fix_cfg.compute_weights_with_inv_sum(f_val, num_bins);
        for (j, w_f) in f_weights.iter() {
            if w_f > 1e-12 {
                entry
                    .0
                    .push(SparseWFixedEntry::new(f_range.lo + j as u16, w_f));
            }
        }
        entry.1 = inv_sum_f; // SPARSE-329-01
    });
    entries
}

/// Compute the joint histogram from a sparse W_fixed^T cache and live moving values.
///
/// Sparse hot-loop variant for CMA-ES iterations after the first. Only moving
/// weights recomputed (`StackWeights`); fixed weights from pre-computed sparse
/// cache (~7 non-zero entries/sample, eliminating full `0..num_bins` scan and
/// `if w_f > 0.0` branch). Rayon parallel reduction (OPT-6) with histogram pool.
///
/// SPARSE-329-01: Full joint normalization `inv_norm = inv_sum_f × inv_sum_m`
/// is now applied, matching the direct path. `inv_sum_f` is stored per-sample
/// in the sparse cache; `inv_sum_m` is computed per-sample from moving values.
/// This eliminates the asymmetry where the sparse path only normalized by
/// `1/sum_m` (Sprint 328), making direct↔sparse histograms numerically identical.
///
/// # Arguments
/// * `sparse_w_fixed` — Sparse fixed-image weights per sample (from `build_sparse_w_fixed_transposed`)
/// * `moving_norm` — Normalized moving-image values `[N]` in `[0, num_bins-1]`
/// * `num_bins` — Number of histogram bins
/// * `sigma_sq_mov` — Moving-image Parzen sigma² (bin-index units)
/// * `oob_mask` — Optional OOB mask `[N]` (1.0 = in-bounds, 0.0 = OOB)
#[allow(private_interfaces)]
pub fn compute_joint_histogram_from_cache_sparse(
    sparse_w_fixed: &SparseWFixedT,
    moving_norm: &[f32],
    num_bins: usize,
    sigma_sq_mov: f32,
    oob_mask: Option<&[f32]>,
    pool: Option<&HistogramPool>,
) -> TensorData {
    // Input validation (DRY-327-05)
    assert!(
        !sparse_w_fixed.is_empty(),
        "sparse_w_fixed must not be empty"
    );
    assert_eq!(
        sparse_w_fixed.len(),
        moving_norm.len(),
        "sparse_w_fixed and moving_norm must have same length"
    );
    validate_inputs(num_bins, sparse_w_fixed.len(), oob_mask);

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
                if let Some((_m_val, m_range, m_weights, inv_sum_m)) =
                    SampleWindow::new_moving_only(i, moving_norm, num_bins, &mov_cfg, oob_mask)
                {
                    // SPARSE-329-01: combine inv_sum_f from cache with inv_sum_m
                    let inv_sum_f = sparse_w_fixed[i].1; // per-sample inv_sum_f
                    let inv_norm = inv_sum_f * inv_sum_m; // full joint normalization
                    accumulate_sample_sparse(
                        &mut local_hist,
                        num_bins,
                        m_range,
                        &m_weights,
                        inv_norm,
                        &sparse_w_fixed[i].0, // fixed entries
                    );
                }
                local_hist
            },
        )
        .reduce(
            || pool.checkout(),
            |mut acc, local| {
                merge_histograms(&mut acc, &local);
                pool.return_buffer(local);
                acc
            },
        );

    TensorData::new(histogram, Shape::new([num_bins, num_bins]))
}

#[cfg(test)]
mod direct_phase_eight_tests;
#[cfg(test)]
mod direct_phase_eleven_tests;
#[cfg(test)]
mod direct_phase_nine_tests;
#[cfg(test)]
mod direct_phase_seven_tests;
#[cfg(test)]
mod direct_phase_six_tests;
#[cfg(test)]
mod direct_phase_ten_tests;
#[cfg(test)]
mod direct_phase_thirteen_tests;
#[cfg(test)]
mod direct_phase_twelve_tests;
#[cfg(test)]
mod direct_property_proptest;
#[cfg(test)]
mod direct_property_tests;
#[cfg(test)]
mod direct_tests;
#[cfg(test)]
mod direct_types_tests;
