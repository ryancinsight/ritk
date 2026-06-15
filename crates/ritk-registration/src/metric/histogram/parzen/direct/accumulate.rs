//! Monomorphized fold bodies and input-validation helpers.
//!
//! Extracted from `mod.rs` (ARCH-330-04) for SRP: each function handles
//! a single concern in the histogram accumulation pipeline.
//!
//! - `accumulate_sample_direct` — direct-path per-sample histogram accumulation
//! - `accumulate_sample_sparse` — sparse-cache-path per-sample accumulation
//! - `merge_histograms` — element-wise histogram merge for parallel reduction
//! - `validate_inputs` — shared input validation (DRY-327-05)

use super::sample::{SampleWindow, SparseWFixedEntry};
use super::types::{BinRange, StackWeights};

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
pub(crate) fn accumulate_sample_direct(hist: &mut [f32], num_bins: usize, window: &SampleWindow) {
    let f_lo_u = window.f_range().lo as usize; // PERF-327-02: hoisted
    let m_lo_u = window.m_range().lo as usize; // PERF-327-02: hoisted
                                               // Invariant: weights length matches bin-range length for each axis.
    debug_assert_eq!(window.f_weights.len(), window.f_range().len());
    debug_assert_eq!(window.m_weights.len(), window.m_range().len());
    let inv_norm = window.inv_sum_f() * window.inv_sum_m(); // PERF-328-01
                                                            // FIX-PROP-NAN-355: skip the sample when `inv_norm` is non-finite
                                                            // (e.g. `+inf * 0.0 = NaN` or `+inf * +inf = +inf`). The contribution
                                                            // is effectively zero for a fully out-of-support sample, and propagating
                                                            // NaN poisons every downstream histogram bin and the entropy
                                                            // computation in `metric::entropy::entropy`. Branch is predicted
                                                            // never-taken in the hot loop, so it costs ~0 cycles in normal use.
    if !inv_norm.is_finite() {
        return;
    }
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
pub(crate) fn accumulate_sample_sparse(
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
pub(crate) fn merge_histograms(dst: &mut [f32], src: &[f32]) {
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
pub(crate) fn validate_inputs(num_bins: usize, sample_count: usize, oob_mask: Option<&[f32]>) {
    assert!(num_bins > 0, "num_bins must be > 0");
    if let Some(mask) = oob_mask {
        assert_eq!(
            mask.len(),
            sample_count,
            "oob_mask length must match sample count"
        );
    }
}
