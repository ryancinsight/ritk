//! Direct NdArray joint histogram computation â€” avoids full `[N, num_bins]` weight matrices.
//!
//! On NdArray (CPU), building `[N, num_bins]` weight matrices dominates cost
//! (~8M temporaries, ~32MB for 32Â³Ã—32). This module computes each sample's
//! Gaussian weights only within Â±3Ïƒ and accumulates directly into the
//! `[num_bins, num_bins]` joint histogram, reducing exp() calls ~4.5Ã— and
//! eliminating all `[N, num_bins]` allocations.
//!
//! A **sparse W_fixed^T** cache path stores only ~7 non-zero entries per sample
//! (vs. ~4MB dense), eliminating the inner `0..num_bins` scan and `if w_f > 0.0`
//! branch. **Limitation**: NdArray backend only, no autodiff.
//!
//! # Architecture (Phase Fifteen â€” Sprint 330)
//!
//! - **ARCH-330-01:** Deep vertical file hierarchy â€” `types/` decomposed into
//!   `half_width.rs`, `stack_weights.rs`, `bin_range.rs`, `parzen_config.rs`.
//! - **ARCH-330-02:** `sample/` decomposed into `sample_window.rs` and
//!   `sparse_entry.rs`.
//! - **ARCH-330-03:** `ParzenConfig::half_width()` and `inv_2sigma_sq()`
//!   promoted to production API (were `#[cfg(test)]`-gated).
//! - **ARCH-330-04:** Computation functions extracted into `accumulate.rs`,
//!   `compute_direct.rs`, `compute_sparse.rs` (SRP/SOC).
//! - **ARCH-330-05:** `compute_half_width` promoted to production API.
//! - **SPARSE-329-01:** Full joint normalization in sparse path â€” `inv_sum_f`
//!   stored per-sample in `SparseWFixedT` alongside fixed entries; sparse path
//!   computes `inv_norm = inv_sum_f Ã— inv_sum_m`, matching the direct path.
//! - **PERF-329-02:** FMA-idiomatic inner loop â€” `hist[idx] += w_f * w_m * inv_norm`
//!   is a canonical FMA pattern that LLVM auto-fuses into `vfmadd231ps`.
//!
//! ## Directâ†”sparse parity (SPARSE-329-01)
//!
//! Both paths now apply the same full joint normalization:
//! - Direct: `inv_norm = inv_sum_f() Ã— inv_sum_m()` from `SampleWindow`
//! - Sparse: `inv_norm = sparse_cache.inv_sum_f Ã— inv_sum_m` from
//!   `SparseWFixedT` + `SampleWindow::new_moving_only`
//!
//! This eliminates the Sprint 328 asymmetry where the sparse path only
//! normalized by `1/sum_m`, producing histograms scaled by `sum_f â‰ˆ âˆš(2Ï€ÏƒÂ²)`.
//!
//! ## ÏƒÂ²-invariance property
//!
//! Per-sample contribution to histogram total is `w_f Â· w_m Â· 1/(sum_f Â· sum_m) = 1.0`
//! for interior samples (no boundary truncation). Boundary-truncated samples
//! contribute slightly less because `sum_f Ã— sum_m` is smaller for clipped
//! windows. The histogram total equals the number of in-bounds samples,
//! regardless of ÏƒÂ². This eliminates a previously-implicit scale factor
//! `n Ã— 2Ï€` that the loss function and gradient had to compensate for.
//!
//! # Prior phases (S327â€“328)
//!
//! - **S328:** Per-sample normalization (PERF-328-01), `inv_sum_f`/`inv_sum_m` on
//!   `SampleWindow`, `compute_weights_with_inv_sum()` API, sparse-path moving-axis
//!   normalization only (partial â€” no `inv_sum_f` in sparse cache).
//! - **S326-327:** `SparseWFixedEntry.bin` u16, `extract_oob_mask()` DRY, hoisted
//!   offsets, dead `total` removal, `validate_inputs()` SSOT.
//! - **S325:** `StackWeights.len` u8, `BinRange::new` assert, `merge_histograms` idiomatic.
//! - **S324:** `BinRange` u16, `accumulate_sample_sparse` monomorphized, `merge_histograms` extracted.
//! - **S319â€“323:** `ParzenConfig` SSOT, exp-ratchet, pool, `SampleWindow`, `StackWeightsIter`.
//! - **S319â€“320:** `ParzenConfig` SRP, exp-ratchet FMA chain, pool checkout.
//!
//! # Inner-loop optimizations
//!
//! - **Exp-ratchet (PERF-319-04):** FMA chain for adjacent exponents; 1 exp() per axis.
//! - **Hoisted moving exp() (OPT-2):** Pre-computed `StackWeights`.
//! - **Stack weights (OPT-5):** `[f32; 32]` SIMD-aligned, `Copy`, no heap. Ïƒ â‰¤ ~5.2 bins.
//! - **Precomputed bin ranges (MEM-316-01):** `SampleWindow` computes once.
//! - **Lock-free checkout (PERF-319-05):** Mutex dropped before zero-fill.
//! - **Parallel reduction (OPT-6):** Moirai fold/reduce with thread-local histograms.
//! - **Histogram pool (ARCH-315-03):** Reusable buffers, Mutex poison recovery.
//! - **Monomorphized fold (PERF-315-02):** `accumulate_sample` factors out common body.
//! - **Branch-eliminated (FIX-316-07):** OOB folded into `Option`-returning constructors.
//!
//! # Safety
//!
//! **No `unsafe` code.** Parallelism via Moirai safe abstractions. `HistogramPool`
//! uses `Mutex` with poison recovery so panics don't propagate.
//!
//! `StackWeights`: zero-filled `[f32; 32]` â€” entries beyond `len` are `0.0`,
//! never uninit. `StackWeightsIter` uses safe `[]` indexing, always in-bounds.
//!
//! `BinRange` u16 (MEM-324-04): `as usize` lossless; `BinRange::new` asserts
//! `num_bins â‰¤ u16::MAX` (MEM-325-02). `StackWeights.len` u8 (MEM-325-01):
//! `as usize` lossless; max active = 31 < `u8::MAX`.
//!
//! PERF-328-01 / SPARSE-329-01: `inv_sum_f`/`inv_sum_m` = `1/sum_weights` where
//! sum > 0 for in-bounds samples (â‰¥1 bin with Gaussian weight > 0). Division
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

// â”€â”€ Sub-modules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub(crate) mod accumulate;
pub(crate) mod compute_direct;
pub(crate) mod compute_sparse;
pub(crate) mod pool;
pub(crate) mod sample;
pub(crate) mod types;

// â”€â”€ Re-exports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// Public API functions (backward-compatible paths from `direct::`).
pub use compute_direct::{compute_joint_histogram_direct, compute_joint_histogram_values};
pub use compute_sparse::{
    build_sparse_w_fixed_transposed, compute_joint_histogram_from_cache_sparse };

// Re-export accumulate helpers so test files with `use super::*;` can find them.
#[cfg(test)]
pub(crate) use accumulate::{
    accumulate_sample_direct, accumulate_sample_sparse, merge_histograms, validate_inputs };

// Re-export for sparse.rs delegation when direct-parzen is enabled.
#[cfg(feature = "direct-parzen")]
pub use pool::HistogramPool;
#[cfg(test)]
pub(crate) use sample::SampleWindow;
#[allow(unused_imports)]
pub use sample::{SparseSampleCache, SparseWFixedEntry, SparseWFixedT};

// Re-export CompactionSizes and compaction_sizes from types module.
pub use types::{compaction_sizes, CompactionSizes};
// Test-only re-exports so `use super::*;` in test files can access these.
#[cfg(test)]
#[allow(unused_imports)] // Re-exported for downstream; not used in mod.rs itself
pub(crate) use types::{BinRange, StackWeights};

// Re-exported for downstream (sparse.rs, tests); not used in mod.rs itself.
#[allow(unused_imports)]
pub(crate) use types::compute_half_width;
pub(crate) use types::ParzenConfig;

/// Normalize intensities into the clamped histogram-bin coordinate range.
///
/// # Panics
///
/// Panics when fewer than two bins are requested or the intensity range is
/// empty or reversed.
#[must_use]
pub fn normalize_intensities(
    values: &[f32],
    min_intensity: f32,
    max_intensity: f32,
    num_bins: usize,
) -> Vec<f32> {
    assert!(num_bins > 1, "normalization requires at least two bins");
    assert!(
        max_intensity > min_intensity,
        "normalization requires max_intensity > min_intensity"
    );
    let upper = (num_bins - 1) as f32;
    let scale = upper / (max_intensity - min_intensity);
    values
        .iter()
        .map(|&value| ((value - min_intensity) * scale).clamp(0.0, upper))
        .collect()
}

// â”€â”€ Test registrations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
mod direct_phase_eight_tests;
#[cfg(test)]
mod direct_phase_eleven_tests;
#[cfg(test)]
mod direct_phase_fifteen_tests;
#[cfg(test)]
mod direct_phase_fourteen_tests;
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
