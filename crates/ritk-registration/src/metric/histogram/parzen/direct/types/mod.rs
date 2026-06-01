//! Shared types for the direct Parzen histogram computation path.
//!
//! Decomposed into vertical hierarchy (ARCH-330-01):
//! - `half_width` — ±3σ half-width computation and constants
//! - `stack_weights` — Stack-allocated weight array and iterator
//! - `bin_range` — Clamped bin range type
//! - `parzen_config` — Precomputed Parzen-window parameters
//!
//! # Design principles
//!
//! - **SSOT**: `compute_half_width` is the sole ±3σ definition; `ParzenConfig`
//!   is the sole per-axis σ holder and precomputed derivatives.
//! - **SRP**: `ParzenConfig` owns normalisation; `SampleWindow` owns
//!   per-sample bin computation. Each type in its own module.
//! - **DRY**: `SampleWindow::new` / `new_moving_only` share an OOB-filter helper.
//! - **Zero-cost**: `SampleWindow` carries `StackWeights` for both axes — no
//!   heap allocation per sample. Sparse-cache path uses `SparseWFixedEntry`.

pub(crate) mod bin_range;
pub(crate) mod half_width;
pub(crate) mod parzen_config;
pub(crate) mod stack_weights;

// ── Re-exports for backward compatibility ──────────────────────────────────

pub(crate) use bin_range::BinRange;
pub(crate) use half_width::compute_half_width;
#[cfg(test)]
pub(crate) use half_width::MAX_PARZEN_BINS;
// ARCH-330-05: promoted to production; used by compute_half_width internally
// and by sparse.rs test code. Not directly used in production by other modules.
#[allow(unused_imports)]
pub(crate) use half_width::MIN_HALF_WIDTH;
pub(crate) use parzen_config::ParzenConfig;
pub(crate) use stack_weights::StackWeights;
#[cfg(test)]
pub(crate) use stack_weights::STACK_WEIGHTS_CAPACITY;

// ── CompactionSizes ────────────────────────────────────────────────────────

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
    /// `ParzenConfig` — 24 bytes with `usize` half_width (u16 attempt reverted; see parzen_config.rs).
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
        stack_weights: std::mem::size_of::<StackWeights>(),
        bin_range: std::mem::size_of::<BinRange>(),
        parzen_config: std::mem::size_of::<ParzenConfig>(),
        sample_window: std::mem::size_of::<super::sample::SampleWindow>(),
        sparse_fixed_entry: std::mem::size_of::<super::sample::SparseWFixedEntry>(),
    }
}
