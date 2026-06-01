//! Sample window and sparse entry types for the direct Parzen path.
//!
//! Extracted from `types.rs` to keep that file under the 500-line
//! structural limit. Contains [`SampleWindow`] (per-sample context)
//! and [`SparseWFixedEntry`] / [`SparseWFixedT`] (sparse cache types).

use super::types::{BinRange, ParzenConfig, StackWeights};

// ── SampleWindow ───────────────────────────────────────────────────────────

/// Precomputed bin-range, weights, and context for a single sample's
/// contribution to the joint histogram (MEM-316-01, ARCH-317-01).
///
/// For the **direct path** (`SampleWindow::new`), both fixed and moving
/// weights are pre-computed as `StackWeights`, making `accumulate_sample`
/// entirely heap-free per sample — no `SparseWFixedEntry` construction.
///
/// For the **sparse-cache path** (`SampleWindow::new_moving_only`), only
/// the moving weights are pre-computed (the fixed weights come from the
/// cached `SparseWFixedT`). The returned tuple includes `inv_sum_m`
/// (the moving-axis normalization factor). The fixed-axis normalization
/// factor `inv_sum_f` is stored in the sparse cache alongside the fixed
/// entries (SPARSE-329-01), and combined into `inv_norm = inv_sum_f × inv_sum_m`
/// at the accumulation site.
///
/// When an OOB mask indicates the sample is out-of-bounds, the constructor
/// returns `None`, eliminating the `if mask_val >= 0.5` branch from the
/// fold closures (FIX-316-07).
///
/// # Encapsulation (ARCH-323-01)
///
/// Bin-range fields (`f_range`, `m_range`) are private. Production code
/// in `accumulate_sample_direct` uses the `f_range()` / `m_range()` accessors.
/// Weight fields (`f_weights`, `m_weights`) remain `pub(crate)` since they
/// are passed by reference in the hot loop and the iterator is already
/// the controlled access path.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SampleWindow {
    /// Fixed-image bin range.
    f_range: BinRange,
    /// Moving-image bin range.
    m_range: BinRange,
    /// Normalized fixed-image value (test-only — not needed in production hot-loop).
    #[cfg(test)]
    pub f_val: f32,
    /// Normalized moving-image value (test-only — not needed in production hot-loop).
    #[cfg(test)]
    pub m_val: f32,
    /// Pre-computed fixed Parzen weights (direct path).
    pub(crate) f_weights: StackWeights,
    /// Pre-computed moving Parzen weights.
    pub(crate) m_weights: StackWeights,
    /// Per-sample normalization: `1.0 / sum_f` (reciprocal of the fixed-axis
    /// Parzen weight total for this sample). Pre-computed in `new()` so the
    /// hot loop applies the normalization with a single multiplication.
    /// PERF-328-01: makes the per-sample contribution to the histogram
    /// `sum_f * inv_sum_f * sum_m * inv_sum_m = 1.0`, so the histogram
    /// total equals the number of in-bounds samples regardless of sigma^2.
    pub(crate) inv_sum_f: f32,
    /// Per-sample normalization: `1.0 / sum_m` (reciprocal of the moving-axis
    /// Parzen weight total for this sample). See `inv_sum_f` for rationale.
    pub(crate) inv_sum_m: f32,
}

impl SampleWindow {
    /// Return the fixed-image bin range.
    #[inline]
    pub fn f_range(&self) -> &BinRange {
        &self.f_range
    }

    /// Return the moving-image bin range.
    #[inline]
    pub fn m_range(&self) -> &BinRange {
        &self.m_range
    }

    /// Return `1 / sum_f` for this sample (PERF-328-01).
    #[inline]
    pub fn inv_sum_f(&self) -> f32 {
        self.inv_sum_f
    }

    /// Return `1 / sum_m` for this sample (PERF-328-01).
    #[inline]
    pub fn inv_sum_m(&self) -> f32 {
        self.inv_sum_m
    }

    /// Compute the `SampleWindow` for sample `i` (direct path — both axes).
    ///
    /// Returns `None` if the OOB mask indicates this sample should be
    /// excluded (`mask_val < 0.5`).
    ///
    /// Both fixed and moving weights are pre-computed as `StackWeights`,
    /// so `accumulate_sample` needs no heap allocation or `SparseWFixedEntry`
    /// construction per sample.
    ///
    /// # Arguments
    /// * `i` — Sample index
    /// * `fixed_norm` — Normalized fixed-image values
    /// * `moving_norm` — Normalized moving-image values
    /// * `num_bins` — Number of histogram bins per axis
    /// * `fix_cfg` — Fixed-axis Parzen configuration
    /// * `mov_cfg` — Moving-axis Parzen configuration
    /// * `oob_mask` — Optional OOB mask (1.0 = in-bounds, 0.0 = excluded)
    #[inline]
    pub fn new(
        i: usize,
        fixed_norm: &[f32],
        moving_norm: &[f32],
        num_bins: usize,
        fix_cfg: &ParzenConfig,
        mov_cfg: &ParzenConfig,
        oob_mask: Option<&[f32]>,
    ) -> Option<Self> {
        let _mask_val = Self::mask_val(i, oob_mask)?;
        let f_val = fixed_norm[i];
        let m_val = moving_norm[i];
        // PERF-328-01: compute weights and inv_sum in one pass each.
        let (f_range, f_weights, inv_sum_f) = fix_cfg.compute_weights_with_inv_sum(f_val, num_bins);
        let (m_range, m_weights, inv_sum_m) = mov_cfg.compute_weights_with_inv_sum(m_val, num_bins);
        Some(SampleWindow {
            f_range,
            m_range,
            #[cfg(test)]
            f_val,
            #[cfg(test)]
            m_val,
            f_weights,
            m_weights,
            inv_sum_f,
            inv_sum_m,
        })
    }

    /// Compute the moving-only window for the sparse-cache path.
    ///
    /// Returns `None` if the OOB mask indicates this sample should be
    /// excluded. The returned tuple contains the moving value, bin range, pre-computed
    /// moving weights, and `inv_sum_m` (moving-axis normalization factor).
    /// The fixed weights come from the sparse cache; `inv_sum_f` is stored
    /// in the cache alongside the fixed entries (SPARSE-329-01). The caller
    /// combines `inv_norm = inv_sum_f × inv_sum_m` before passing to
    /// `accumulate_sample_sparse`.
    #[inline]
    pub fn new_moving_only(
        i: usize,
        moving_norm: &[f32],
        num_bins: usize,
        mov_cfg: &ParzenConfig,
        oob_mask: Option<&[f32]>,
    ) -> Option<(f32, BinRange, StackWeights, f32)> {
        let _mask_val = Self::mask_val(i, oob_mask)?;
        let m_val = moving_norm[i];
        // PERF-328-02: compute weights and inv_sum in one pass.
        let (m_range, m_weights, inv_sum_m) = mov_cfg.compute_weights_with_inv_sum(m_val, num_bins);
        Some((m_val, m_range, m_weights, inv_sum_m))
    }

    // ── Private helpers (DRY) ──────────────────────────────────────────
    //

    /// Return the OOB mask value, or `None` if the sample is excluded.
    ///
    /// This is the shared inner logic that was duplicated between `new`
    /// and `new_moving_only` — each had its own `match oob_mask` / `if
    /// mask_val < 0.5` block. Now both call this single helper.
    #[inline]
    pub(crate) fn mask_val(i: usize, oob_mask: Option<&[f32]>) -> Option<f32> {
        let val = match oob_mask {
            Some(m) => m[i],
            None => 1.0,
        };
        if val < 0.5 {
            None
        } else {
            Some(val)
        }
    }
}

// ── SparseWFixedEntry ──────────────────────────────────────────────────────

/// A single `(bin_index, weight)` entry in a sparse Parzen weight row.
///
/// Newtype wrapper around `(u16, f32)` that prevents accidental index/weight
/// swaps — a subtle bug when working with bare tuples where both types are
/// numeric. Provides named field access (`entry.bin`, `entry.weight`) and
/// `Copy` semantics for zero-cost passing in the hot loop.
///
/// The `bin` field is `u16` (PERF-326-02): Parzen histograms never exceed
/// 65535 bins (practical limit is ~256), so 2 bytes suffice. This compacts
/// `SparseWFixedEntry` from 16 to 8 bytes (2+2 padding + 4 f32), halving
/// the sparse cache memory footprint (~3.5 KB → ~1.75 KB for 32K samples
/// with 7 entries each).
///
/// Used by the **sparse-cache path** only; the direct path uses `StackWeights`
/// for both axes and never constructs `SparseWFixedEntry` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseWFixedEntry {
    /// Histogram bin index on the fixed-image axis (`u16` since num_bins ≤ 65535).
    pub bin: u16,
    /// Gaussian Parzen weight for this bin.
    pub weight: f32,
}

impl SparseWFixedEntry {
    /// Construct a new sparse entry.
    #[inline]
    pub fn new(bin: u16, weight: f32) -> Self {
        Self { bin, weight }
    }
}

/// Sparse representation of W_fixed^T, with per-sample normalization.
///
/// Each element is a `(Vec<SparseWFixedEntry>, inv_sum_f)` pair:
/// - The `Vec<SparseWFixedEntry>` contains the non-zero fixed-axis bin entries
///   within ±3σ (typically ~7 for σ ≈ 1 bin-width).
/// - `inv_sum_f` is `1/sum_f` for this sample (SPARSE-329-01), enabling full
///   joint normalization `inv_norm = inv_sum_f × inv_sum_m` in the sparse path,
///   matching the direct path (PERF-328-01).
///
/// Used by `compute_joint_histogram_from_cache_sparse`; the direct path
/// avoids this type entirely by pre-computing both axes' weights as
/// `StackWeights` inside `SampleWindow`.
///
/// # Memory layout
///
/// Per sample: `Vec<SparseWFixedEntry>` (~56 bytes for 7 entries × 8 bytes)
/// + `f32` inv_sum_f (4 bytes). Total ≈ 60 bytes/sample. For 32K samples:
/// ~1.875 MB, of which inv_sum_f adds only 128 KB overhead.
pub type SparseWFixedT = Vec<(Vec<SparseWFixedEntry>, f32)>;
