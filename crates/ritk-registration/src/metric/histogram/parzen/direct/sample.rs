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
/// cached `SparseWFixedT`).
///
/// When an OOB mask indicates the sample is out-of-bounds, the constructor
/// returns `None`, eliminating the `if mask_val >= 0.5` branch from the
/// fold closures (FIX-316-07).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SampleWindow {
    /// Fixed-image bin range.
    pub f_range: BinRange,
    /// Moving-image bin range.
    pub m_range: BinRange,
    /// Normalized fixed-image value.
    #[allow(dead_code)] // used by tests and potential future callers
    pub f_val: f32,
    /// Normalized moving-image value.
    #[allow(dead_code)] // used by tests and potential future callers
    pub m_val: f32,
    /// Pre-computed fixed Parzen weights (direct path).
    pub f_weights: StackWeights,
    /// Pre-computed moving Parzen weights.
    pub m_weights: StackWeights,
}

impl SampleWindow {
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
        // ARCH-320-03: delegate to ParzenConfig::compute_weights (DRY)
        let (f_range, f_weights) = fix_cfg.compute_weights(f_val, num_bins);
        let (m_range, m_weights) = mov_cfg.compute_weights(m_val, num_bins);
        Some(SampleWindow {
            f_range,
            m_range,
            f_val,
            m_val,
            f_weights,
            m_weights,
        })
    }

    /// Compute the moving-only window for the sparse-cache path.
    ///
    /// Returns `None` if the OOB mask indicates this sample should be
    /// excluded. The returned tuple contains the moving value, bin range,
    /// and pre-computed moving weights (the fixed weights come from the
    /// sparse cache).
    #[inline]
    pub fn new_moving_only(
        i: usize,
        moving_norm: &[f32],
        num_bins: usize,
        mov_cfg: &ParzenConfig,
        oob_mask: Option<&[f32]>,
    ) -> Option<(f32, BinRange, StackWeights)> {
        let _mask_val = Self::mask_val(i, oob_mask)?;
        let m_val = moving_norm[i];
        // ARCH-320-03: delegate to ParzenConfig::compute_weights (DRY)
        let (m_range, m_weights) = mov_cfg.compute_weights(m_val, num_bins);
        Some((m_val, m_range, m_weights))
    }

    // ── Private helpers (DRY) ──────────────────────────────────────────
    //

    /// Return the OOB mask value, or `None` if the sample is excluded.
    ///
    /// This is the shared inner logic that was duplicated between `new`
    /// and `new_moving_only` — each had its own `match oob_mask` / `if
    /// mask_val < 0.5` block. Now both call this single helper.
    #[inline]
    fn mask_val(i: usize, oob_mask: Option<&[f32]>) -> Option<f32> {
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
/// Newtype wrapper around `(usize, f32)` that prevents accidental index/weight
/// swaps — a subtle bug when working with bare tuples where both types are
/// numeric. Provides named field access (`entry.bin`, `entry.weight`) and
/// `Copy` semantics for zero-cost passing in the hot loop.
///
/// Used by the **sparse-cache path** only; the direct path uses `StackWeights`
/// for both axes and never constructs `SparseWFixedEntry` values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseWFixedEntry {
    /// Histogram bin index on the fixed-image axis.
    pub bin: usize,
    /// Gaussian Parzen weight for this bin.
    pub weight: f32,
}

impl SparseWFixedEntry {
    /// Construct a new sparse entry.
    #[inline]
    pub fn new(bin: usize, weight: f32) -> Self {
        Self { bin, weight }
    }
}

/// Sparse representation of W_fixed^T.
///
/// Each inner `Vec` corresponds to one sample and contains `SparseWFixedEntry`
/// values for the non-zero bins within ±3σ. Typically ~7 entries per sample
/// (for σ ≈ 1 bin-width with the minimum half-width of 3).
///
/// Used by `compute_joint_histogram_from_cache_sparse`; the direct path
/// avoids this type entirely by pre-computing both axes' weights as
/// `StackWeights` inside `SampleWindow`.
pub type SparseWFixedT = Vec<Vec<SparseWFixedEntry>>;
