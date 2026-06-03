//! Sparse fixed-weight entry and type alias for the sparse-cache Parzen path.

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
///
/// ~1.875 MB, of which inv_sum_f adds only 128 KB overhead.
pub type SparseWFixedT = Vec<(Vec<SparseWFixedEntry>, f32)>;
