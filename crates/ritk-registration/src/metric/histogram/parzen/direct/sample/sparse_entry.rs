//! Sparse fixed-weight entry and type alias for the sparse-cache Parzen path.

// â”€â”€ SparseWFixedEntry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A single `(bin_index, weight)` entry in a sparse Parzen weight row.
///
/// Newtype wrapper around `(u16, f32)` that prevents accidental index/weight
/// swaps â€” a subtle bug when working with bare tuples where both types are
/// numeric. Provides named field access (`entry.bin`, `entry.weight`) and
/// `Copy` semantics for zero-cost passing in the hot loop.
///
/// The `bin` field is `u16` (PERF-326-02): Parzen histograms never exceed
/// 65535 bins (practical limit is ~256), so 2 bytes suffice. This compacts
/// `SparseWFixedEntry` from 16 to 8 bytes (2+2 padding + 4 f32), halving
/// the sparse cache memory footprint (~3.5 KB â†’ ~1.75 KB for 32K samples
/// with 7 entries each).
///
/// Used by the **sparse-cache path** only; the direct path uses `StackWeights`
/// for both axes and never constructs `SparseWFixedEntry` values.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SparseWFixedEntry {
    /// Histogram bin index on the fixed-image axis (`u16` since num_bins â‰¤ 65535).
    pub bin: u16,
    /// Gaussian Parzen weight for this bin.
    pub weight: f32 }

impl SparseWFixedEntry {
    /// Construct a new sparse entry.
    #[inline]
    pub fn new(bin: u16, weight: f32) -> Self {
        Self { bin, weight }
    }
}

// â”€â”€ SparseSampleCache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A stack-allocated collection of sparse fixed-image entries.
///
/// Avoids heap allocating a `Vec` for each voxel sample (OPT-5/MEM-330-x).
/// Uses a fixed-capacity array of 32 `SparseWFixedEntry` and a `u8` length tracker,
/// collapsing the per-sample cache into a single contiguous allocation within `SparseWFixedT`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseSampleCache {
    entries: [SparseWFixedEntry; 32],
    /// Number of active entries. `u8` is sufficient for max capacity 32.
    len: u8 }

impl Default for SparseSampleCache {
    #[inline]
    fn default() -> Self {
        Self {
            entries: [SparseWFixedEntry::default(); 32],
            len: 0 }
    }
}

impl SparseSampleCache {
    /// Construct a new empty sparse sample cache.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an entry onto the stack cache.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if capacity is exceeded.
    #[inline]
    pub fn push(&mut self, entry: SparseWFixedEntry) {
        let index = self.len as usize;
        debug_assert!(index < 32, "SparseSampleCache capacity exceeded");
        if index < 32 {
            self.entries[index] = entry;
            self.len += 1;
        }
    }

    /// Get a slice containing the active entries.
    #[inline]
    pub fn as_slice(&self) -> &[SparseWFixedEntry] {
        &self.entries[..self.len as usize]
    }

    /// Get a mutable slice containing the active entries.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [SparseWFixedEntry] {
        &mut self.entries[..self.len as usize]
    }

    /// Iterate over active entries.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, SparseWFixedEntry> {
        self.as_slice().iter()
    }

    /// Returns the number of active entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns true if there are no active entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl std::ops::Deref for SparseSampleCache {
    type Target = [SparseWFixedEntry];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::ops::DerefMut for SparseSampleCache {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

/// Sparse representation of W_fixed^T, with per-sample normalization.
///
/// Each element is a `(SparseSampleCache, inv_sum_f)` pair:
/// - The `SparseSampleCache` contains the non-zero fixed-axis bin entries
///   within Â±3Ïƒ (typically ~7 for Ïƒ â‰ˆ 1 bin-width).
/// - `inv_sum_f` is `1/sum_f` for this sample (SPARSE-329-01), enabling full
///   joint normalization `inv_norm = inv_sum_f Ã— inv_sum_m` in the sparse path,
///   matching the direct path (PERF-328-01).
///
/// Used by `compute_joint_histogram_from_cache_sparse`; the direct path
/// avoids this type entirely by pre-computing both axes' weights as
/// `StackWeights` inside `SampleWindow`.
///
/// # Memory layout
///
/// Per sample: `SparseSampleCache` (260 bytes) + `f32` inv_sum_f (4 bytes).
/// Total 264 bytes/sample. For 32K samples:
///
/// ~8.25 MB, allocated contiguously in a single heap vector allocation,
/// dropping allocations from 32K+ individual heap arrays to exactly 1.
pub type SparseWFixedT = Vec<(SparseSampleCache, f32)>;
