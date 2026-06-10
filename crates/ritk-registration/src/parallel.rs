//! Shared helpers for safe parallel writes to disjoint slice regions.

/// Send+Sync wrapper for a mutable slice pointer, enabling safe parallel
/// access to disjoint sub-slices via offset arithmetic.
///
/// The pointer and length are stored; callers must construct disjoint
/// `&mut [f32]` slices from non-overlapping offset ranges, which
/// z-slice parallelism guarantees by construction.
pub(crate) struct CellSlice {
    ptr: *mut f32,
    len: usize,
}

impl CellSlice {
    pub(crate) fn from_mut(s: &mut [f32]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
        }
    }

    /// Reconstruct a mutable slice at `offset` with length `chunk_len`.
    ///
    /// # Safety
    /// Caller must ensure `[offset, offset + chunk_len)` is within `[0, len)`
    /// and no other reference to the same memory range exists.
    ///
    /// The `&self → &mut [f32]` pattern is sound here because the caller
    /// partitions `[0, len)` into disjoint offset ranges before passing them
    /// to parallel threads, so no two `&mut` references alias the same memory.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slice_mut(&self, offset: usize, chunk_len: usize) -> &mut [f32] {
        debug_assert!(offset + chunk_len <= self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(offset), chunk_len)
    }
}

// SAFETY: CellSlice is only used to create disjoint mutable slices within
// parallel z-slice iteration, which guarantees non-overlapping regions per
// closure invocation. No two threads ever write to the same memory via this
// wrapper.
unsafe impl Send for CellSlice {}
unsafe impl Sync for CellSlice {}
