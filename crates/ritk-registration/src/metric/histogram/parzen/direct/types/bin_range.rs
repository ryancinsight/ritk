//! Clamped bin range for one axis of a single sample (ARCH-316-04).

/// Clamped bin range `[lo, hi]` for one axis of a single sample (ARCH-316-04).
///
/// Newtype over `(u16, u16)` — prevents `(hi, lo)` swaps. `u16` since Parzen
/// histograms never exceed ~256 bins in practice, reducing size from 16→4 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BinRange {
    /// Lower bound of the support window (inclusive), clamped to `≥ 0`.
    pub(crate) lo: u16,
    /// Upper bound of the support window (inclusive), clamped to `< num_bins`.
    pub(crate) hi: u16,
}

impl BinRange {
    /// Construct from primary bin, half-width, and bin count.
    ///
    /// Clamps to `[0, num_bins-1]`. If primary is entirely out of range,
    /// collapses to a single boundary bin.
    #[inline]
    pub fn new(primary: i32, half_width: usize, num_bins: usize) -> Self {
        // MEM-325-02: Guard against silent u16 truncation.
        // Parzen histograms never exceed ~256 bins in practice,
        // but a misconfigured caller could pass num_bins > 65535.
        assert!(
            num_bins <= u16::MAX as usize,
            "num_bins={num_bins} exceeds u16::MAX={u16_max}, would truncate BinRange fields",
            u16_max = u16::MAX
        );
        let lo = (primary - half_width as i32).max(0) as usize as u16;
        let hi = ((primary + half_width as i32).min(num_bins as i32 - 1)).max(0) as usize as u16;
        // If primary is entirely beyond num_bins (e.g. primary=22, num_bins=16),
        // lo would exceed hi after clamping. Clamp lo down to hi so the range
        // covers at least the boundary bin.
        let lo = lo.min(hi);
        debug_assert!(lo <= hi, "BinRange lo={lo} > hi={hi}");
        BinRange { lo, hi }
    }

    /// Number of bins in the range (inclusive).
    ///
    /// Production use (ARCH-328-05): weight normalization and range validation.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        (self.hi - self.lo + 1) as usize
    }

    /// Returns `true` when `lo > hi`, meaning no bins fall in range.
    ///
    /// Test-only: production code checks `len() > 0` or iterates directly.
    #[cfg(test)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi
    }

    /// Iterate over bin indices in this range.
    ///
    /// Test-only: production code uses `StackWeights::iter()` offsets.
    #[cfg(test)]
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = usize> + use<'_> {
        self.lo as usize..=self.hi as usize
    }
}
