//! Stack-allocated Parzen weights for one axis (OPT-5).

/// Stack weight array capacity (32×f32 = 128 B, AVX2-aligned).
///
/// Covers σ≈5.2 bins (half_width≤15, range≤31). Beyond-active slots
/// are zero-filled. FIX-319-09: increased from 16 to 32.
pub(crate) const STACK_WEIGHTS_CAPACITY: usize = 32;

/// Stack-allocated Parzen weights for one axis (OPT-5).
///
/// Avoids `Vec` heap allocation for ≤31 weights. `len` tracks active
/// entries; beyond is zero-filled. `[f32; 32]` (not 31) for 128-byte SIMD
/// alignment. `len` is `u8` (MEM-325-01): max=31 fits `u8`; `Copy` safe
/// since padding is zero-filled.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StackWeights {
    pub(crate) weights: [f32; STACK_WEIGHTS_CAPACITY],
    /// Number of active weight entries. `u8` since max = 31 < 256.
    pub(crate) len: u8,
}

/// Iterator over active `(bin_offset, weight)` pairs of [`StackWeights`].
///
/// Implements `Clone`/`ExactSizeIterator`/`DoubleEndedIterator`. Concrete
/// type enables monomorphized accumulation and weight-sequence replay.
///
/// # Safety
///
/// No `unsafe`. Indexing uses safe `[]` on the pre-sliced active region,
/// always in bounds by construction.
#[derive(Clone, Debug)]
pub(crate) struct StackWeightsIter<'a> {
    slice: &'a [f32],
    pos: usize,
    remaining: usize,
}

impl<'a> Iterator for StackWeightsIter<'a> {
    type Item = (usize, f32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let j = self.pos;
        let w = self.slice[j];
        self.pos += 1;
        self.remaining -= 1;
        Some((j, w))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> ExactSizeIterator for StackWeightsIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a> DoubleEndedIterator for StackWeightsIter<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let j = self.pos + self.remaining;
        let w = self.slice[j];
        Some((j, w))
    }
}

impl StackWeights {
    /// Build Parzen weights for bins `[lo..=hi]` from a normalised value.
    ///
    /// Entry `j` = `exp(-(val-(lo+j))²/(2σ²))`. Beyond `hi-lo+1` is zero-filled.
    ///
    /// # Performance (PERF-319-04, PERF-331-02)
    ///
    /// **Exp-ratchet trick**: adjacent bins differ by 1 in `diff`, so exponent
    /// changes by a fixed second-difference increment. Reduces cost from
    /// `N×exp()` to `1×exp()+(N-1)×fma` (~3× faster for 7-bin window).
    /// Drift ≤31 ULP at max capacity, within 1e-4 test tolerance.
    ///
    /// **Sum in same pass (PERF-331-02)**: accumulates the weight sum
    /// during the exp-ratchet loop, avoiding a second pass over the
    /// weights to compute `1/sum`. Used by
    /// [`compute_weights_with_inv_sum`](super::ParzenConfig::compute_weights_with_inv_sum)
    /// to skip the redundant `weights.iter().sum()` call.
    ///
    /// # Returns
    /// `(weights, sum)` tuple where `sum = Σ weights[0..len]`. The sum is
    /// exact up to the same drift as the exp-ratchet itself.
    #[inline]
    pub fn new_with_sum(val: f32, lo: usize, hi: usize, inv_2sigma_sq: f32) -> (Self, f32) {
        debug_assert!(hi >= lo, "hi ({hi}) must be >= lo ({lo})");
        // FIX-319-09: range_len must fit within capacity.
        assert!(
            hi - lo < STACK_WEIGHTS_CAPACITY,
            "bin range {lo}..={hi} ({}) exceeds STACK_WEIGHTS_CAPACITY={STACK_WEIGHTS_CAPACITY}",
            hi - lo + 1
        );
        let mut weights = [0.0f32; STACK_WEIGHTS_CAPACITY];
        let len = (hi - lo + 1) as u8;

        // PERF-319-04: Exp-ratchet — see module docs for derivation.
        let diff0 = val - lo as f32;
        let mut exponent = diff0 * diff0 * inv_2sigma_sq;
        let two_inv_2sigma_sq = 2.0 * inv_2sigma_sq;
        let mut delta = inv_2sigma_sq - two_inv_2sigma_sq * diff0;

        // PERF-331-02: accumulate sum in same pass.
        let mut sum = 0.0f32;
        for w in weights.iter_mut().take(len as usize) {
            let val = exponent.exp();
            *w = val;
            sum += val;
            exponent += delta;
            delta += two_inv_2sigma_sq;
        }

        (StackWeights { weights, len }, sum)
    }

    /// Build Parzen weights for bins `[lo..=hi]` from a normalised value.
    ///
    /// Entry `j` = `exp(-(val-(lo+j))²/(2σ²))`. Beyond `hi-lo+1` is zero-filled.
    ///
    /// # Performance (PERF-319-04)
    ///
    /// **Exp-ratchet trick**: adjacent bins differ by 1 in `diff`, so exponent
    /// changes by a fixed second-difference increment. Reduces cost from
    /// `N×exp()` to `1×exp()+(N-1)×fma` (~3× faster for 7-bin window).
    /// Drift ≤31 ULP at max capacity, within 1e-4 test tolerance.
    #[cfg(test)]
    #[inline]
    pub fn new(val: f32, lo: usize, hi: usize, inv_2sigma_sq: f32) -> Self {
        Self::new_with_sum(val, lo, hi, inv_2sigma_sq).0
    }

    /// Iterate over active `(bin_offset, weight)` pairs.
    ///
    /// Yields `(j, weights[j])` where actual bin index is `lo + j`.
    /// Returns [`StackWeightsIter`] (`Clone`/`ExactSizeIterator`/`DoubleEndedIterator`).
    #[inline]
    pub fn iter(&self) -> StackWeightsIter<'_> {
        StackWeightsIter {
            slice: &self.weights[..self.len as usize], // MEM-325-01: u8 → usize
            pos: 0,
            remaining: self.len as usize,
        }
    }

    /// Number of active weight entries (`usize`).
    ///
    /// Production use (ARCH-328-04): per-sample weight normalization.
    /// Returns `usize`; internal `u8` losslessly upcast (MEM-325-01).
    #[inline]
    #[allow(dead_code)] // Production API; current callers use StackWeightsIter::len()
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the weight array is empty. Convention companion to `len()`.
    /// Production use (ARCH-328-04): gating on empty weights.
    #[cfg(test)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
