//! Shared types for the direct Parzen histogram computation path.
//!
//! Factored out of `direct/mod.rs` to keep it under the 500-line limit.
//!
//! # Design principles
//!
//! - **SSOT**: `compute_half_width` is the sole ±3σ definition; `ParzenConfig`
//!   is the sole per-axis σ holder and precomputed derivatives.
//! - **SRP**: `ParzenConfig` owns normalisation; `SampleWindow` owns
//!   per-sample bin computation.
//! - **DRY**: `SampleWindow::new` / `new_moving_only` share an OOB-filter helper.
//! - **Zero-cost**: `SampleWindow` carries `StackWeights` for both axes — no
//!   heap allocation per sample. Sparse-cache path uses `SparseWFixedEntry`.

/// Maximum Parzen bins a single sample can touch on one axis.
///
/// ±3σ with σ≈5.2 bins → half_width=15 → range≤31. Rounded to 32 for
/// 128-byte SIMD alignment (32×f32 = four AVX2 `__m256` registers).
#[cfg(test)]
pub(crate) const MAX_PARZEN_BINS: usize = 31;

/// Stack weight array capacity (32×f32 = 128 B, AVX2-aligned).
///
/// Covers σ≈5.2 bins (half_width≤15, range≤31). Beyond-active slots
/// are zero-filled. FIX-319-09: increased from 16 to 32.
pub(crate) const STACK_WEIGHTS_CAPACITY: usize = 32;

/// Minimum support half-width. Ensures ≥3 bins per side even for σ<1 bin
/// (B-spline-like continuity).
pub(crate) const MIN_HALF_WIDTH: usize = 3;

// ── Half-width computation (SSOT) ─────────────────────────────────────────

/// Support half-width from sigma² via the ±3σ rule.
///
/// Returns `ceil(3*sqrt(sigma_sq)).max(MIN_HALF_WIDTH)` — captures >99.7%
/// of Gaussian mass. SSOT: `sparse.rs` duplicate is `#[cfg(test)]`-only
/// and delegates here.
#[inline]
pub fn compute_half_width(sigma_sq: f32) -> usize {
    let sigma = sigma_sq.sqrt();
    let computed = (3.0 * sigma).ceil() as usize;
    computed.max(MIN_HALF_WIDTH)
}

// ── StackWeights ───────────────────────────────────────────────────────────

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
    /// [`compute_weights_with_inv_sum`](ParzenConfig::compute_weights_with_inv_sum)
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
    #[inline]
    #[allow(dead_code)] // Convention companion; no current production caller
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ── BinRange ───────────────────────────────────────────────────────────────

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
    #[allow(dead_code)] // Production API; current callers use BinRange fields directly
    pub fn len(&self) -> usize {
        (self.hi - self.lo + 1) as usize
    }

    /// Whether the range is empty (should never be true after `new()`).
    /// Production use (ARCH-328-05).
    #[inline]
    #[allow(dead_code)] // Convention companion; no current production caller
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

// ── ParzenConfig ───────────────────────────────────────────────────────────

/// Precomputed Parzen-window parameters for one axis (SRP/SSOT).
///
/// Groups σ² and derived `half_width`/`inv_2sigma_sq`. `SampleWindow` takes
/// `&ParzenConfig` instead of raw parameters.
///
/// # Invariants
///
/// - `half_width >= MIN_HALF_WIDTH` (via `compute_half_width`).
/// - `inv_2sigma_sq == -0.5 / sigma_sq`.
///
/// # Encapsulation (ARCH-322-03)
///
/// All fields private. Construct via [`new`](Self::new) or
/// [`from_intensity_sigma`](Self::from_intensity_sigma). Access via
/// [`sigma_sq()`](Self::sigma_sq) (production), [`half_width()`](Self::half_width)
/// / [`inv_2sigma_sq()`](Self::inv_2sigma_sq) (test-only).
///
/// # Memory layout
///
/// `half_width` is `usize` (reverted from MEM-328-03 `u16`): `from_intensity_sigma`
/// can produce σ²>10⁹ (near-equal range), yielding `half_width`>65535.
/// `u16` truncation would corrupt bin-range. Saving is negligible since
/// `ParzenConfig` is instantiated once per histogram.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParzenConfig {
    /// Parzen σ² in bin-index units.
    sigma_sq: f32,
    /// Half-width in bins (`usize`, reverted MEM-328-03 — see struct doc).
    half_width: usize,
    /// Precomputed `-0.5 / sigma_sq`.
    inv_2sigma_sq: f32,
}

impl ParzenConfig {
    /// Return σ² in bin-index units.
    ///
    /// Raw parameter from which `half_width`/`inv_2sigma_sq` are derived.
    #[inline]
    pub fn sigma_sq(&self) -> f32 {
        self.sigma_sq
    }

    /// Return the support half-width (always `>= MIN_HALF_WIDTH`).
    #[cfg(test)]
    #[inline]
    pub fn half_width(&self) -> usize {
        self.half_width as usize
    }

    /// Return `-0.5 / sigma_sq`. Used by `StackWeights::new` and test code.
    #[cfg(test)]
    #[inline]
    pub fn inv_2sigma_sq(&self) -> f32 {
        self.inv_2sigma_sq
    }

    /// Construct from σ² in bin-index units.
    ///
    /// Derives `half_width` (via `compute_half_width`) and `inv_2sigma_sq`.
    #[inline]
    pub fn new(sigma_sq: f32) -> Self {
        assert!(sigma_sq > 0.0, "sigma_sq must be positive, got {sigma_sq}");
        assert!(
            sigma_sq.is_finite(),
            "sigma_sq must be finite, got {sigma_sq}"
        );
        Self {
            sigma_sq,
            half_width: compute_half_width(sigma_sq),
            inv_2sigma_sq: -0.5 / sigma_sq,
        }
    }

    /// Construct from intensity-space sigma, range, and bin count.
    ///
    /// Converts via `sigma_sq=(sigma/bin_width)²`, then delegates to [`new`](Self::new).
    /// SSOT constructor (SSOT-319-02). Panics if `sigma<=0`, non-finite,
    /// `max<=min`, or `num_bins==0`.
    #[inline]
    pub fn from_intensity_sigma(sigma: f32, min: f32, max: f32, num_bins: usize) -> Self {
        assert!(num_bins > 0, "num_bins must be > 0, got {num_bins}");
        assert!(max > min, "max must be > min, got max={max} min={min}");
        let num_bins_f = (num_bins - 1) as f32;
        let bin_width = (max - min) / num_bins_f.max(1.0);
        let sigma_in_bins = sigma / bin_width.max(f32::EPSILON);
        Self::new(sigma_in_bins * sigma_in_bins)
    }

    /// Full support window size: `2 * half_width + 1`.
    ///
    /// Must fit within `STACK_WEIGHTS_CAPACITY`.
    #[cfg(test)]
    #[inline]
    pub fn support_bins(&self) -> usize {
        2 * self.half_width + 1
    }

    /// Clamped bin range for a normalised value (ARCH-320-03).
    ///
    /// Encapsulates `floor → BinRange::new` (was inlined at multiple sites).
    ///
    /// # Arguments
    /// * `val` — Normalised intensity in `[0, num_bins-1]`
    /// * `num_bins` — Histogram bins per axis
    #[inline]
    pub fn bin_range(&self, val: f32, num_bins: usize) -> BinRange {
        let primary = val.floor() as i32;
        BinRange::new(primary, self.half_width, num_bins)
    }

    /// Compute Parzen weights for a normalised value (ARCH-320-03).
    ///
    /// Encapsulates `bin_range → StackWeights::new` (was inlined).
    ///
    /// # Arguments
    /// * `val` — Normalised intensity in `[0, num_bins-1]`
    /// * `num_bins` — Histogram bins per axis
    #[inline]
    pub fn compute_weights(&self, val: f32, num_bins: usize) -> (BinRange, StackWeights) {
        let range = self.bin_range(val, num_bins);
        let weights = StackWeights::new(
            val,
            range.lo as usize,
            range.hi as usize,
            self.inv_2sigma_sq,
        );
        (range, weights)
    }

    /// Compute weights, range, and `1/sum` in one pass (PERF-328-01, PERF-331-02).
    ///
    /// Returns `(range, weights, inv_sum)` where `inv_sum = 1/Σ weights[j]`.
    /// Uses [`StackWeights::new_with_sum`] to compute the sum in the same
    /// pass as the exp-ratchet, avoiding a redundant `weights.iter().sum()`
    /// call. Production: `SampleWindow::new` and sparse-cache normalization.
    ///
    /// # Arguments
    /// * `val` — Normalised intensity in `[0, num_bins-1]`
    /// * `num_bins` — Histogram bins per axis
    #[inline]
    pub fn compute_weights_with_inv_sum(
        &self,
        val: f32,
        num_bins: usize,
    ) -> (BinRange, StackWeights, f32) {
        let range = self.bin_range(val, num_bins);
        let (weights, sum) = StackWeights::new_with_sum(
            val,
            range.lo as usize,
            range.hi as usize,
            self.inv_2sigma_sq,
        );
        let inv_sum = 1.0 / sum;
        (range, weights, inv_sum)
    }

    /// Sum of Parzen weights (ARCH-320-06, ARCH-325-06).
    ///
    /// Returns `Σ exp(-(val-bin)²×inv_2sigma_sq)`. For interior values
    /// approximates `√(2πσ²)`. Useful for cross-validating exp-ratchet.
    ///
    /// # Arguments
    /// * `val` — Normalised intensity in `[0, num_bins-1]`
    /// * `num_bins` — Histogram bins per axis
    #[inline]
    #[allow(dead_code)] // Public API; internal callers use compute_weights_with_inv_sum
    pub fn sum_weights(&self, val: f32, num_bins: usize) -> f32 {
        let (_, weights) = self.compute_weights(val, num_bins);
        weights.iter().map(|(_, w)| w).sum()
    }

    /// `1/sum_weights` — normalization factor (PERF-328-01).
    ///
    /// Computes weights once. Approximates `1/√(2πσ²)` for interior values;
    /// larger near boundaries (compensating truncated tail) so normalized
    /// weights sum to ~1.0.
    ///
    /// # Arguments
    /// * `val` — Normalised intensity in `[0, num_bins-1]`
    /// * `num_bins` — Histogram bins per axis
    #[inline]
    #[allow(dead_code)] // Public API; internal callers use compute_weights_with_inv_sum
    pub fn inv_sum_weights(&self, val: f32, num_bins: usize) -> f32 {
        let (_, weights) = self.compute_weights(val, num_bins);
        let sum: f32 = weights.iter().map(|(_, w)| w).sum();
        1.0 / sum
    }
}
