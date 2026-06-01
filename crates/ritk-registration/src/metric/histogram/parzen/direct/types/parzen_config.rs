//! Precomputed Parzen-window parameters for one axis (SRP/SSOT).

use super::bin_range::BinRange;
use super::half_width::compute_half_width;
use super::stack_weights::StackWeights;

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
/// # Encapsulation (ARCH-322-03, ARCH-330-03)
///
/// All fields private. Construct via [`new`](Self::new) or
/// [`from_intensity_sigma`](Self::from_intensity_sigma). Access via
/// [`sigma_sq()`](Self::sigma_sq), [`half_width()`](Self::half_width),
/// and [`inv_2sigma_sq()`](Self::inv_2sigma_sq) (all production API).
/// `half_width()` and `inv_2sigma_sq()` were promoted from test-only in
/// Sprint 330 (ARCH-330-03) to support downstream consumers that need
/// the support window size or exponent factor without reconstructing
/// a `ParzenConfig`.
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
    ///
    /// Promoted to production API in ARCH-330-03 (was `#[cfg(test)]`).
    /// Downstream consumers (e.g. bin-range validation, capacity checks)
    /// need access to the support window size without test gating.
    #[allow(dead_code)] // Production API; currently called from test code and ParzenConfig internals
    #[inline]
    pub fn half_width(&self) -> usize {
        self.half_width
    }

    /// Return `-0.5 / sigma_sq`. Used by `StackWeights::new` and test code.
    ///
    /// Promoted to production API in ARCH-330-03 (was `#[cfg(test)]`).
    /// Downstream consumers (e.g. custom weight computation) need the
    /// exponent factor without test gating.
    #[allow(dead_code)] // Production API; currently called from test code and ParzenConfig internals
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
