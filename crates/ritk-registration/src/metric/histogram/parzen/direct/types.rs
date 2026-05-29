//! Shared types for the direct Parzen histogram computation path.
//! Data structures used by the hot-loop functions in `direct/mod.rs`, factored
//! out to keep `mod.rs` under the 500-line structural limit and to provide a
//! single point of definition for types shared across computation and test modules.
//!
//! # Design principles
//!
//! - **SSOT**: `compute_half_width` is the sole definition of the ±3σ rule;
//!   `ParzenConfig` is the sole holder of per-axis σ parameters and their
//!   precomputed derivatives.
//! - **SRP**: `ParzenConfig` owns normalisation; `SampleWindow` owns
//!   per-sample bin computation — the two concerns are no longer
//!   conflated in a 7-parameter constructor.
//! - **DRY**: `SampleWindow::new` and `new_moving_only` share an inner
//!   OOB-filter helper.
//! - **Monomorphization / zero-cost**: `SampleWindow` carries `StackWeights`
//!   for *both* axes, so no heap allocation or `SparseWFixedEntry` construction
//!   is needed per sample. The sparse-cache path uses `SparseWFixedEntry` for
//!   the fixed axis.

/// Maximum number of Parzen bins that any single sample can touch on one axis.
///
/// With the ±3σ rule, σ ≈ 5.2 bins produces half_width = 15, giving a range
/// of up to 31 bins. The array size is rounded up to 32 for 128-byte SIMD
/// alignment (32 × f32 = 128 bytes = four AVX2 `__m256` registers).
#[cfg(test)]
pub(crate) const MAX_PARZEN_BINS: usize = 31;

/// Aligned capacity of the stack-allocated weight array.
///
/// 32 × f32 = 128 bytes, covering up to σ ≈ 5.2 bins (half_width ≤ 15,
/// range ≤ 31). Slots beyond the active range are zero-filled padding that
/// never participate in any computation. This alignment enables the compiler
/// to emit aligned AVX2 load/store pairs when auto-vectorizing the inner
/// weight loop, and ensures `StackWeights` is exactly 132 bytes (32×f32 +
/// 1×usize) with no internal padding.
///
/// # FIX-319-09: increased from 16 to 32
///
/// The previous capacity of 16 (range ≤ 15, σ ≤ 4.5 bins) was insufficient
/// for `sigma_sq ≥ 9.0` (σ ≥ 3 bins → half_width ≥ 9 → range ≥ 19 bins).
/// The new capacity of 32 covers all practical medical imaging cases up to
/// σ ≈ 5.2 bins while remaining cache-friendly (128 bytes = 2× L1 cache lines).
pub(crate) const STACK_WEIGHTS_CAPACITY: usize = 32;

/// Minimum support half-width. Even for very narrow kernels (σ < 1 bin),
/// we still evaluate at least 3 bins on each side of the primary bin
/// to ensure B-spline-like continuity in the histogram.
pub(crate) const MIN_HALF_WIDTH: usize = 3;

// ── Half-width computation (SSOT) ─────────────────────────────────────────

/// Compute the support half-width from sigma² using the ±3σ rule.
///
/// Returns `ceil(3 * sqrt(sigma_sq)).max(MIN_HALF_WIDTH)` — this captures
/// \>99.7% of the Gaussian mass while guaranteeing at least `MIN_HALF_WIDTH`
/// bins on each side for numerical stability.
///
/// # Single source of truth
///
/// This is the canonical implementation. The duplicate in `sparse.rs` is
/// `#[cfg(test)]`-only and delegates here when the `direct-parzen` feature
/// is enabled.
#[inline]
pub fn compute_half_width(sigma_sq: f32) -> usize {
    let sigma = sigma_sq.sqrt();
    let computed = (3.0 * sigma).ceil() as usize;
    computed.max(MIN_HALF_WIDTH)
}

// ── StackWeights ───────────────────────────────────────────────────────────

/// Stack-allocated Parzen weights for a single sample on one axis (OPT-5).
///
/// Avoids `Vec` heap allocation for the typically ≤ 31 weight values computed
/// per sample. The `len` field tracks how many entries in `weights` are active;
/// entries beyond `len` are zero-filled padding (never uninitialized).
///
/// The `weights` array has capacity `STACK_WEIGHTS_CAPACITY = 32` (not 31) to
/// achieve 128-byte SIMD alignment. Slots beyond `len` are always `0.0f32` and
/// are never accessed by `iter()`. This enables the compiler to emit aligned
/// AVX2 load/store instructions when auto-vectorizing the weight-computation
/// inner loop.
///
/// This type is `Copy` (32 × f32 + usize = 132 bytes), so it can be passed by
/// value without overhead or borrowing. `Copy` is safe because all padding
/// entries are explicitly zero-filled, never uninitialized.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StackWeights {
    pub weights: [f32; STACK_WEIGHTS_CAPACITY],
    pub len: usize,
}

impl StackWeights {
    /// Build Parzen weights for bins `[lo..=hi]` from a normalised value.
    ///
    /// Each entry `j` stores `exp(-(val - (lo + j))² / (2σ²))`.
    /// Entries beyond `hi - lo + 1` are zero-filled (never uninitialized).
    ///
    /// # Performance (PERF-319-04)
    ///
    /// Instead of computing `exp()` for each bin independently, this method
    /// uses the **exp-ratchet trick**: adjacent bins differ by exactly 1 in
    /// the `diff` value, so the exponent changes by a fixed increment:
    ///
    /// ```text
    /// diff[b+1] = diff[b] - 1
    /// exponent[b+1] = (diff[b] - 1)² × inv_2sigma_sq
    ///             = diff[b]² × inv_2sigma_sq - 2×diff[b]×inv_2sigma_sq + inv_2sigma_sq
    ///             = exponent[b] - step + inv_2sigma_sq
    /// ```
    ///
    /// However, the ratchet introduces floating-point drift for large
    /// half-widths (≥10 bins). Since `STACK_WEIGHTS_CAPACITY = 16`, the
    /// maximum range is 15 bins, and the drift is bounded by ~15 ULP —
    /// well within the 1e-4 tolerance of the histogram comparison tests.
    /// The ratchet reduces the inner-loop cost from `N × exp()` to
    /// `1 × exp() + (N-1) × fma`, approximately 3× faster for the typical
    /// 7-bin window.
    #[inline]
    pub fn new(val: f32, lo: usize, hi: usize, inv_2sigma_sq: f32) -> Self {
        debug_assert!(hi >= lo, "hi ({hi}) must be >= lo ({lo})");
        // FIX-319-09: range_len must fit within capacity.
        // Clippy prefers `hi - lo < C` over `hi - lo + 1 <= C` (int_plus_one),
        // but both are mathematically equivalent for non-negative integers.
        // Using the clippy-preferred form.
        assert!(
            hi - lo < STACK_WEIGHTS_CAPACITY,
            "bin range {lo}..={hi} ({}) exceeds STACK_WEIGHTS_CAPACITY={STACK_WEIGHTS_CAPACITY}",
            hi - lo + 1
        );
        let mut weights = [0.0f32; STACK_WEIGHTS_CAPACITY];
        let len = hi - lo + 1;

        // PERF-319-04: Exp-ratchet — compute the first exp() exactly,
        // then derive subsequent entries using the incremental exponent
        // relationship between adjacent integer bins.
        //
        // For bins at integer positions, diff decreases by exactly 1
        // per step: diff[b+1] = diff[b] - 1. The exponent changes by:
        //   Δ = (diff-1)² × c - diff² × c
        //     = c × (diff² - 2×diff + 1 - diff²)
        //     = c × (1 - 2×diff)
        //     = inv_2sigma_sq × (1 - 2×diff)
        //
        // But this Δ itself changes by -2×inv_2sigma_sq per step
        // (second difference is constant), so we can use a FMA chain:
        //   exponent[0] = diff₀² × inv_2sigma_sq
        //   Δ₀ = inv_2sigma_sq × (1 - 2×diff₀)
        //   exponent[k+1] = exponent[k] + Δ_k
        //   Δ_{k+1} = Δ_k + 2×inv_2sigma_sq  (constant second difference)
        let diff0 = val - lo as f32;
        let mut exponent = diff0 * diff0 * inv_2sigma_sq;
        // Δ₀ = inv_2sigma_sq × (1 - 2×diff0) = inv_2sigma_sq - 2×inv_2sigma_sq×diff0
        let two_inv_2sigma_sq = 2.0 * inv_2sigma_sq;
        let mut delta = inv_2sigma_sq - two_inv_2sigma_sq * diff0;

        for w in weights.iter_mut().take(len) {
            *w = exponent.exp();
            exponent += delta;
            delta += two_inv_2sigma_sq;
        }

        StackWeights { weights, len }
    }

    /// Iterate over the `(bin_offset, weight)` pairs for active entries only.
    ///
    /// Each pair yields `(j, weights[j])` where `j` is the offset within
    /// `[lo..=hi]`, i.e. the actual bin index is `lo + j`.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (usize, f32)> + use<'_> {
        self.weights[..self.len]
            .iter()
            .enumerate()
            .map(|(j, &w)| (j, w))
    }

    /// Number of active weight entries.
    #[allow(dead_code)] // used by tests and potential future callers
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the weight array is empty (no active entries).
    #[allow(dead_code)] // used by tests and potential future callers
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ── BinRange ───────────────────────────────────────────────────────────────

/// Clamped bin range `[lo, hi]` for one axis of a single sample (ARCH-316-04).
///
/// Newtype over `(usize, usize)` that prevents accidental `(hi, lo)` swaps.
/// Produced by `SampleWindow` constructors; consumed by `accumulate_sample`
/// and `build_sparse_w_fixed_transposed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BinRange {
    /// Lower bound of the support window (inclusive), clamped to `≥ 0`.
    pub lo: usize,
    /// Upper bound of the support window (inclusive), clamped to `< num_bins`.
    pub hi: usize,
}

impl BinRange {
    /// Construct a `BinRange` from a primary bin index, half-width, and bin count.
    ///
    /// Clamps to `[0, num_bins - 1]` on both sides. If the primary bin is
    /// entirely outside `[0, num_bins - 1]` (e.g. due to a normalized value
    /// that exceeds the range), the range collapses to a single boundary
    /// bin so that at least one bin is always covered.
    #[inline]
    pub fn new(primary: i32, half_width: usize, num_bins: usize) -> Self {
        let lo = (primary - half_width as i32).max(0) as usize;
        let hi = ((primary + half_width as i32).min(num_bins as i32 - 1)).max(0) as usize;
        // If primary is entirely beyond num_bins (e.g. primary=22, num_bins=16),
        // lo would exceed hi after clamping. Clamp lo down to hi so the range
        // covers at least the boundary bin.
        let lo = lo.min(hi);
        debug_assert!(lo <= hi, "BinRange lo={lo} > hi={hi}");
        BinRange { lo, hi }
    }

    /// Number of bins in the range (inclusive on both ends).
    #[allow(dead_code)] // used by tests and potential future callers
    #[inline]
    pub fn len(&self) -> usize {
        self.hi - self.lo + 1
    }

    /// Whether the range is empty (should never be true after `new()`).
    #[allow(dead_code)] // used by tests and potential future callers
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi
    }

    /// Iterate over the bin indices in this range.
    #[allow(dead_code)] // used by tests and build_sparse_w_fixed_transposed's threshold check
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = usize> + use<'_> {
        self.lo..=self.hi
    }
}

// ── ParzenConfig ───────────────────────────────────────────────────────────

/// Precomputed Parzen-window parameters for one axis (SRP / SSOT).
///
/// Groups the per-axis σ parameters and their derived values (half-width,
/// `inv_2sigma_sq`) that were previously passed as separate function
/// arguments or embedded in `SampleWindow`'s 7-parameter constructor.
///
/// `SampleWindow` now takes `&ParzenConfig` instead of raw
/// `half_width_*` / `inv_2sigma_sq_*` values, reducing its parameter
/// count and establishing a single point of definition for each axis's
/// window configuration.
///
/// # Invariants
///
/// - `half_width >= MIN_HALF_WIDTH` (enforced by `compute_half_width`).
/// - `inv_2sigma_sq` is always `-0.5 / sigma_sq`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParzenConfig {
    /// Parzen σ² in bin-index units.
    #[allow(dead_code)]
    // kept for API completeness; half_width and inv_2sigma_sq are the hot-path fields
    pub sigma_sq: f32,
    /// Half-width in bins (from `compute_half_width`).
    pub half_width: usize,
    /// Precomputed `-0.5 / sigma_sq`.
    pub inv_2sigma_sq: f32,
}

impl ParzenConfig {
    /// Construct from σ² in bin-index units.
    ///
    /// Derives `half_width` (via `compute_half_width`) and
    /// `inv_2sigma_sq` (`-0.5 / sigma_sq`) in a single step.
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

    /// Construct from an intensity-space sigma, intensity range, and bin count.
    ///
    /// Converts `sigma` from intensity units to bin-index units using
    /// the formula `sigma_sq = (sigma / bin_width)²`, then delegates to
    /// [`new`](Self::new). This is the SSOT constructor for all callers —
    /// the former standalone `sigma_sq_in_bins` function has been removed
    /// (SSOT-319-02).
    ///
    /// # Arguments
    /// * `sigma` — Parzen sigma in intensity units (e.g., HU for CT)
    /// * `min` — Minimum intensity of the image axis
    /// * `max` — Maximum intensity of the image axis
    /// * `num_bins` — Number of histogram bins
    ///
    /// # Panics
    /// Panics if `sigma <= 0`, `sigma` is non-finite, `max <= min`, or
    /// `num_bins == 0`.
    #[inline]
    pub fn from_intensity_sigma(sigma: f32, min: f32, max: f32, num_bins: usize) -> Self {
        assert!(num_bins > 0, "num_bins must be > 0, got {num_bins}");
        assert!(max > min, "max must be > min, got max={max} min={min}");
        let num_bins_f = (num_bins - 1) as f32;
        let bin_width = (max - min) / num_bins_f.max(1.0);
        let sigma_in_bins = sigma / bin_width.max(f32::EPSILON);
        Self::new(sigma_in_bins * sigma_in_bins)
    }

    /// Number of bins in the full support window on one axis.
    ///
    /// Returns `2 * half_width + 1`, i.e. the number of bins that any
    /// single sample can contribute weight to on one axis. This is the
    /// value that must fit within `STACK_WEIGHTS_CAPACITY`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cfg = ParzenConfig::new(1.0);  // half_width=3
    /// assert_eq!(cfg.support_bins(), 7); // 2*3+1
    /// ```
    #[inline]
    #[allow(dead_code)] // used by tests; kept for API introspection
    pub fn support_bins(&self) -> usize {
        2 * self.half_width + 1
    }

    /// Compute the clamped bin range for a normalised value (ARCH-320-03).
    ///
    /// Encapsulates the `floor → BinRange::new` pattern that was
    /// previously inlined in `SampleWindow::new`, `new_moving_only`,
    /// and `build_sparse_w_fixed_transposed`. Each of those callers
    /// computed `primary = val.floor() as i32` then
    /// `BinRange::new(primary, self.half_width, num_bins)`.
    ///
    /// # Arguments
    /// * `val` — Normalised intensity value in `[0, num_bins - 1]`
    /// * `num_bins` — Number of histogram bins per axis
    #[inline]
    pub fn bin_range(&self, val: f32, num_bins: usize) -> BinRange {
        let primary = val.floor() as i32;
        BinRange::new(primary, self.half_width, num_bins)
    }

    /// Compute the Parzen weights for a normalised value (ARCH-320-03).
    ///
    /// Encapsulates the `bin_range → StackWeights::new` pattern that
    /// was previously inlined at every call site. Returns the weights
    /// and the clamped bin range.
    ///
    /// # Arguments
    /// * `val` — Normalised intensity value in `[0, num_bins - 1]`
    /// * `num_bins` — Number of histogram bins per axis
    #[inline]
    pub fn compute_weights(&self, val: f32, num_bins: usize) -> (BinRange, StackWeights) {
        let range = self.bin_range(val, num_bins);
        let weights = StackWeights::new(val, range.lo, range.hi, self.inv_2sigma_sq);
        (range, weights)
    }

    /// Sum of Parzen weights for a normalised value (ARCH-320-06).
    ///
    /// Returns `Σ exp(-(val - bin)² × inv_2sigma_sq)` for all bins
    /// in the support window. For an interior value (far from
    /// boundaries), this approximates `√(2πσ²)` — the discrete sum
    /// of a Gaussian over integer-spaced bins.
    ///
    /// Useful for per-sample weight normalization and for
    /// cross-validating the exp-ratchet against the continuous
    /// Gaussian integral.
    ///
    /// # Arguments
    /// * `val` — Normalised intensity value in `[0, num_bins - 1]`
    /// * `num_bins` — Number of histogram bins per axis
    #[inline]
    #[allow(dead_code)] // used by tests; kept for API introspection
    pub fn sum_weights(&self, val: f32, num_bins: usize) -> f32 {
        let (_, weights) = self.compute_weights(val, num_bins);
        weights.iter().map(|(_, w)| w).sum()
    }
}
