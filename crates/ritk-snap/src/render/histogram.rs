//! Voxel intensity histogram computation SSOT.
//!
//! # Mathematical specification
//!
//! Given a float slice `data`, a range `[min, max)` (exclusive upper bound for
//! binning purposes), and a bin count `bins`, the histogram partitions the
//! range into `bins` equal-width intervals:
//!
//! ```text
//! w   = (max − min) / bins            (bin width)
//! i(v) = floor((v − min) / w)          (raw bin index)
//! ```
//!
//! Values below `min` are clamped into bin 0; values ≥ `max` are clamped into
//! bin `bins − 1`. This ensures all finite voxel values are counted regardless
//! of the chosen range.
//!
//! # Degenerate inputs
//!
//! If `max ≤ min`, `bins == 0`, or either bound is non-finite, an empty
//! `Histogram` (zero bins, empty counts vector) is returned.
//!
//! # Complexity
//!
//! O(N) time and O(bins) space. Suitable for volumes up to ~512³ voxels
//! without pre-filtering; a typical 256-bin histogram over a CT volume runs
//! in < 20 ms on a single core.

use eunomia::FloatElement;

// ── Histogram ──────────────────────────────────────────────────────────────────

/// Output of [`compute_histogram`]: per-bin voxel counts plus the input range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Histogram {
    /// Per-bin voxel counts; length equals `bins`.
    pub counts: Vec<u64>,
    /// Lower bound of the histogram range (inclusive).
    ///
    /// Stored as the bit pattern of an `f32` (converted via
    /// `f32::to_bits` / `f32::from_bits`) so that `Histogram` implements
    /// `Eq` without `f32`'s NaN semantics. Callers use the provided
    /// accessor [`Histogram::min`] to recover the float value.
    min_bits: u32,
    /// Upper bound of the histogram range (exclusive of binning, inclusive
    /// of clamping).
    max_bits: u32,
    /// Number of bins; equals `counts.len()`.
    pub bins: usize,
}

impl Histogram {
    /// Lower bound of the histogrammed intensity range.
    #[inline]
    pub fn min(&self) -> f32 {
        f32::from_bits(self.min_bits)
    }

    /// Upper bound of the histogrammed intensity range.
    #[inline]
    pub fn max(&self) -> f32 {
        f32::from_bits(self.max_bits)
    }

    /// Return an empty degenerate histogram (used for invalid inputs).
    fn empty(min: f32, max: f32) -> Self {
        Self {
            counts: Vec::new(),
            min_bits: min.to_bits(),
            max_bits: max.to_bits(),
            bins: 0,
        }
    }
}

// ── compute_histogram ──────────────────────────────────────────────────────────

/// Compute a `bins`-bin histogram of `data` over the range `[min, max]`.
///
/// # Mathematical specification
///
/// `w = (max − min) / bins`.  For each value `v`:
/// * `raw_f = (v − min) / w`
/// * `i = 0`             if `raw_f < 0.0`
/// * `i = bins − 1`      if `raw_f ≥ bins as f32`
/// * `i = raw_f as usize` otherwise
///
/// All finite values in `data` are counted. Non-finite values (`NaN`, ±∞)
/// are silently skipped.
///
/// # Degenerate inputs
///
/// Returns an empty histogram when `max ≤ min`, `bins == 0`, or either
/// bound is non-finite.
///
/// # Examples
///
/// ```rust
/// use ritk_snap::render::histogram::compute_histogram;
/// let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
/// let h = compute_histogram(&data, 0.0, 10.0, 2);
/// assert_eq!(h.counts[0], 5); // [0,5)
/// assert_eq!(h.counts[1], 5); // [5,10)
/// ```
pub fn compute_histogram(data: &[f32], min: f32, max: f32, bins: usize) -> Histogram {
    if bins == 0 || max <= min || !min.is_finite() || !max.is_finite() {
        return Histogram::empty(min, max);
    }

    let mut counts = vec![0u64; bins];
    let w = (max - min) / bins as f32;
    let bins_f = bins as f32;

    for &v in data {
        if !v.is_finite() {
            continue;
        }
        let raw_f = (v - min) / w;
        let i = if raw_f < 0.0 {
            0
        } else if raw_f >= bins_f {
            bins - 1
        } else {
            raw_f as usize
        };
        counts[i] += 1;
    }

    Histogram {
        counts,
        min_bits: min.to_bits(),
        max_bits: max.to_bits(),
        bins,
    }
}

// ── histogram_peak_count ───────────────────────────────────────────────────────

/// Return the maximum count across all bins.
///
/// Returns 0 for an empty histogram.
#[inline]
pub fn histogram_peak_count(h: &Histogram) -> u64 {
    h.counts.iter().copied().max().unwrap_or(0)
}

// ── histogram_bin_center ───────────────────────────────────────────────────────

/// Return the centre intensity value of bin `i`.
///
/// `center(i) = min + (i + 0.5) × w`  where `w = (max − min) / bins`.
///
/// Returns `min` for an empty (zero-bin) histogram.
#[inline]
pub fn histogram_bin_center(h: &Histogram, bin: usize) -> f32 {
    if h.bins == 0 {
        return h.min();
    }
    let w = (h.max() - h.min()) / h.bins as f32;
    h.min() + (bin as f32 + 0.5) * w
}

// ── bar_height_log ─────────────────────────────────────────────────────────────

/// Compute the display height of a histogram bar using a log₁₊₁ scale.
///
/// # Contract
///
/// `height(count, peak, h) = ln(count + 1) / ln(peak + 1) × h`
///
/// where `ln` is the natural logarithm. The invariants are:
/// - `bar_height_log(peak, peak, h) = h` (tallest bar fills the canvas).
/// - `bar_height_log(0, peak, h) = 0.0` (empty bin has zero height).
/// - `bar_height_log(count, 0, h) = 0.0` (no data, no bars).
///
/// Uses `f64` internally to avoid significant rounding errors for large
/// counts (up to ~2³² per bin).
///
/// `usize::try_from` is std's exact count conversion; bin counts derive from
/// in-memory voxel slices so they fit the address space. `from_count` is
/// exact below 2⁵³ with nearest rounding above, and `from_f64` narrows the
/// finite unit-range ratio exactly as the cast did.
#[inline]
pub fn bar_height_log(count: u64, peak: u64, available_height: f32) -> f32 {
    if peak == 0 || count == 0 {
        return 0.0;
    }
    let count_f =
        f64::from_count(usize::try_from(count).expect("bin count fits the address space"));
    let peak_f = f64::from_count(usize::try_from(peak).expect("bin count fits the address space"));
    let ratio = (count_f + 1.0).ln() / (peak_f + 1.0).ln();
    f32::from_f64(ratio * f64::from(available_height))
}

// ── wl_to_x ──────────────────────────────────────────────────────────────────

/// Map an intensity `value` from histogram range `[hist_min, hist_max]`
/// linearly to an x-pixel coordinate in `[x_left, x_right]`.
///
/// Returns `x_left` when the range is degenerate (`hist_max − hist_min < ε`).
#[inline]
pub fn wl_to_x(value: f32, hist_min: f32, hist_max: f32, x_left: f32, x_right: f32) -> f32 {
    let span = hist_max - hist_min;
    if span.abs() < f32::EPSILON {
        return x_left;
    }
    let t = (value - hist_min) / span;
    let x = x_left + t * (x_right - x_left);
    x.clamp(x_left, x_right)
}

#[cfg(test)]
#[path = "tests_histogram.rs"]
mod tests;
