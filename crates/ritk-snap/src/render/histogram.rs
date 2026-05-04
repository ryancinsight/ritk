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

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_histogram tests ───────────────────────────────────────────────

    /// Uniform data from 0.0 to 255.0 in 256 bins: each bin contains exactly 1.
    #[test]
    fn uniform_data_256_bins_each_count_one() {
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let h = compute_histogram(&data, 0.0, 256.0, 256);
        assert_eq!(h.bins, 256);
        assert_eq!(h.counts.len(), 256);
        for (i, &c) in h.counts.iter().enumerate() {
            assert_eq!(c, 1, "bin {i} expected count 1, got {c}");
        }
    }

    /// All values equal to min are placed in bin 0.
    #[test]
    fn all_values_at_min_placed_in_bin_zero() {
        let data = vec![0.0f32; 100];
        let h = compute_histogram(&data, 0.0, 100.0, 10);
        assert_eq!(h.counts[0], 100, "all 100 values at min must be in bin 0");
        for i in 1..10 {
            assert_eq!(h.counts[i], 0, "bin {i} must be empty");
        }
    }

    /// Values exactly at max (= 100.0) clamp to the last bin.
    #[test]
    fn values_at_max_clamp_to_last_bin() {
        let data = vec![100.0f32; 50];
        let h = compute_histogram(&data, 0.0, 100.0, 10);
        // raw_f = (100.0 - 0.0) / 10.0 = 10.0 ≥ bins(10) → clamped to bin 9
        assert_eq!(h.counts[9], 50, "all 50 values at max must be in last bin");
        for i in 0..9 {
            assert_eq!(h.counts[i], 0, "bin {i} must be empty");
        }
    }

    /// Values below min clamp to bin 0.
    #[test]
    fn below_min_clamped_to_bin_zero() {
        let data = vec![-50.0f32, 5.0, 5.0];
        let h = compute_histogram(&data, 0.0, 10.0, 2);
        // -50.0 → raw_f < 0 → bin 0
        // 5.0 → raw_f = 1.0 → bin 1
        assert_eq!(h.counts[0], 1, "value below min must be in bin 0");
        assert_eq!(h.counts[1], 2, "5.0 values must be in bin 1");
    }

    /// Values above max clamp to the last bin.
    #[test]
    fn above_max_clamped_to_last_bin() {
        let data = vec![200.0f32, 5.0];
        let h = compute_histogram(&data, 0.0, 10.0, 2);
        // 200.0 → raw_f ≥ 2 → bin 1
        // 5.0 → raw_f = 1.0 → bin 1
        assert_eq!(h.counts[0], 0, "bin 0 must be empty");
        assert_eq!(h.counts[1], 2, "both values must be in last bin");
    }

    /// Empty input produces an all-zero count vector.
    #[test]
    fn empty_data_produces_all_zero_counts() {
        let h = compute_histogram(&[], 0.0, 100.0, 10);
        assert_eq!(h.bins, 10);
        assert_eq!(h.counts.len(), 10);
        assert!(
            h.counts.iter().all(|&c| c == 0),
            "all counts must be zero for empty input"
        );
    }

    /// Two equal-sized bins, exact half-split at bin boundary.
    ///
    /// Analytical: w=5.0; values 0..=4 fall in [0,5) → bin 0;
    /// values 5..=9 fall in [5,10) → bin 1.
    #[test]
    fn two_bin_exact_half_split() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let h = compute_histogram(&data, 0.0, 10.0, 2);
        assert_eq!(h.counts[0], 5, "values [0,5) → bin 0");
        assert_eq!(h.counts[1], 5, "values [5,10) → bin 1");
    }

    /// Degenerate range (max == min) returns an empty histogram.
    #[test]
    fn degenerate_max_equals_min_returns_empty() {
        let h = compute_histogram(&[1.0, 2.0, 3.0], 5.0, 5.0, 10);
        assert_eq!(h.bins, 0, "degenerate range must produce 0 bins");
        assert!(h.counts.is_empty(), "counts must be empty for degenerate range");
    }

    /// `histogram_bin_center` returns analytically correct center for each bin.
    ///
    /// w = (10.0 − 0.0) / 4 = 2.5; center(i) = 0.0 + (i + 0.5) × 2.5
    #[test]
    fn bin_center_matches_analytical_formula() {
        let h = compute_histogram(&[1.0f32], 0.0, 10.0, 4);
        // w = 2.5; centers = 1.25, 3.75, 6.25, 8.75
        let expected = [1.25f32, 3.75, 6.25, 8.75];
        for (i, &exp) in expected.iter().enumerate() {
            let got = histogram_bin_center(&h, i);
            assert!(
                (got - exp).abs() < 1e-5,
                "bin {i}: expected center {exp}, got {got}"
            );
        }
    }
}
