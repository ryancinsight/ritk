//! Triangle thresholding method (Zack, Rogers & Latt 1977), matching ITK's
//! `itk::TriangleThresholdCalculator`.
//!
//! # Mathematical Specification
//!
//! The triangle algorithm draws a line from the histogram peak to the opposite
//! end of the distribution and selects the bin of maximum vertical distance
//! between the histogram and that line. ITK departs from the textbook form in
//! three details that materially shift the result on skewed histograms; all
//! three are reproduced here:
//!
//! Given a histogram `h[0..N−1]` with peak bin `p = argmax h`:
//!
//! 1. The line endpoints are the **1st and 99th percentile bins**
//!    (`p_lo`, `p_hi`) — the first bin whose cumulative count reaches 1 % / 99 %
//!    of the total — not the first/last non-empty bins. This makes the endpoint
//!    robust to single-voxel outlier tails.
//! 2. The active side is the one **farther** from the peak:
//!    `|p − p_lo| > |p − p_hi|` selects the low side, otherwise the high side.
//! 3. The bin of maximum distance is **incremented by one**, and the threshold
//!    is the **bin centre**: `x_min + (idx + 0.5) · (x_max − x_min)/N`.
//!
//! For the high side the distance at bin `k ∈ [p, p_hi)` is
//! `slope·(k − p) + h[p] − h[k]` with `slope = −h[p]/(p_hi − p)`; for the low
//! side, bin `k ∈ [p_lo, p)` uses `slope·(k − p_lo) − h[k]` with
//! `slope = h[p]/(p − p_lo)`. The normalising `1/√(A²+B²)` of the perpendicular
//! distance is a positive constant over the search range, so `argmax` is
//! unaffected and the cheaper signed triangle height is used.
//!
//! # Complexity
//! Histogram construction: O(n) voxels. Threshold search: O(N) bins.
//!
//! # References
//! - Zack G.W., Rogers W.E., Latt S.A. (1977). "Automatic measurement of
//!   sister chromatid exchange frequency." *J. Histochem. Cytochem.* 25(7):741–753.
//! - ITK `itkTriangleThresholdCalculator.hxx` (percentile endpoints, +1 shift).

use ritk_image::tensor::Backend;
use ritk_image::Image;

use super::auto_threshold::{itk_bin_width, threshold_from_slice, AutoThreshold};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Triangle thresholding segmentation.
///
/// Selects a threshold by maximising the perpendicular distance from each
/// histogram bin to the line connecting the histogram peak and the lowest tail.
#[derive(Debug, Clone)]
pub struct TriangleThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl TriangleThreshold {
    /// Create a `TriangleThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `TriangleThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the optimal triangle threshold for `image`.
    ///
    /// Returns the intensity value t* that maximises the perpendicular distance
    /// to the peak–tail line. For a constant image, returns the image's uniform
    /// intensity (degenerate case).
    ///
    /// Delegates to [`AutoThreshold::compute`].
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the triangle threshold to produce a binary mask.
    ///
    /// - Pixels with intensity ≥ t* → 1.0 (foreground).
    /// - Pixels with intensity <  t* → 0.0 (background).
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    ///
    /// Delegates to [`AutoThreshold::apply`].
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        <Self as AutoThreshold>::apply(self, image)
    }

    /// Apply the auto-threshold to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the native output image cannot be constructed.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::native::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        <Self as AutoThreshold>::apply_native(self, image, backend)
    }
}

impl Default for TriangleThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// ── AutoThreshold implementation ───────────────────────────────────────────────

impl AutoThreshold for TriangleThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Triangle geometric criterion, matching `itk::TriangleThresholdCalculator`
    /// (see the module docs for the percentile-endpoint / +1 / bin-centre rules).
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let counts: Vec<u64> = hist.iter().map(|&c| c as u64).collect();
        let bin_size = itk_bin_width(x_min, x_max, n_bins);
        triangle_from_counts(&counts, x_min as f64, bin_size)
    }
}

// ── ITK-faithful core ────────────────────────────────────────────────────────

/// First bin index whose cumulative count reaches `quantile · total`
/// (the bin containing the `quantile` percentile), mirroring ITK's use of
/// `Histogram::Quantile` followed by `GetIndex`.
fn percentile_bin(counts: &[u64], total: u64, quantile: f64) -> usize {
    let target = quantile * total as f64;
    let mut running = 0u64;
    for (i, &c) in counts.iter().enumerate() {
        running += c;
        if running as f64 >= target {
            return i;
        }
    }
    counts.len().saturating_sub(1)
}

/// Compute the triangle threshold intensity from a histogram, following
/// `itk::TriangleThresholdCalculator`. `x_min` is the lower intensity bound and
/// `bin_size = (x_max − x_min)/N`; the returned value is the bin centre of the
/// selected (`+1`-shifted) bin.
#[allow(clippy::needless_range_loop)]
fn triangle_from_counts(counts: &[u64], x_min: f64, bin_size: f64) -> f32 {
    let n = counts.len();
    let total: u64 = counts.iter().sum();
    if n == 0 || total == 0 {
        return x_min as f32;
    }

    let bin_centre = |idx: usize| -> f32 { (x_min + (idx as f64 + 0.5) * bin_size) as f32 };

    // Peak bin (highest frequency).
    let peak = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let peak_h = counts[peak] as f64;

    // Percentile endpoints (robust to outlier tails).
    let p_lo = percentile_bin(counts, total, 0.01);
    let p_hi = percentile_bin(counts, total, 0.99);

    // Operate on the side of the peak that is farther away.
    let toward_lo = peak.abs_diff(p_lo) > peak.abs_diff(p_hi);

    let mut best = f64::NEG_INFINITY;
    let mut best_idx = peak;
    if toward_lo && peak > p_lo {
        let slope = peak_h / (peak as f64 - p_lo as f64);
        for k in p_lo..peak {
            let d = slope * (k as f64 - p_lo as f64) - counts[k] as f64;
            if d > best {
                best = d;
                best_idx = k;
            }
        }
    } else if !toward_lo && p_hi > peak {
        let slope = -peak_h / (p_hi as f64 - peak as f64);
        for k in peak..p_hi {
            let d = slope * (k as f64 - peak as f64) + peak_h - counts[k] as f64;
            if d > best {
                best = d;
                best_idx = k;
            }
        }
    } else {
        // Degenerate: percentile endpoint coincides with the peak.
        return bin_centre(peak);
    }

    // ITK increments the selected bin by one before reporting its centre.
    bin_centre((best_idx + 1).min(n - 1))
}

// ── Convenience functions ──────────────────────────────────────────────────────

/// Convenience function: compute the triangle threshold with 256 bins.
pub fn triangle_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    TriangleThreshold::new().compute(image)
}

/// Compute the triangle threshold for a contiguous f32 intensity slice,
/// matching `itk::TriangleThresholdCalculator`.
///
/// Delegates to the shared `threshold_from_slice` pipeline so it is
/// bit-identical to [`TriangleThreshold::compute`].
pub fn compute_triangle_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    threshold_from_slice(&TriangleThreshold::with_bins(num_bins), slice)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_triangle.rs"]
mod tests;
