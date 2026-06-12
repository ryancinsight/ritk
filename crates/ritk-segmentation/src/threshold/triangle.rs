//! Triangle thresholding method (Zack, Rogers & Latt 1977).
//!
//! # Mathematical Specification
//!
//! The triangle algorithm selects a threshold by maximising the perpendicular
//! distance from each histogram bin to a line drawn between the histogram peak
//! and the lowest-count tail.
//!
//! Given a normalised histogram h\[0..N−1\]:
//!
//! 1. Find the peak bin p = argmax h\[i\].
//! 2. Identify the tail bin t as the bin farthest from p (either 0 or N−1)
//!    that has the minimum count in that direction.
//! 3. For each bin i between p and t, compute the perpendicular distance d(i)
//!    from the point (i, h\[i\]) to the line segment (p, h\[p\])→(t, h\[t\]):
//!
//!      d(i) = |A·i + B·h\[i\] + C| / √(A² + B²)
//!
//!    where the line Ax + By + C = 0 passes through (p, h\[p\]) and (t, h\[t\]):
//!    A = h\[t\] − h\[p\]
//!    B = p − t
//!    C = t·h\[p\] − p·h\[t\]
//!
//! 4. t* = argmax_i d(i).
//! 5. Convert to intensity: t*_intensity = x_min + t* · (x_max − x_min) / (N − 1).
//!
//! # Complexity
//! Histogram construction: O(n) voxels.
//! Threshold search:       O(N) bins.
//! Total:                  O(n + N).
//!
//! # References
//! - Zack G.W., Rogers W.E., Latt S.A. (1977). "Automatic measurement of
//!   sister chromatid exchange frequency." *J. Histochem. Cytochem.* 25(7):741–753.

use burn::tensor::backend::Backend;
use ritk_image::Image;

use super::auto_threshold::AutoThreshold;

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

    /// Triangle geometric criterion (Zack et al. 1977).
    ///
    /// # Algorithm
    /// 1. Find the peak bin (highest raw count).
    /// 2. Identify the tail bin (farthest non-zero end from the peak).
    /// 3. For each bin between peak and tail, compute the perpendicular
    ///    distance to the peak–tail line using f64 arithmetic.
    /// 4. t* = argmax distance.
    /// 5. t*_intensity = x_min + t* / (N−1) · (x_max − x_min).
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        // Work with u64 counts to match the precision of the existing
        // compute_triangle_threshold_from_slice implementation.
        let counts: Vec<u64> = hist.iter().map(|&c| c as u64).collect();

        // Find peak bin (highest count).
        let peak_bin = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Identify tail bin (farthest end from the peak).
        let tail_bin = if peak_bin <= n_bins / 2 {
            counts.iter().rposition(|&c| c > 0).unwrap_or(n_bins - 1)
        } else {
            counts.iter().position(|&c| c > 0).unwrap_or(0)
        };

        // Degenerate: peak and tail coincide.
        if peak_bin == tail_bin {
            return x_min + peak_bin as f32 / (n_bins - 1) as f32 * (x_max - x_min);
        }

        // Line equation coefficients (f64 for numerical stability).
        let x1 = peak_bin as f64;
        let y1 = counts[peak_bin] as f64;
        let x2 = tail_bin as f64;
        let y2 = counts[tail_bin] as f64;

        // Line: A·x + B·y + C = 0
        let a = y2 - y1;
        let b = x1 - x2;
        let c = x2 * y1 - x1 * y2;
        let norm = (a * a + b * b).sqrt();

        // Search for maximum perpendicular distance.
        let (start, end) = if peak_bin < tail_bin {
            (peak_bin + 1, tail_bin)
        } else {
            (tail_bin + 1, peak_bin)
        };

        let mut best_dist = 0.0_f64;
        let mut best_bin = start;

        for (i, &cnt) in counts.iter().enumerate().take(end).skip(start) {
            let xi = i as f64;
            let yi = cnt as f64;
            let dist = (a * xi + b * yi + c).abs() / norm;
            if dist > best_dist {
                best_dist = dist;
                best_bin = i;
            }
        }

        // Convert bin index to intensity.
        x_min + best_bin as f32 / (n_bins - 1) as f32 * (x_max - x_min)
    }
}

// ── Convenience functions ──────────────────────────────────────────────────────

/// Convenience function: compute the triangle threshold with 256 bins.
pub fn triangle_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    TriangleThreshold::new().compute(image)
}

/// Compute the triangle threshold for a contiguous f32 intensity slice.
pub fn compute_triangle_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    assert!(num_bins >= 2, "num_bins must be >= 2");

    let n = slice.len();
    if n == 0 {
        return 0.0;
    }

    // ── Intensity range ────────────────────────────────────────────────────────
    let x_min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Degenerate case: constant image.
    if (x_max - x_min).abs() < f32::EPSILON {
        return x_min;
    }

    let range = x_max - x_min;
    let num_bins_f = (num_bins - 1) as f64;

    // ── Build histogram ────────────────────────────────────────────────────────
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) as f64 / range as f64 * num_bins_f).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }

    // ── Find peak bin ──────────────────────────────────────────────────────────
    let peak_bin = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);

    // ── Identify tail bin ──────────────────────────────────────────────────────
    // The tail is the end of the histogram farthest from the peak.
    // If the peak is in the left half, the tail is the rightmost non-zero bin
    // (or N−1). Otherwise, the tail is the leftmost non-zero bin (or 0).
    let tail_bin = if peak_bin <= num_bins / 2 {
        // Tail on the right side.
        counts.iter().rposition(|&c| c > 0).unwrap_or(num_bins - 1)
    } else {
        // Tail on the left side.
        counts.iter().position(|&c| c > 0).unwrap_or(0)
    };

    // Degenerate: peak and tail coincide.
    if peak_bin == tail_bin {
        return x_min + peak_bin as f32 / num_bins_f as f32 * range;
    }

    // ── Line equation coefficients ─────────────────────────────────────────────
    // Line through (peak_bin, counts[peak_bin]) and (tail_bin, counts[tail_bin]).
    // Using f64 for numerical stability.
    let x1 = peak_bin as f64;
    let y1 = counts[peak_bin] as f64;
    let x2 = tail_bin as f64;
    let y2 = counts[tail_bin] as f64;

    // Line: A·x + B·y + C = 0
    let a = y2 - y1;
    let b = x1 - x2;
    let c = x2 * y1 - x1 * y2;
    let norm = (a * a + b * b).sqrt();

    // ── Search for maximum perpendicular distance ──────────────────────────────
    let (start, end) = if peak_bin < tail_bin {
        (peak_bin + 1, tail_bin)
    } else {
        (tail_bin + 1, peak_bin)
    };

    let mut best_dist = 0.0_f64;
    let mut best_bin = start;

    for (i, &cnt) in counts.iter().enumerate().take(end).skip(start) {
        let xi = i as f64;
        let yi = cnt as f64;
        let dist = (a * xi + b * yi + c).abs() / norm;

        if dist > best_dist {
            best_dist = dist;
            best_bin = i;
        }
    }

    // ── Convert bin index to intensity ─────────────────────────────────────────
    x_min + best_bin as f32 / num_bins_f as f32 * range
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_triangle.rs"]
mod tests;
