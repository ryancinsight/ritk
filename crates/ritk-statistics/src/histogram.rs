//! Standalone histogram computation with explicit `[min, max]` range.
//!
//! # Mathematical Specification
//!
//! Given a vector of intensity values `V = {v₁, …, vₙ}` and a range
//! `[min, max]` partitioned into `bins` equal-width subintervals
//!
//!   Δw = (max − min) / bins
//!   bin_k = [min + k · Δw, min + (k+1) · Δw)   for k = 0, …, bins−1
//!
//! the histogram count for `bin_k` is
//!
//!   h_k = |{ v ∈ V : min + k·Δw ≤ v < min + (k+1)·Δw }|
//!
//! The **last bin is half-open `[…, max]`** so that `v == max` is included;
//! the contract matches `numpy.histogram` and `scipy.ndimage.histogram`.
//!
//! Values strictly below `min` or strictly above `max` are silently
//! excluded — this is the scipy.ndimage semantic, not numpy's full
//! `[-inf, +inf]` extension. Callers wanting the numpy behaviour should
//! pass `min = v_min`, `max = v_max` from `compute_statistics`.
//!
//! # Complexity
//!
//! O(n) where n = |V|. One pass; per-element cost is one comparison, one
//! multiply, one floor, and one bounds check. `bins + 1` returns the
//! vector storage (excluding the `min`, `max` scalars).
//!
//! # Type-class dispatch
//!
//! The function is **generic over `B: Backend` and `const D: usize`** —
//! the same implementation serves 1-D, 2-D, 3-D, and arbitrary-D images
//! by extracting the contiguous f32 storage and iterating. No
//! `dyn Trait`, no vtable indirection.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

/// Histogram counts over a fixed `[min, max]` range.
///
/// `counts[k]` is the number of voxels falling into bin `k`. The bin
/// edges are evenly spaced with `Δw = (max − min) / bins`, and the
/// rightmost edge is **inclusive** (scipy.ndimage convention).
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    /// Lower edge of the histogram range.
    pub min: f32,
    /// Upper edge of the histogram range (inclusive).
    pub max: f32,
    /// Number of equal-width bins.
    pub bins: usize,
    /// Per-bin counts of length `bins`.
    pub counts: Vec<usize>,
}

impl Histogram {
    /// Total number of voxels counted (sum of `counts`).
    ///
    /// May be less than the image's voxel count if some values fell
    /// outside `[min, max]`.
    #[inline]
    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Bin width Δw = (max − min) / bins.
    ///
    /// Returns 0 if `max == min` (degenerate range; all in-range voxels
    /// fall into bin 0).
    #[inline]
    pub fn bin_width(&self) -> f32 {
        (self.max - self.min) / self.bins as f32
    }
}

/// Compute the intensity histogram of `image` over `[min, max]` with
/// `bins` equal-width bins.
///
/// `bins` must be ≥ 1. `min` must be strictly less than `max`. Voxels
/// outside `[min, max]` are excluded.
///
/// # Examples
///
/// ```ignore
/// let img = Image::<f32, 3>::from_vec_f32([2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])?;
/// let h = histogram(&img, 0.0, 7.0, 7);
/// assert_eq!(h.counts, vec![1, 1, 1, 1, 1, 1, 2]);
/// // last bin is inclusive of max=7.0
/// ```
///
/// # Panics
///
/// Panics if `bins == 0` or `min >= max`.
pub fn histogram<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
    min: f32,
    max: f32,
    bins: usize,
) -> Histogram {
    assert!(bins >= 1, "histogram: bins must be ≥ 1, got {}", bins);
    assert!(
        min < max,
        "histogram: min must be strictly less than max (min={}, max={})",
        min,
        max
    );

    let (vals, _) = extract_vec_infallible(image);
    let slice: &[f32] = &vals;
    histogram_from_slice(slice, min, max, bins)
}

/// Slice-level histogram: zero domain logic, public for callers that
/// already have borrowed f32 storage.
pub fn histogram_from_slice(slice: &[f32], min: f32, max: f32, bins: usize) -> Histogram {
    assert!(bins >= 1, "histogram_from_slice: bins must be ≥ 1");
    assert!(
        min < max,
        "histogram_from_slice: min must be strictly less than max"
    );

    let mut counts = vec![0_usize; bins];

    // Δw = (max − min) / bins; bin index = floor((v − min) / Δw).
    // We multiply by `inv_dw = bins / (max − min)` for one division
    // outside the hot loop.
    let inv_dw = bins as f32 / (max - min);
    let last = bins - 1;

    for &v in slice {
        if v < min || v > max {
            continue;
        }
        // (v − min) · inv_dw ∈ [0, bins] because v ∈ [min, max].
        let raw = (v - min) * inv_dw;
        let mut k = raw as usize;
        // Clamp floating-point edge cases: v == max → raw == bins → k == bins
        // → must collapse into last bin per scipy convention.
        if k > last {
            k = last;
        }
        counts[k] += 1;
    }

    Histogram {
        min,
        max,
        bins,
        counts,
    }
}

#[cfg(test)]
#[path = "tests_histogram.rs"]
mod tests;
