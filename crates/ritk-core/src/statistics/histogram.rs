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

use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use burn::tensor::backend::Backend;

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
    image: &Image<B, D>,
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
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
        let n = data.len();
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

    fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    // ── Positive tests ────────────────────────────────────────────────────────

    #[test]
    fn histogram_3d_uniform_distribution() {
        // 8 voxels, range [0, 8), 8 bins → exactly one per bin
        let img = make_image_3d(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 2, 2]);
        let h = histogram(&img, 0.0, 8.0, 8);
        assert_eq!(h.counts, vec![1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(h.total(), 8);
    }

    #[test]
    fn histogram_3d_last_bin_inclusive_of_max() {
        // Two voxels at v=7.0, range [0,7], 7 bins
        // Bin edges: [0,1), [1,2), [2,3), [3,4), [4,5), [5,6), [6,7]
        let img = make_image_3d(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 2, 2]);
        let h = histogram(&img, 0.0, 7.0, 7);
        assert_eq!(h.counts, vec![1, 1, 1, 1, 1, 1, 2]);
    }

    #[test]
    fn histogram_3d_single_bin_collects_all_in_range() {
        // 1 bin: [0, 10] → all 8 in-range voxels go into bin 0
        let img = make_image_3d(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 2, 2]);
        let h = histogram(&img, 0.0, 10.0, 1);
        assert_eq!(h.counts, vec![8]);
    }

    #[test]
    fn histogram_3d_values_outside_range_excluded() {
        // Range [0, 5] excludes 5.0+ (last bin inclusive of 5.0 only).
        // v=5 → in (last) bin; v=6,7 → excluded.
        let img = make_image_3d(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 2, 2]);
        let h = histogram(&img, 0.0, 5.0, 5);
        // Bin 0 [0,1): {0.0} → 1
        // Bin 1 [1,2): {1.0} → 1
        // Bin 2 [2,3): {2.0} → 1
        // Bin 3 [3,4): {3.0} → 1
        // Bin 4 [4,5]: {4.0, 5.0} → 2
        // 6.0, 7.0 excluded
        assert_eq!(h.counts, vec![1, 1, 1, 1, 2]);
        assert_eq!(h.total(), 6);
    }

    #[test]
    fn histogram_1d_constant_lands_in_first_bin() {
        // All values = 5.0; range [0, 10], 5 bins. v=5.0 → bin 2.
        let img = make_image_1d(vec![5.0; 10]);
        let h = histogram(&img, 0.0, 10.0, 5);
        assert_eq!(h.counts, vec![0, 0, 10, 0, 0]);
    }

    #[test]
    fn histogram_3d_constant_at_min_lands_in_bin_zero() {
        // v == min → bin 0
        let img = make_image_3d(vec![3.0; 8], [2, 2, 2]);
        let h = histogram(&img, 3.0, 5.0, 4);
        // Δw = 0.5, bin 0 = [3.0, 3.5) ... but v=3.0 hits bin 0
        assert_eq!(h.counts[0], 8);
        assert_eq!(h.total(), 8);
    }

    #[test]
    fn histogram_3d_constant_at_max_lands_in_last_bin() {
        // v == max → last bin (inclusive convention)
        let img = make_image_3d(vec![7.0; 8], [2, 2, 2]);
        let h = histogram(&img, 0.0, 7.0, 7);
        assert_eq!(h.counts[6], 8);
        assert_eq!(h.total(), 8);
    }

    #[test]
    fn histogram_3d_negative_range() {
        // Range [-10, 0], 5 bins
        let img = make_image_3d(
            vec![-10.0, -7.5, -5.0, -2.5, 0.0, -9.0, -1.0, -100.0],
            [2, 2, 2],
        );
        // -100.0 is excluded (below min)
        let h = histogram(&img, -10.0, 0.0, 5);
        // Bin 0 [-10,-8): {-10.0, -9.0} → 2
        // Bin 1 [-8, -6): {-7.5} → 1
        // Bin 2 [-6, -4): {-5.0} → 1
        // Bin 3 [-4, -2): {-2.5} → 1
        // Bin 4 [-2,  0]: {-1.0, 0.0} → 2
        assert_eq!(h.counts, vec![2, 1, 1, 1, 2]);
        assert_eq!(h.total(), 7);
    }

    // ── Properties ────────────────────────────────────────────────────────────

    #[test]
    fn histogram_bin_width_is_correct() {
        let img = make_image_3d(vec![0.0; 1], [1, 1, 1]);
        let h = histogram(&img, 0.0, 10.0, 4);
        assert!((h.bin_width() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn histogram_total_equals_in_range_voxel_count() {
        let img = make_image_3d(vec![0.0, 1.0, 2.0, 100.0, -100.0, 5.0, 6.0, 7.0], [2, 2, 2]);
        // Range [0, 10]: 100.0 and -100.0 excluded → 6 in-range
        let h = histogram(&img, 0.0, 10.0, 10);
        assert_eq!(h.total(), 6);
    }

    #[test]
    fn histogram_values_outside_range_yield_zero_counts() {
        // All voxels = 20.0, range [0, 10] → all excluded.
        let img = make_image_3d(vec![20.0_f32; 8], [2, 2, 2]);
        let h = histogram(&img, 0.0, 10.0, 5);
        assert_eq!(h.counts, vec![0, 0, 0, 0, 0]);
        assert_eq!(h.total(), 0);
    }

    // ── Negative / boundary ───────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "bins must be ≥ 1")]
    fn histogram_zero_bins_panics() {
        let img = make_image_3d(vec![1.0; 1], [1, 1, 1]);
        let _ = histogram(&img, 0.0, 10.0, 0);
    }

    #[test]
    #[should_panic(expected = "min must be strictly less than max")]
    fn histogram_min_equal_max_panics() {
        let img = make_image_3d(vec![1.0; 1], [1, 1, 1]);
        let _ = histogram(&img, 5.0, 5.0, 4);
    }

    #[test]
    #[should_panic(expected = "min must be strictly less than max")]
    fn histogram_min_greater_than_max_panics() {
        let img = make_image_3d(vec![1.0; 1], [1, 1, 1]);
        let _ = histogram(&img, 10.0, 0.0, 4);
    }

    // ── D-type genericity ─────────────────────────────────────────────────────

    #[test]
    fn histogram_works_on_1d_image() {
        let img = make_image_1d(vec![0.5, 1.5, 2.5, 3.5, 4.5]);
        // Range [0, 5], 5 bins, Δw=1
        let h = histogram(&img, 0.0, 5.0, 5);
        // Each value lands in its own bin; 4.5 is the last bin's interior.
        assert_eq!(h.counts, vec![1, 1, 1, 1, 1]);
    }
}
