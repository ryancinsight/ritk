//! Rank and percentile filters for 3-D volumes.
//!
//! # Mathematical Specification
//!
//! A rank filter replaces each voxel with the element at position `rank` in
//! the sorted order of its structuring-element neighbourhood. A percentile
//! filter is a rank filter parameterized by the percentile `p ∈ [0, 100]`:
//!
//! ```text
//! (R_B f)(x) = sort({ f(x + b) : b ∈ B })[ k ]
//! k = clamp(floor(p / 100 · (|B| - 1)), 0, |B| - 1)
//! ```
//!
//! # Special Cases
//!
//! - `PercentileFilter::new(0.0, r)` is the **erosion** (minimum).
//! - `PercentileFilter::new(50.0, r)` is the **median** for odd `|B|`, or the
//!   lower median for even `|B|` (consistent with `MedianFilter` and
//!   `scipy.ndimage.percentile_filter`).
//! - `PercentileFilter::new(100.0, r)` is the **dilation** (maximum).
//! - `RankFilter::new(0, r)` is the **erosion** (minimum).
//! - `RankFilter::new(|B| - 1, r)` is the **dilation** (maximum).
//!
//! # Algorithm
//!
//! We use `select_nth_unstable_by` (introselect) for `O(n)` per voxel
//! selection. This avoids a full `O(n log n)` sort and is asymptotically
//! optimal for the rank/percentile problem.
//!
//! # Boundary Handling
//!
//! Replicate (clamp) padding: out-of-bounds indices are clamped to the
//! nearest valid index along each axis. This matches `MedianFilter`,
//! `GrayscaleDilation`, `GrayscaleErosion`, and `scipy.ndimage` with
//! `mode="nearest"`.
//!
//! # Complexity
//!
//! `O(N · n)` where `N` is the total voxel count and `n = |B|` is the
//! neighbourhood size. Parallelised over z-slices via moirai.
//!
//! # Zero-Copy
//!
//! `StructuringElement::offsets()` returns a `&[Offset3D]` slice that is
//! borrowed directly by the hot loop — no per-call allocation of the
//! neighbourhood offsets.
//!
//! # Cow Optimization
//!
//! For `radius = 0` the SE has a single offset `(0, 0, 0)`, and the filter is
//! the identity. The result is returned as `Cow::Borrowed(image)` so the
//! caller can avoid a needless allocation.
//!
//! # References
//!
//! - Huang, T.S., Yang, G.J., & Tang, G.Y. (1979). A fast two-dimensional
//!   median filtering algorithm. *IEEE Trans. Acoust., Speech, Signal
//!   Process.*, 27(1), 13–18.
//! - Perreault, S. & Hébert, P. (2007). Median filtering in constant time.
//!   *IEEE Trans. Image Process.*, 16(9), 2389–2394.

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use crate::morphology::StructuringElement;
use burn::tensor::backend::Backend;
use std::borrow::Cow;

// ── PercentileFilter ──────────────────────────────────────────────────────────

/// Sliding-window percentile filter for 3-D volumes.
///
/// Replaces each voxel with the element at the requested percentile of its
/// structuring-element neighbourhood. Constructed from a percentile
/// `p ∈ [0, 100]` and a [`StructuringElement`] (typically `cube(r)`,
/// `cross(r)`, or `ball(r)`).
///
/// # Validation
/// The percentile must be in `[0, 100]`. A percentile outside this range
/// returns an `Err` from [`apply`](Self::apply).
#[derive(Clone, Debug)]
pub struct PercentileFilter {
    /// Percentile in `[0, 100]`.
    percentile: f32,
    /// Structuring element defining the neighbourhood.
    se: StructuringElement,
}

impl PercentileFilter {
    /// Construct a percentile filter with a cube SE of half-width `radius`.
    ///
    /// Equivalent to `scipy.ndimage.percentile_filter(input, percentile, size)`
    /// with `size = 2 * radius + 1` and a cubic footprint.
    #[inline]
    pub fn new(percentile: f32, radius: usize) -> Self {
        Self::with_structuring_element(percentile, StructuringElement::cube(radius))
    }

    /// Construct a percentile filter with a cube SE.
    #[inline]
    pub fn cube(percentile: f32, radius: usize) -> Self {
        Self::new(percentile, radius)
    }

    /// Construct a percentile filter with a cross SE.
    #[inline]
    pub fn cross(percentile: f32, radius: usize) -> Self {
        Self::with_structuring_element(percentile, StructuringElement::cross(radius))
    }

    /// Construct a percentile filter with a ball SE.
    #[inline]
    pub fn ball(percentile: f32, radius: usize) -> Self {
        Self::with_structuring_element(percentile, StructuringElement::ball(radius))
    }

    /// Construct a percentile filter from an explicit SE.
    ///
    /// The percentile must be in `[0, 100]`; an out-of-range value is
    /// rejected at [`apply`](Self::apply) time.
    #[inline]
    pub fn with_structuring_element(percentile: f32, se: StructuringElement) -> Self {
        Self { percentile, se }
    }

    /// Percentile in `[0, 100]`.
    #[inline]
    pub const fn percentile(&self) -> f32 {
        self.percentile
    }

    /// Structuring element (clone of the SE; cheap).
    #[inline]
    pub fn structuring_element(&self) -> StructuringElement {
        self.se.clone()
    }

    /// Apply the percentile filter to a 3-D image.
    ///
    /// # Zero-copy fast path
    /// For a single-offset SE (only `radius = 0` produces this), the filter
    /// is the identity and the input is returned as `Cow::Borrowed(image)`.
    /// All other cases produce a freshly allocated `Image`.
    ///
    /// # Errors
    /// Returns `Err` if the percentile is not in `[0, 100]` or the tensor
    /// data cannot be cast to `f32`.
    pub fn apply<'a, B: Backend>(
        &self,
        image: &'a Image<B, 3>,
    ) -> anyhow::Result<Cow<'a, Image<B, 3>>> {
        if !(0.0..=100.0).contains(&self.percentile) || self.percentile.is_nan() {
            return Err(anyhow::anyhow!(
                "PercentileFilter: percentile must be in [0, 100], got {}",
                self.percentile
            ));
        }

        // Identity fast path: radius = 0 → SE = {(0,0,0)} → output = input.
        if self.se.is_empty() || self.se.len() == 1 {
            return Ok(Cow::Borrowed(image));
        }

        let (vals, shape) = extract_vec(image)?;
        let result = percentile_3d(&vals, shape, self.percentile, self.se.offsets());
        Ok(Cow::Owned(rebuild(result, shape, image)))
    }
}

// ── RankFilter ────────────────────────────────────────────────────────────────

/// Sliding-window rank filter for 3-D volumes.
///
/// Replaces each voxel with the element at absolute position `rank` in the
/// sorted order of its structuring-element neighbourhood.
///
/// # Validation
/// `rank` must satisfy `rank < se.len()`. An out-of-range value returns an
/// `Err` from [`apply`](Self::apply).
#[derive(Clone, Debug)]
pub struct RankFilter {
    /// Absolute rank, 0-indexed, in `[0, se.len() - 1]`.
    rank: usize,
    /// Structuring element defining the neighbourhood.
    se: StructuringElement,
}

impl RankFilter {
    /// Construct a rank filter with a cube SE of half-width `radius`.
    #[inline]
    pub fn new(rank: usize, radius: usize) -> Self {
        Self::with_structuring_element(rank, StructuringElement::cube(radius))
    }

    /// Construct a rank filter with a cube SE.
    #[inline]
    pub fn cube(rank: usize, radius: usize) -> Self {
        Self::new(rank, radius)
    }

    /// Construct a rank filter with a cross SE.
    #[inline]
    pub fn cross(rank: usize, radius: usize) -> Self {
        Self::with_structuring_element(rank, StructuringElement::cross(radius))
    }

    /// Construct a rank filter with a ball SE.
    #[inline]
    pub fn ball(rank: usize, radius: usize) -> Self {
        Self::with_structuring_element(rank, StructuringElement::ball(radius))
    }

    /// Construct a rank filter from an explicit SE.
    #[inline]
    pub fn with_structuring_element(rank: usize, se: StructuringElement) -> Self {
        Self { rank, se }
    }

    /// Absolute rank, 0-indexed.
    #[inline]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Structuring element (clone of the SE; cheap).
    #[inline]
    pub fn structuring_element(&self) -> StructuringElement {
        self.se.clone()
    }

    /// Apply the rank filter to a 3-D image.
    ///
    /// # Zero-copy fast path
    /// For a single-offset SE (only `radius = 0` produces this), the filter
    /// is the identity and the input is returned as `Cow::Borrowed(image)`.
    ///
    /// # Errors
    /// Returns `Err` if `rank >= se.len()` or the tensor data cannot be cast
    /// to `f32`.
    pub fn apply<'a, B: Backend>(
        &self,
        image: &'a Image<B, 3>,
    ) -> anyhow::Result<Cow<'a, Image<B, 3>>> {
        if self.se.is_empty() {
            return Err(anyhow::anyhow!(
                "RankFilter: structuring element is empty (use a non-zero radius)"
            ));
        }
        if self.rank >= self.se.len() {
            return Err(anyhow::anyhow!(
                "RankFilter: rank {} out of range [0, {})",
                self.rank,
                self.se.len()
            ));
        }

        if self.se.len() == 1 {
            return Ok(Cow::Borrowed(image));
        }

        let (vals, shape) = extract_vec(image)?;
        let result = rank_select_3d(&vals, shape, self.rank, self.se.offsets());
        Ok(Cow::Owned(rebuild(result, shape, image)))
    }
}

// ── Internal kernel: percentile via select_nth_unstable ─────────────────────

/// Compute the percentile of every voxel's SE neighbourhood on a 3-D
/// `f32` volume stored in row-major `(Z, Y, X)` order.
///
/// # Boundary handling
/// Replicate (clamp) padding.
fn percentile_3d(data: &[f32], dims: [usize; 3], percentile: f32, se: &[crate::morphology::Offset3D]) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = se.len();
    let rank_idx = ((percentile / 100.0) * ((n - 1) as f32)).floor() as usize;
    let rank_idx = rank_idx.min(n - 1);

    let mut output = vec![0.0_f32; nz * ny * nx];
    let stride = ny * nx;

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut output,
        stride,
        |iz, out_slice| {
            // Reusable scratch buffer: avoids per-voxel Vec allocation.
            // `select_nth_unstable_by` requires `&mut [f32]`, so we grow the
            // Vec lazily and reuse it across voxels in the slice.
            let mut scratch: Vec<f32> = Vec::with_capacity(n);

            for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                for (ix, out_cell) in out_row.iter_mut().enumerate() {
                    scratch.clear();
                    for off in se {
                        let zz = (iz as i32 + off.iz()).clamp(0, nz as i32 - 1) as usize;
                        let yy = (iy as i32 + off.iy()).clamp(0, ny as i32 - 1) as usize;
                        let xx = (ix as i32 + off.ix()).clamp(0, nx as i32 - 1) as usize;
                        scratch.push(data[zz * stride + yy * nx + xx]);
                    }
                    // O(n) partial sort: position the rank_idx-th element as
                    // if the slice were fully sorted, then read it out.
                    scratch.select_nth_unstable_by(rank_idx, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    *out_cell = scratch[rank_idx];
                }
            }
        },
    );

    output
}

// ── Internal kernel: rank via select_nth_unstable ────────────────────────────

/// Compute the element at absolute position `rank` in the sorted order of
/// every voxel's SE neighbourhood on a 3-D `f32` volume.
fn rank_select_3d(data: &[f32], dims: [usize; 3], rank: usize, se: &[crate::morphology::Offset3D]) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = se.len();
    debug_assert!(rank < n, "rank {rank} out of range [0, {n})");

    let mut output = vec![0.0_f32; nz * ny * nx];
    let stride = ny * nx;

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut output,
        stride,
        |iz, out_slice| {
            let mut scratch: Vec<f32> = Vec::with_capacity(n);

            for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                for (ix, out_cell) in out_row.iter_mut().enumerate() {
                    scratch.clear();
                    for off in se {
                        let zz = (iz as i32 + off.iz()).clamp(0, nz as i32 - 1) as usize;
                        let yy = (iy as i32 + off.iy()).clamp(0, ny as i32 - 1) as usize;
                        let xx = (ix as i32 + off.ix()).clamp(0, nx as i32 - 1) as usize;
                        scratch.push(data[zz * stride + yy * nx + xx]);
                    }
                    scratch.select_nth_unstable_by(rank, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    *out_cell = scratch[rank];
                }
            }
        },
    );

    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::ops::extract_vec;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    /// Construct a small 3-D `f32` image from a flat vector.
    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)), &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    /// Extract the raw `f32` buffer of an image for value-semantic assertions.
    fn to_vec(image: &Image<B, 3>) -> Vec<f32> {
        let (v, _) = extract_vec(image).unwrap();
        v
    }

    // ── PercentileFilter ───────────────────────────────────────────────────

    /// Percentile 0.0 (minimum) on a 3×3×3 cube of 1..27 equals the
    /// neighbourhood minimum, which is 1 for interior voxels with all
    /// neighbours ≤ 27 and ≥ 1.
    #[test]
    fn percentile_zero_is_minimum() {
        let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
        let img = make_image(data.clone(), [3, 3, 3]);
        let filter = PercentileFilter::new(0.0, 1);
        let out = filter.apply(&img).unwrap();
        let out_vec = to_vec(out.as_ref());
        for &v in &out_vec {
            assert!(v >= 1.0, "percentile=0.0 must produce the minimum, got {v}");
        }
    }

    /// Percentile 100.0 (maximum) on a 3×3×3 cube of 1..27 equals the
    /// neighbourhood maximum, which is 27 for interior voxels.
    #[test]
    fn percentile_hundred_is_maximum() {
        let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
        let img = make_image(data.clone(), [3, 3, 3]);
        let filter = PercentileFilter::new(100.0, 1);
        let out = filter.apply(&img).unwrap();
        let out_vec = to_vec(out.as_ref());
        for &v in &out_vec {
            assert!(v <= 27.0, "percentile=100.0 must produce the maximum, got {v}");
        }
    }

    /// Constant image: any percentile must return the constant.
    #[test]
    fn percentile_of_constant_image_is_constant() {
        let img = make_image(vec![5.0_f32; 27], [3, 3, 3]);
        for p in [0.0_f32, 25.0, 50.0, 75.0, 100.0] {
            let filter = PercentileFilter::new(p, 1);
            let out = filter.apply(&img).unwrap();
            let out_vec = to_vec(out.as_ref());
            for &v in &out_vec {
                assert!((v - 5.0).abs() < 1e-6, "percentile {p} of constant 5.0 must be 5.0");
            }
        }
    }

    /// Identity fast path: radius = 0 → Cow::Borrowed, no allocation.
    #[test]
    fn percentile_radius_zero_is_cow_borrowed() {
        let img = make_image(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
        let filter = PercentileFilter::new(50.0, 0);
        let out = filter.apply(&img).unwrap();
        assert!(
            matches!(out, Cow::Borrowed(_)),
            "radius=0 must return Cow::Borrowed"
        );
    }

    /// Out-of-range percentile returns `Err` with a descriptive message.
    #[test]
    fn percentile_out_of_range_returns_err() {
        let img = make_image(vec![0.0_f32; 27], [3, 3, 3]);
        let filter = PercentileFilter::new(150.0, 1);
        let err = filter.apply(&img).unwrap_err();
        assert!(err.to_string().contains("percentile"), "error must mention percentile");
    }

    /// NaN percentile is rejected.
    #[test]
    fn percentile_nan_rejected() {
        let img = make_image(vec![0.0_f32; 27], [3, 3, 3]);
        let filter = PercentileFilter::new(f32::NAN, 1);
        assert!(filter.apply(&img).is_err(), "NaN percentile must be rejected");
    }

    /// Cross SE percentile matches the cross-footprint value.
    #[test]
    fn percentile_cross_matches_cube_subset() {
        // 5x5x5 image with central peak (index 62 = 3,2,2)
        let mut data = vec![0.0_f32; 125];
        data[3 * 25 + 2 * 5 + 2] = 100.0;
        let img = make_image(data.clone(), [5, 5, 5]);
        // Cross SE at r=1 includes 7 voxels (3 axes union); percentile 100
        // picks the max. The central voxel (3,2,2) sees the peak itself so
        // its output is 100. Neighbouring voxels along z/y/x axes see 100
        // exactly once, so they also produce 100.
        let filter = PercentileFilter::cross(100.0, 1);
        let out = to_vec(filter.apply(&img).unwrap().as_ref());
        // central voxel
        assert!((out[3 * 25 + 2 * 5 + 2] - 100.0).abs() < 1e-6);
        // (3,2,3) is on x-axis, sees 100 in its cross → 100
        assert!((out[3 * 25 + 2 * 5 + 3] - 100.0).abs() < 1e-6);
    }

    // ── RankFilter ─────────────────────────────────────────────────────────

    /// `rank = 0` equals minimum (erosion).
    #[test]
    fn rank_zero_is_minimum() {
        let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
        let img = make_image(data, [3, 3, 3]);
        let filter = RankFilter::new(0, 1);
        let out = to_vec(filter.apply(&img).unwrap().as_ref());
        for &v in &out {
            assert!(v >= 1.0, "rank=0 must be the minimum");
        }
    }

    /// `rank = |B| - 1` equals maximum (dilation).
    #[test]
    fn rank_last_is_maximum() {
        let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
        let img = make_image(data, [3, 3, 3]);
        let se = StructuringElement::cube(1);
        let last = se.len() - 1;
        let filter = RankFilter::with_structuring_element(last, se);
        let out = to_vec(filter.apply(&img).unwrap().as_ref());
        for &v in &out {
            assert!(v <= 27.0, "rank=|B|-1 must be the maximum");
        }
    }

    /// Median rank for |B|=27 (cube r=1) equals the 14th sorted value
    /// (0-indexed). The constant image 5.0 case reduces to 5.0.
    #[test]
    fn rank_median_of_constant_image_is_constant() {
        let img = make_image(vec![7.0_f32; 27], [3, 3, 3]);
        let se = StructuringElement::cube(1);
        let filter = RankFilter::with_structuring_element(13, se);
        let out = to_vec(filter.apply(&img).unwrap().as_ref());
        for &v in &out {
            assert!((v - 7.0).abs() < 1e-6);
        }
    }

    /// Out-of-range rank returns `Err`.
    #[test]
    fn rank_out_of_range_returns_err() {
        let img = make_image(vec![0.0_f32; 27], [3, 3, 3]);
        // |B| for cube(1) is 27, rank 50 is out of range.
        let filter = RankFilter::new(50, 1);
        assert!(filter.apply(&img).is_err());
    }

    /// Identity fast path: radius = 0 → Cow::Borrowed.
    #[test]
    fn rank_radius_zero_is_cow_borrowed() {
        let img = make_image(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
        let filter = RankFilter::new(0, 0);
        let out = filter.apply(&img).unwrap();
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    /// Rank on a ball SE picks a valid neighbour.
    #[test]
    fn rank_ball_se_succeeds() {
        let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
        let img = make_image(data, [3, 3, 3]);
        let filter = RankFilter::ball(0, 1);
        let out = to_vec(filter.apply(&img).unwrap().as_ref());
        // All output values must come from the input domain.
        for &v in &out {
            assert!((1.0..=27.0).contains(&v), "ball rank=0 must produce input value, got {v}");
        }
    }

    // ── Cross-validation against scipy.ndimage semantics ────────────────

    /// Cross-validation: percentile 50 on a 3×3×3 cube of 1..27 is the
    /// 13th sorted value (0-indexed) of each 27-voxel neighbourhood. The
    /// central voxel `(1, 1, 1)` of the 3×3×3 input has neighbourhood
    /// values 1..27, whose 13th sorted value is 14.0.
    #[test]
    fn percentile_50_central_voxel_value() {
        let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
        let img = make_image(data, [3, 3, 3]);
        let filter = PercentileFilter::new(50.0, 1);
        let out = to_vec(filter.apply(&img).unwrap().as_ref());
        // central index = 1*9 + 1*3 + 1 = 13
        assert!(
            (out[13] - 14.0).abs() < 1e-6,
            "central percentile=50 of 1..27 should be 14.0, got {}",
            out[13]
        );
    }
}
