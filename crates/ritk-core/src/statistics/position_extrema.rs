//! Position-of-extrema queries: indices of minimum and maximum voxel values.
//!
//! # Mathematical Specification
//!
//! For a D-dimensional image `I` of shape `[d_0, d_1, …, d_{D-1}]` stored in
//! row-major order, the **argmin** is the unique (or first) index vector
//!
//! p_min = argmin_{i ∈ \[0, n\)} I.flat\[i\]
//!
//! which is then converted to multi-index coordinates via
//!
//!   i_0 = p_min / (d_1 · d_2 · … · d_{D-1})
//!   i_1 = (p_min mod (d_1 · d_2 · … · d_{D-1})) / (d_2 · … · d_{D-1})
//!   …
//!
//! and likewise for `argmax`. Ties resolve to the lowest flat index, matching
//! `Iterator::position` semantics and the convention used by
//! `scipy.ndimage.minimum_position` / `maximum_position`.
//!
//! # Complexity
//!
//! O(n) where n = ∏ d_k. One pass over the data, no allocation beyond the
//! 4-byte (or 8-byte) running extremum and the returned `[usize; D]`.

use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Return the multi-index of the **minimum** voxel value.
///
/// Returns `Some([i_0, i_1, …, i_{D-1}])` for the lexicographically-first
/// voxel achieving the minimum, or `None` if the image is empty.
///
/// Ties resolve to the lowest flat (row-major) index, matching
/// `scipy.ndimage.minimum_position` and `Iterator::position`.
///
/// # Examples
///
/// ```ignore
/// let img = Image::<f32, 3>::from_vec_f32([2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, -1.0, 6.0, 7.0, 8.0])?;
/// assert_eq!(minimum_position(&img), Some([1, 0, 0]));
/// ```
pub fn minimum_position<B: Backend, const D: usize>(image: &Image<B, D>) -> Option<[usize; D]> {
    let (vals, dims) = extract_vec_infallible(image);
    let slice: &[f32] = &vals;
    argmin_position(slice, dims)
}

/// Return the multi-index of the **maximum** voxel value.
///
/// Returns `Some([i_0, i_1, …, i_{D-1}])` for the lexicographically-first
/// voxel achieving the maximum, or `None` if the image is empty.
///
/// Ties resolve to the lowest flat (row-major) index, matching
/// `scipy.ndimage.maximum_position` and `Iterator::position`.
pub fn maximum_position<B: Backend, const D: usize>(image: &Image<B, D>) -> Option<[usize; D]> {
    let (vals, dims) = extract_vec_infallible(image);
    let slice: &[f32] = &vals;
    argmax_position(slice, dims)
}

/// Slice-level argmin: returns the multi-index of the minimum.
///
/// # Panics
///
/// Panics if `slice.len() != dims.iter().product()`.
fn argmin_position<const D: usize>(slice: &[f32], dims: [usize; D]) -> Option<[usize; D]> {
    if slice.is_empty() {
        return None;
    }
    let mut best_idx: usize = 0;
    let mut best_val: f32 = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        // NaN comparison: keep current best (NaN propagates as not-less).
        if v < best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some(flat_to_multi(best_idx, dims))
}

/// Slice-level argmax: returns the multi-index of the maximum.
///
/// # Panics
///
/// Panics if `slice.len() != dims.iter().product()`.
fn argmax_position<const D: usize>(slice: &[f32], dims: [usize; D]) -> Option<[usize; D]> {
    if slice.is_empty() {
        return None;
    }
    let mut best_idx: usize = 0;
    let mut best_val: f32 = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some(flat_to_multi(best_idx, dims))
}

/// Convert a flat row-major index to a multi-index `[i_0, i_1, …, i_{D-1}]`.
///
/// # Layout
///
/// Row-major: `i = i_0 · (d_1 · d_2 · …) + i_1 · (d_2 · …) + … + i_{D-1}`.
///
/// This matches the layout produced by `Image::from_vec_f32` and
/// `extract_vec_infallible`.
fn flat_to_multi<const D: usize>(flat: usize, dims: [usize; D]) -> [usize; D] {
    let mut out = [0_usize; D];
    let mut remaining = flat;
    for (k, out_k) in out.iter_mut().enumerate() {
        // Stride for axis k: ∏_{j > k} dims[j]
        let mut stride: usize = 1;
        for &d in dims.iter().skip(k + 1) {
            stride *= d;
        }
        *out_k = remaining / stride;
        remaining %= stride;
    }
    out
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

    // ── 1-D ────────────────────────────────────────────────────────────────────

    #[test]
    fn minimum_position_1d_unique() {
        let img = make_image_1d(vec![3.0, 1.0, 4.0, 1.5, 9.0]);
        assert_eq!(minimum_position(&img), Some([1]));
    }

    #[test]
    fn maximum_position_1d_unique() {
        let img = make_image_1d(vec![3.0, 1.0, 4.0, 1.5, 9.0]);
        assert_eq!(maximum_position(&img), Some([4]));
    }

    #[test]
    fn minimum_position_1d_tie_breaks_to_lowest_index() {
        let img = make_image_1d(vec![2.0, 1.0, 3.0, 1.0, 4.0]);
        // Two minima at indices 1 and 3; lowest wins.
        assert_eq!(minimum_position(&img), Some([1]));
    }

    #[test]
    fn maximum_position_1d_tie_breaks_to_lowest_index() {
        let img = make_image_1d(vec![5.0, 9.0, 1.0, 9.0, 4.0]);
        assert_eq!(maximum_position(&img), Some([1]));
    }

    #[test]
    fn minimum_position_1d_at_index_zero() {
        let img = make_image_1d(vec![-5.0, 1.0, 2.0, 3.0]);
        assert_eq!(minimum_position(&img), Some([0]));
    }

    // ── 3-D row-major layout ───────────────────────────────────────────────────

    #[test]
    fn minimum_position_3d_simple() {
        // 2×2×2 image: min is at (iz=1, iy=0, ix=1) → flat index = 1*4 + 0*2 + 1 = 5
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, -7.0, 7.0, 8.0];
        let img = make_image_3d(data, [2, 2, 2]);
        assert_eq!(minimum_position(&img), Some([1, 0, 1]));
    }

    #[test]
    fn maximum_position_3d_simple() {
        // 2×2×2 image: max is 99.0 at flat index 6 = (1, 1, 0)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, -7.0, 99.0, 8.0];
        let img = make_image_3d(data, [2, 2, 2]);
        assert_eq!(maximum_position(&img), Some([1, 1, 0]));
    }

    #[test]
    fn minimum_position_3d_first_voxel() {
        // Min at flat index 0
        let mut data = vec![1.0_f32; 27];
        data[0] = -100.0;
        let img = make_image_3d(data, [3, 3, 3]);
        assert_eq!(minimum_position(&img), Some([0, 0, 0]));
    }

    #[test]
    fn minimum_position_3d_last_voxel() {
        // Min at flat index 26 = (2, 2, 2)
        let mut data = vec![1.0_f32; 27];
        data[26] = -100.0;
        let img = make_image_3d(data, [3, 3, 3]);
        assert_eq!(minimum_position(&img), Some([2, 2, 2]));
    }

    #[test]
    fn minimum_position_3d_tie_breaks_to_lowest_flat() {
        // Two minima at flat 3 = (0, 1, 0) and flat 12 = (1, 1, 0)
        // Lowest flat wins: 3 → (0, 1, 0)
        let mut data = vec![5.0_f32; 27];
        data[3] = -10.0; // (0, 1, 0)
        data[12] = -10.0; // (1, 1, 0)
        let img = make_image_3d(data, [3, 3, 3]);
        assert_eq!(minimum_position(&img), Some([0, 1, 0]));
    }

    #[test]
    fn minimum_position_3d_constant() {
        let img = make_image_3d(vec![7.0_f32; 27], [3, 3, 3]);
        // All tied at value 7 → lowest flat index wins → (0, 0, 0)
        assert_eq!(minimum_position(&img), Some([0, 0, 0]));
        assert_eq!(maximum_position(&img), Some([0, 0, 0]));
    }

    #[test]
    fn maximum_position_3d_single_voxel() {
        let img = make_image_3d(vec![42.0_f32], [1, 1, 1]);
        assert_eq!(minimum_position(&img), Some([0, 0, 0]));
        assert_eq!(maximum_position(&img), Some([0, 0, 0]));
    }

    #[test]
    fn minimum_position_3d_negative_values() {
        let data = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let img = make_image_3d(data, [2, 2, 2]);
        // Min is -8 at flat 7 = (1, 1, 1)
        assert_eq!(minimum_position(&img), Some([1, 1, 1]));
    }

    #[test]
    fn flat_to_multi_round_trip() {
        // Spot-check: every flat index in a 2×3×4 volume maps correctly.
        let dims = [2_usize, 3, 4];
        for flat in 0..(2 * 3 * 4) {
            let m = flat_to_multi(flat, dims);
            let mut recovered = 0_usize;
            for k in 0..3 {
                let mut stride = 1;
                for j in (k + 1)..3 {
                    stride *= dims[j];
                }
                recovered += m[k] * stride;
            }
            assert_eq!(recovered, flat, "flat={} → {:?} → {}", flat, m, recovered);
        }
    }
}
