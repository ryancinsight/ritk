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

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

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
#[path = "tests_position_extrema.rs"]
mod tests;
