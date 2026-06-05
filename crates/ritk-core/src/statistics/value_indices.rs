//! Per-value index map: for each distinct voxel value, the multi-indices
//! where it occurs.
//!
//! # Mathematical Specification
//!
//! Given a D-dimensional image `I` of shape `[d_0, d_1, …, d_{D-1}]`
//! stored in row-major order, the **value-index map** is a dictionary
//!
//!   VI = { v : M_v : v ∈ distinct(I) }
//!
//! where `M_v` is the set of multi-indices `[i_0, i_1, …, i_{D-1}]` such
//! that `I[i_0, i_1, …, i_{D-1}] = v`. The occurrences within each `M_v`
//! are sorted in **row-major** order (lexicographic on the index tuple),
//! matching `scipy.ndimage.value_indices` and the convention used by
//! `Iterator::position` for tie-breaking.
//!
//! The `ignore_value` parameter (when supplied) removes one value from
//! the output dictionary entirely, matching the scipy keyword
//! `ignore_value=None`.
//!
//! # Complexity
//!
//! O(n) where n = ∏ d_k. One pass over the data; per-voxel cost is one
//! `HashMap` lookup, one `flat_to_multi` conversion (O(D) where D is the
//! rank, typically 2–4), and one `Vec::push`.
//!
//! Space: O(n) in the worst case (one `usize` per multi-index, one entry
//! per distinct value), matching scipy's `dict[int, tuple[ndarray, ...]]`
//! return type.
//!
//! # Key type
//!
//! The dictionary key is a [`F32Key`] newtype around `f32` that compares
//! and hashes by **bit pattern** (transparent over `u32`). This is the
//! canonical Rust solution for using `f32` as a `HashMap` key (which
//! requires `Eq + Hash`, but `f32` cannot implement `Eq` due to `NaN`).
//!
//! Practical consequences for categorical/segmentation inputs:
//! - Bit-equal ±0.0 are **distinct** keys (0x00000000 vs 0x80000000).
//! - All `NaN` payloads collapse to a single key.
//! - For integer-valued f32 inputs (the dominant use case, as scipy
//!   itself requires integer arrays), there is no observable difference
//!   from mathematical equality.
//!
//! # Row-major layout
//!
//! Matches `Image::from_vec_f32` and `extract_vec_infallible`. A flat
//! index `p` corresponds to multi-index
//!
//!   p = i_0 · (d_1 · d_2 · …) + i_1 · (d_2 · …) + … + i_{D-1}.
//!
//! # SciPy Reference
//!
//! [`scipy.ndimage.value_indices`] (added in scipy 1.10.0). The default
//! `ignore_value=None` includes every distinct value in the output.

use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use burn::tensor::backend::Backend;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A `f32` keyed by its bit pattern, suitable for use as a `HashMap` key.
///
/// `f32` cannot implement `Eq` because `NaN != NaN`, but `HashMap`
/// requires `Eq + Hash` keys. This newtype uses the **bit pattern** of
/// the float as both the equality and hash source, which is the
/// canonical Rust solution (used throughout the standard library for
/// byte-oriented keys).
///
/// - **NaN**: all `NaN` payloads hash to the same value and compare
///   equal by bit pattern, so a `HashMap<F32Key, _>` treats all `NaN`
///   voxels as a single key. (Mathematical `NaN != NaN` semantics
///   cannot be represented in a `HashMap` without external tagging.)
/// - **±0.0**: distinct keys (`+0.0` is `0x00000000`, `-0.0` is
///   `0x80000000`). Categorical/segmentation inputs do not produce
///   signed zero, so this is observable only in pathological cases.
#[derive(Copy, Clone, Debug)]
pub struct F32Key(f32);

impl F32Key {
    /// Wrap a `f32` for use as a `HashMap` key.
    #[inline]
    pub const fn new(v: f32) -> Self {
        Self(v)
    }

    /// Recover the original `f32` value.
    #[inline]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl PartialEq for F32Key {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for F32Key {}

impl Hash for F32Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Per-value index map returned by [`value_indices`].
///
/// For each distinct voxel value (bit-equal) in the input image, stores
/// the list of multi-indices where that value occurs, in row-major
/// order. This is the Rust equivalent of scipy's
/// `dict[value, tuple[axis0_array, axis1_array, ...]]` return type —
/// the multi-index form is more compact (one `Vec<[usize; D]>` per
/// value vs D per-axis `Vec`s) and equally efficient to consume.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueIndices<const D: usize> {
    /// Maps each distinct voxel value to the multi-indices where it
    /// occurs, in row-major order.
    pub indices: HashMap<F32Key, Vec<[usize; D]>>,
}

impl<const D: usize> ValueIndices<D> {
    /// Total number of voxels accounted for across all distinct values.
    ///
    /// Equal to the image's voxel count when `ignore_value` was
    /// `None`, or `n − k` where `k` is the number of voxels equal to
    /// `ignore_value` (and `k` is dropped).
    #[inline]
    pub fn total(&self) -> usize {
        self.indices.values().map(|v| v.len()).sum()
    }

    /// Number of distinct voxel values in the map.
    #[inline]
    pub fn num_distinct(&self) -> usize {
        self.indices.len()
    }

    /// Number of voxels equal to `value` (by bit pattern).
    #[inline]
    pub fn len(&self, value: f32) -> usize {
        self.indices.get(&F32Key::new(value)).map_or(0, |v| v.len())
    }

    /// Look up the multi-indices where `value` occurs, by bit pattern.
    #[inline]
    pub fn get(&self, value: f32) -> Option<&[[usize; D]]> {
        self.indices.get(&F32Key::new(value)).map(Vec::as_slice)
    }

    /// `true` if the map contains no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Build the per-value index map for an image.
///
/// Equivalent to `scipy.ndimage.value_indices(arr, ignore_value=…)`:
/// returns a dictionary mapping each distinct voxel value to the
/// multi-indices where it occurs, in row-major order.
///
/// # Parameters
///
/// - `image`: any D-dimensional image backed by `f32` storage
///   (mirrors the contract of the other `ritk_core::statistics`
///   routines that use `extract_vec_infallible`).
/// - `ignore_value`: if `Some(v)`, voxels equal to `v` (by bit
///   pattern) are excluded from the output. `None` includes every
///   distinct value.
///
/// # Examples
///
/// ```ignore
/// // 3-D image with two distinct non-zero values.
/// let img = Image::<f32, 3>::from_vec_f32(
///     [2, 2, 2],
///     vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
/// )?;
/// let vi = value_indices(&img, None);
/// assert_eq!(vi.len(1.0), 4);
/// assert_eq!(vi.len(2.0), 4);
/// // 1.0 occurs at (0,0,0), (0,1,0), (1,0,0), (1,1,0) in row-major.
/// assert_eq!(vi.get(1.0), Some(&[[0,0,0], [0,1,0], [1,0,0], [1,1,0]]));
/// ```
///
/// # Panics
///
/// Panics if the underlying backend's tensor data cannot be extracted
/// as `Vec<f32>` (only possible with non-`f32` backends; see
/// `extract_vec_infallible`).
pub fn value_indices<B: Backend, const D: usize>(
    image: &Image<B, D>,
    ignore_value: Option<f32>,
) -> ValueIndices<D> {
    let (vals, dims) = extract_vec_infallible(image);

    let ignore_key = ignore_value.map(F32Key::new);
    let mut indices: HashMap<F32Key, Vec<[usize; D]>> = HashMap::new();

    for (flat, &v) in vals.iter().enumerate() {
        let key = F32Key(v);
        if Some(key) == ignore_key {
            continue;
        }
        indices
            .entry(key)
            .or_default()
            .push(flat_to_multi(flat, dims));
    }

    ValueIndices { indices }
}

/// Convert a flat row-major index to a multi-index `[i_0, i_1, …, i_{D-1}]`.
///
/// Row-major: `p = i_0 · (d_1 · d_2 · …) + i_1 · (d_2 · …) + … + i_{D-1}`.
///
/// Matches the layout produced by `Image::from_vec_f32` and
/// `extract_vec_infallible`. Equivalent to the private helper of the
/// same name in `position_extrema`; duplicated here to keep modules
/// independent (each is a leaf under the `statistics` bounded context).
#[inline]
fn flat_to_multi<const D: usize>(flat: usize, dims: [usize; D]) -> [usize; D] {
    let mut out = [0_usize; D];
    let mut remaining = flat;
    for (k, out_k) in out.iter_mut().enumerate() {
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

    fn make_image_2d(data: Vec<f32>, dims: [usize; 2]) -> Image<TestBackend, 2> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0]),
            Spacing::new([1.0, 1.0]),
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
    fn value_indices_1d_basic() {
        // [10, 20, 10, 30, 20]  →  10: [0,2]; 20: [1,4]; 30: [3]
        let img = make_image_1d(vec![10.0, 20.0, 10.0, 30.0, 20.0]);
        let vi = value_indices(&img, None);
        assert_eq!(vi.num_distinct(), 3);
        assert_eq!(vi.total(), 5);
        let exp_10: &[[usize; 1]] = &[[0], [2]];
        let exp_20: &[[usize; 1]] = &[[1], [4]];
        let exp_30: &[[usize; 1]] = &[[3]];
        assert_eq!(vi.get(10.0), Some(exp_10));
        assert_eq!(vi.get(20.0), Some(exp_20));
        assert_eq!(vi.get(30.0), Some(exp_30));
        assert_eq!(vi.get(99.0), None);
    }

    #[test]
    fn value_indices_1d_constant() {
        let img = make_image_1d(vec![7.0; 4]);
        let vi = value_indices(&img, None);
        assert_eq!(vi.num_distinct(), 1);
        assert_eq!(vi.total(), 4);
        let exp: &[[usize; 1]] = &[[0], [1], [2], [3]];
        assert_eq!(vi.get(7.0), Some(exp));
    }

    #[test]
    fn value_indices_1d_single_voxel() {
        let img = make_image_1d(vec![42.0]);
        let vi = value_indices(&img, None);
        assert_eq!(vi.num_distinct(), 1);
        let exp: &[[usize; 1]] = &[[0]];
        assert_eq!(vi.get(42.0), Some(exp));
    }

    #[test]
    fn value_indices_1d_ignore_value() {
        let img = make_image_1d(vec![1.0, 2.0, 1.0, 3.0, 1.0]);
        let vi = value_indices(&img, Some(1.0));
        assert_eq!(vi.num_distinct(), 2);
        assert_eq!(vi.total(), 2);
        assert_eq!(vi.get(1.0), None);
        let exp_2: &[[usize; 1]] = &[[1]];
        let exp_3: &[[usize; 1]] = &[[3]];
        assert_eq!(vi.get(2.0), Some(exp_2));
        assert_eq!(vi.get(3.0), Some(exp_3));
    }

    // ── 2-D (scipy docstring example) ─────────────────────────────────────────

    #[test]
    fn value_indices_2d_docstring_example() {
        // 6×6 array from the scipy.ndimage.value_indices docstring.
        //   [[2 2 2 0 0 3]
        //    [2 2 2 0 0 0]
        //    [0 0 1 1 0 0]
        //    [0 0 1 1 0 0]
        //    [0 0 0 0 1 0]
        //    [0 0 0 0 0 0]]
        let mut a = vec![0.0_f32; 36];
        // value 2 block: [0:2, 0:3] and [1, 0:3]
        for r in 0..2 {
            for c in 0..3 {
                a[r * 6 + c] = 2.0;
            }
        }
        // value 3: (0, 5)
        a[5] = 3.0;
        // value 1 block: [2:4, 2:4]
        for r in 2..4 {
            for c in 2..4 {
                a[r * 6 + c] = 1.0;
            }
        }
        // value 1: (4, 4)
        a[4 * 6 + 4] = 1.0;
        let img = make_image_2d(a, [6, 6]);
        let vi = value_indices(&img, None);

        assert_eq!(vi.num_distinct(), 4);
        assert_eq!(vi.total(), 36);

        // 1.0 at (2,2), (2,3), (3,2), (3,3), (4,4)
        let exp_1: &[[usize; 2]] = &[[2, 2], [2, 3], [3, 2], [3, 3], [4, 4]];
        assert_eq!(vi.get(1.0), Some(exp_1));
        // 2.0 at (0,0..3), (1,0..3) — row-major
        let exp_2: &[[usize; 2]] = &[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]];
        assert_eq!(vi.get(2.0), Some(exp_2));
        // 3.0 at (0, 5)
        let exp_3: &[[usize; 2]] = &[[0, 5]];
        assert_eq!(vi.get(3.0), Some(exp_3));
        // 0.0 fills the rest — verify count = 36 - 6 - 5 - 1 = 24
        assert_eq!(vi.len(0.0), 24);
    }

    #[test]
    fn value_indices_2d_ignore_value() {
        let mut a = vec![0.0_f32; 36];
        a[5] = 3.0;
        for r in 2..4 {
            for c in 2..4 {
                a[r * 6 + c] = 1.0;
            }
        }
        let img = make_image_2d(a, [6, 6]);
        let vi = value_indices(&img, Some(0.0));
        assert_eq!(vi.num_distinct(), 2);
        assert_eq!(vi.total(), 5);
        assert_eq!(vi.get(0.0), None);
        let exp_3: &[[usize; 2]] = &[[0, 5]];
        assert_eq!(vi.get(3.0), Some(exp_3));
    }

    // ── 3-D ────────────────────────────────────────────────────────────────────

    #[test]
    fn value_indices_3d_two_corner_voxels_and_center() {
        // 3×3×3 with 1.0 at (0,0,0) and (2,2,2), 5.0 at (1,1,1), rest = 0.0.
        let mut a = vec![0.0_f32; 27];
        a[0] = 1.0; // (0,0,0)
        a[26] = 1.0; // (2,2,2)
        a[13] = 5.0; // (1,1,1)
        let img = make_image_3d(a, [3, 3, 3]);
        let vi = value_indices(&img, None);

        assert_eq!(vi.num_distinct(), 3);
        assert_eq!(vi.total(), 27);

        let exp_1: &[[usize; 3]] = &[[0, 0, 0], [2, 2, 2]];
        let exp_5: &[[usize; 3]] = &[[1, 1, 1]];
        assert_eq!(vi.get(1.0), Some(exp_1));
        assert_eq!(vi.get(5.0), Some(exp_5));
        assert_eq!(vi.len(0.0), 24);
    }

    #[test]
    fn value_indices_3d_all_same_value() {
        // 2×2×2 filled with 7.0 → 8 occurrences of a single value.
        let img = make_image_3d(vec![7.0_f32; 8], [2, 2, 2]);
        let vi = value_indices(&img, None);
        assert_eq!(vi.num_distinct(), 1);
        assert_eq!(vi.total(), 8);
        let exp: &[[usize; 3]] = &[
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ];
        assert_eq!(vi.get(7.0), Some(exp));
    }

    #[test]
    fn value_indices_3d_single_voxel() {
        let img = make_image_3d(vec![42.0_f32], [1, 1, 1]);
        let vi = value_indices(&img, None);
        assert_eq!(vi.num_distinct(), 1);
        let exp: &[[usize; 3]] = &[[0, 0, 0]];
        assert_eq!(vi.get(42.0), Some(exp));
    }

    #[test]
    fn value_indices_3d_ignore_value_excludes_voxels() {
        // 2×3×4 with 6 distinct non-zero values placed at known locations;
        // ignore_value=0.0 drops the 18 zero voxels from the output.
        let mut a = vec![0.0_f32; 24];
        a[0] = 1.0; // (0,0,0)
        a[1] = 2.0; // (0,0,1)
        a[4] = 3.0; // (0,1,0)
        a[5] = 4.0; // (0,1,1)
        a[12] = 5.0; // (1,0,0)
        a[23] = 6.0; // (1,2,3)
        let img = make_image_3d(a, [2, 3, 4]);

        let vi_full = value_indices(&img, None);
        assert_eq!(vi_full.num_distinct(), 7);
        assert_eq!(vi_full.total(), 24);

        let vi_ignore = value_indices(&img, Some(0.0));
        assert_eq!(vi_ignore.num_distinct(), 6);
        assert_eq!(vi_ignore.total(), 6);
        assert_eq!(vi_ignore.get(0.0), None);
        let exp_1: &[[usize; 3]] = &[[0, 0, 0]];
        let exp_2: &[[usize; 3]] = &[[0, 0, 1]];
        let exp_3: &[[usize; 3]] = &[[0, 1, 0]];
        let exp_4: &[[usize; 3]] = &[[0, 1, 1]];
        let exp_5: &[[usize; 3]] = &[[1, 0, 0]];
        let exp_6: &[[usize; 3]] = &[[1, 2, 3]];
        assert_eq!(vi_ignore.get(1.0), Some(exp_1));
        assert_eq!(vi_ignore.get(2.0), Some(exp_2));
        assert_eq!(vi_ignore.get(3.0), Some(exp_3));
        assert_eq!(vi_ignore.get(4.0), Some(exp_4));
        assert_eq!(vi_ignore.get(5.0), Some(exp_5));
        assert_eq!(vi_ignore.get(6.0), Some(exp_6));
    }

    #[test]
    fn value_indices_3d_ignore_value_not_present() {
        // 2×2×2, all zero except one 1.0; ignore_value=999 has no effect.
        let mut a = vec![0.0_f32; 8];
        a[0] = 1.0;
        let img = make_image_3d(a, [2, 2, 2]);
        let vi = value_indices(&img, Some(999.0));
        assert_eq!(vi.num_distinct(), 2);
        assert_eq!(vi.total(), 8);
        let zeros: &[[usize; 3]] = &[
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ];
        assert_eq!(vi.get(0.0), Some(zeros));
        let exp_1: &[[usize; 3]] = &[[0, 0, 0]];
        assert_eq!(vi.get(1.0), Some(exp_1));
    }

    // ── Properties / invariants ────────────────────────────────────────────────

    #[test]
    fn value_indices_3d_row_major_ordering() {
        // 2×2×2 with values 1..=8 in flat order; verify that
        // value_indices returns each value at the corresponding
        // multi-index without reordering.
        let img = make_image_3d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
        let vi = value_indices(&img, None);
        assert_eq!(vi.num_distinct(), 8);
        assert_eq!(vi.total(), 8);
        for v in 1..=8u32 {
            let value = v as f32;
            let expected = flat_to_multi((v - 1) as usize, [2, 2, 2]);
            let exp: &[[usize; 3]] = &[expected];
            assert_eq!(vi.get(value), Some(exp));
        }
    }

    #[test]
    fn value_indices_3d_total_equals_voxel_count_without_ignore() {
        // Random-ish pattern; verify total() equals n.
        let mut a = Vec::with_capacity(125);
        for i in 0..125 {
            a.push(((i * 7 + 3) % 5) as f32);
        }
        let img = make_image_3d(a, [5, 5, 5]);
        let vi = value_indices(&img, None);
        assert_eq!(vi.total(), 125);
    }

    #[test]
    fn value_indices_3d_total_equals_n_minus_ignored_count() {
        // 3×3×3 with 9 voxels of value 5.0; ignore them and verify total
        // drops by 9.
        let mut a = vec![0.0_f32; 27];
        for flat in (0..27).step_by(3) {
            a[flat] = 5.0; // 9 voxels: 0, 3, 6, ..., 24
        }
        let img = make_image_3d(a, [3, 3, 3]);
        let vi_full = value_indices(&img, None);
        assert_eq!(vi_full.total(), 27);
        assert_eq!(vi_full.len(5.0), 9);
        let vi_ignored = value_indices(&img, Some(5.0));
        assert_eq!(vi_ignored.total(), 18);
        assert_eq!(vi_ignored.get(5.0), None);
    }

    #[test]
    fn flat_to_multi_round_trip_3d() {
        // Spot-check: every flat index in a 2×3×4 volume maps to a
        // multi-index that recovers the original flat index.
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

    #[test]
    fn f32_key_bit_equality() {
        // 0.0 and -0.0 are distinct bit patterns → distinct keys.
        let k_pos = F32Key::new(0.0_f32);
        let k_neg = F32Key::new(-0.0_f32);
        assert_ne!(k_pos, k_neg);
        assert_eq!(k_pos, F32Key::new(0.0_f32));
        // Bit-equal copies are equal
        assert_eq!(k_pos, F32Key::new(0.0_f32));
    }
}
