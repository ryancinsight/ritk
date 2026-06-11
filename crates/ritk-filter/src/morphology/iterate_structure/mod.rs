//! Boolean structuring element and `iterate_structure` algorithm.
//!
//! # Mathematical Specification
//!
//! scipy's `iterate_structure(structure, iterations, origin=None)` returns the
//! input structure dilated by itself `iterations - 1` times. Equivalently:
//!
//! 1. Let `ni = iterations - 1`.
//! 2. Output shape: `[S[k] + ni * (S[k] - 1)]` per axis, where `S` is the
//!    input structure's shape.
//! 3. Place the input structure at position `[ni * (S[k] // 2)]` in the
//!    output, which is "centre of input structure, repeated `ni` times over".
//! 4. Dilate the output by the input structure `ni` times, using scipy's
//!    default origin for the kernel (`kernel.shape()[k] // 2`).
//!
//! # Origin
//!
//! When `origin` is provided to `iterate_structure_with_origin`, the returned
//! `new_origin[k] = iterations * origin[k]`. The dilation itself always uses
//! scipy's default origin (`shape[k] // 2`).
//!
//! # Edge Cases
//!
//! - `iterations < 2`: returns a copy of the input structure, unchanged.
//! - Empty structure (all false): output has the iterated shape, all false.
//! - Singleton structure (1×1×1…): output has the same shape regardless of
//!   `iterations`.
//!
//! # Complexity
//!
//! - `iterate_structure`: O(ni · N · K) where `N` is the output voxel count
//!   and `K` is the number of true voxels in the input structure.
//! - `iterate_structure_with_origin`: same plus O(D) for the origin
//!   computation.
//!
//! # References
//!
//! - scipy.ndimage.iterate_structure:
//!   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html>
//! - scipy source: `scipy/ndimage/_morphology.py::iterate_structure`.

// ── BoolStructure ─────────────────────────────────────────────────────────────

/// A D-dimensional boolean structuring element.
///
/// `BoolStructure` is the natural input/output type for `iterate_structure`
/// and other binary morphology operations on structuring elements. It is
/// backend-agnostic (its data lives in a `Vec<bool>`) and is therefore
/// allocation-free to move, trivial to copy, and `Eq`-comparable.
///
/// # Memory Layout
///
/// Data is stored in row-major order: the last axis varies fastest.
/// Indexing from multi-index `[i_0, i_1, ..., i_{D-1}]` to flat index is
/// `i_0 * S_1 * S_2 * ... + i_1 * S_2 * ... + ... + i_{D-1}`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoolStructure<const D: usize> {
    /// Dimensions `[size_axis_0, size_axis_1, ..., size_axis_{D-1}]`.
    shape: [usize; D],
    /// Row-major data; `data.len() == shape.iter().product()`.
    data: Vec<bool>,
}

impl<const D: usize> BoolStructure<D> {
    /// Construct a `BoolStructure` from a shape and flat data.
    ///
    /// # Panics
    /// Panics if `data.len() != shape.iter().product()`.
    pub fn from_data(shape: [usize; D], data: Vec<bool>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "BoolStructure: data length {} does not match shape product {}",
            data.len(),
            expected,
        );
        Self { shape, data }
    }

    /// Construct a `BoolStructure` from a shape and a closure that produces
    /// each voxel from its multi-index. The closure is invoked once per voxel
    /// in row-major order.
    pub fn from_shape_fn<F: FnMut(&[usize; D]) -> bool>(shape: [usize; D], mut f: F) -> Self {
        let total: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total);
        let mut idx = [0usize; D];
        for _ in 0..total {
            data.push(f(&idx));
            increment_index(&mut idx, &shape);
        }
        Self { shape, data }
    }

    /// Total number of voxels (product of shape).
    #[inline]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Number of true voxels.
    #[inline]
    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&v| v).count()
    }

    /// Returns `true` iff no voxel is set.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Returns the shape.
    #[inline]
    pub fn shape(&self) -> &[usize; D] {
        &self.shape
    }

    /// Returns the flat data slice.
    #[inline]
    pub fn as_slice(&self) -> &[bool] {
        &self.data
    }

    /// Returns a mutable flat data slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [bool] {
        &mut self.data
    }

    /// Returns the centre of this structure (anchor for morphology
    /// operations), defined as `[shape[k] / 2 for k in 0..D]`. Matches
    /// scipy's default origin for `binary_dilation` and `binary_erosion`.
    #[inline]
    pub fn center(&self) -> [usize; D] {
        std::array::from_fn(|i| self.shape[i] / 2)
    }

    /// Returns the multi-index for a flat index.
    ///
    /// # Panics
    /// Panics if `flat >= self.size()`.
    #[inline]
    pub fn flat_to_multi(&self, flat: usize) -> [usize; D] {
        assert!(
            flat < self.data.len(),
            "BoolStructure::flat_to_multi: flat index {} out of bounds {}",
            flat,
            self.data.len(),
        );
        let mut idx = [0usize; D];
        let mut remaining = flat;
        for i in (0..D).rev() {
            let s = self.shape[i];
            if s > 0 {
                idx[i] = remaining % s;
                remaining /= s;
            }
        }
        idx
    }

    /// Returns the flat index for a multi-index.
    ///
    /// # Panics
    /// Panics if any axis is out of bounds.
    #[inline]
    pub fn multi_to_flat(&self, multi: &[usize; D]) -> usize {
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..D).rev() {
            let v = multi[i];
            assert!(
                v < self.shape[i],
                "BoolStructure::multi_to_flat: axis {} index {} out of bounds {}",
                i,
                v,
                self.shape[i],
            );
            flat += v * stride;
            stride *= self.shape[i];
        }
        flat
    }

    /// Dilate this structure by `kernel` `iterations` times, using scipy's
    /// default origin for the kernel (`kernel.shape()[k] // 2`).
    ///
    /// # Algorithm
    /// scipy's `binary_dilation` is implemented as a C-level binary erosion
    /// of the negated image, with the **flipped** structuring element and a
    /// **negated** origin. The visible semantics are:
    ///
    /// ```text
    /// out[i] = OR over q in kernel of in[i − q + c + o]
    /// ```
    ///
    /// where `q` ranges over the absolute positions of the true voxels of the
    /// original (un-flipped) kernel, `c = kernel.shape / 2` is the original
    /// kernel centre, and `o` is the original origin (default 0). Out-of-bounds
    /// positions are clipped (treated as background).
    ///
    /// The Rust implementation works in the equivalent "flipped" coordinate
    /// frame used by scipy's C kernel to keep the gather formula a flat
    /// `i + q_flip − c_flip − o_neg` (no negation in front of `i`).
    pub fn dilate(self, kernel: &BoolStructure<D>, iterations: usize) -> BoolStructure<D> {
        let mut result = self;
        for _ in 0..iterations {
            result = dilate_once(&result, kernel);
        }
        result
    }
}

/// Increment a multi-index in row-major order. After the call, the index
/// either advances to the next position or wraps back to `[0; D]` after
/// reaching the end.
#[inline]
fn increment_index<const D: usize>(idx: &mut [usize; D], shape: &[usize; D]) {
    for i in (0..D).rev() {
        idx[i] += 1;
        if idx[i] < shape[i] {
            return;
        }
        idx[i] = 0;
    }
}

/// Convert flat index to multi-index for a given shape (row-major).
#[inline]
fn flat_to_multi_generic<const D: usize>(flat: usize, shape: &[usize; D]) -> [usize; D] {
    let mut idx = [0usize; D];
    let mut remaining = flat;
    for i in (0..D).rev() {
        if shape[i] > 0 {
            idx[i] = remaining % shape[i];
            remaining /= shape[i];
        }
    }
    idx
}

/// Convert multi-index to flat index for a given shape (row-major).
#[inline]
fn multi_to_flat_generic<const D: usize>(multi: &[usize; D], shape: &[usize; D]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for i in (0..D).rev() {
        flat += multi[i] * stride;
        stride *= shape[i];
    }
    flat
}

/// One iteration of binary dilation. Output has the same shape as input.
///
/// Implements scipy's `binary_dilation` with default `origin=0` and
/// `border_value=0` (out-of-bounds positions treated as background).
///
/// Scatter approach: for each True input voxel `p`, stamp the kernel
/// centered at `p` into the output. The effective center is
/// `center[k] = shape[k] // 2` per axis, with an additional `-1` offset
/// for even-sized axes (matching scipy's `binary_dilation` origin
/// convention for `origin=0`).
///
/// `for each p where input[p] is True:
///     for each q where kernel[q] is True:
///         output[p + q − center − even_offset] = True` (if in bounds)
fn dilate_once<const D: usize>(
    input: &BoolStructure<D>,
    kernel: &BoolStructure<D>,
) -> BoolStructure<D> {
    let in_shape = *input.shape();
    let k_shape = *kernel.shape();
    let k_center: [usize; D] = std::array::from_fn(|i| k_shape[i] / 2);
    // scipy applies an extra −1 origin offset for even-sized axes.
    let even_offset: [isize; D] = std::array::from_fn(|i| if k_shape[i] & 1 == 0 { 1 } else { 0 });

    // Collect the per-voxel offsets of true voxels in the kernel.
    let k_offsets: Vec<[isize; D]> = (0..kernel.as_slice().len())
        .filter(|&i| kernel.as_slice()[i])
        .map(|i| {
            let multi = kernel.flat_to_multi(i);
            std::array::from_fn(|k| multi[k] as isize - k_center[k] as isize - even_offset[k])
        })
        .collect();

    if k_offsets.is_empty() {
        return BoolStructure::from_data(in_shape, vec![false; in_shape.iter().product()]);
    }

    let out_size: usize = in_shape.iter().product();
    let mut out = vec![false; out_size];

    // Scatter: for each True input voxel, stamp the kernel offsets.
    for (in_flat, &in_val) in input.as_slice().iter().enumerate() {
        if !in_val {
            continue;
        }
        let in_multi = flat_to_multi_generic(in_flat, &in_shape);
        for offset in &k_offsets {
            let mut out_multi = [0usize; D];
            let mut in_bounds = true;
            for i in 0..D {
                let pos = in_multi[i] as isize + offset[i];
                if pos < 0 || pos >= in_shape[i] as isize {
                    in_bounds = false;
                    break;
                }
                out_multi[i] = pos as usize;
            }
            if in_bounds {
                let out_flat = multi_to_flat_generic(&out_multi, &in_shape);
                out[out_flat] = true;
            }
        }
    }

    BoolStructure::from_data(in_shape, out)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Iterate a structure: dilate it by itself `iterations - 1` times.
///
/// Matches scipy's `scipy.ndimage.iterate_structure(structure, iterations)`
/// with `origin=None`.
///
/// # Arguments
/// * `structure` — the input structure (any D-dimensional bool structure).
/// * `iterations` — the number of iterations. `iterations < 2` returns a
///   copy of the input unchanged.
///
/// # Returns
/// A new `BoolStructure<D>` with the iterated shape and contents.
pub fn iterate_structure<const D: usize>(
    structure: BoolStructure<D>,
    iterations: usize,
) -> BoolStructure<D> {
    if iterations < 2 {
        return structure;
    }

    let ni = iterations - 1;
    let in_shape = *structure.shape();
    let out_shape: [usize; D] = std::array::from_fn(|k| in_shape[k] + ni * (in_shape[k] - 1));
    let pos: [usize; D] = std::array::from_fn(|k| ni * (in_shape[k] / 2));

    let total: usize = out_shape.iter().product();
    let mut out = BoolStructure::from_data(out_shape, vec![false; total]);

    // Stamp input at pos
    stamp(&mut out, &pos, &structure);

    // Dilate by structure, ni times
    out.dilate(&structure, ni)
}

/// Iterate a structure, also returning the scaled origin.
///
/// Matches scipy's `scipy.ndimage.iterate_structure(structure, iterations,
/// origin=...)` with `origin` provided. The dilation itself always uses
/// scipy's default origin (`shape[k] // 2`); the `origin` argument here is
/// only used to compute the returned `new_origin`.
///
/// # Returns
/// A tuple `(iterated_structure, new_origin)` where
/// `new_origin[k] = iterations * origin[k]`.
pub fn iterate_structure_with_origin<const D: usize>(
    structure: BoolStructure<D>,
    iterations: usize,
    origin: [usize; D],
) -> (BoolStructure<D>, [usize; D]) {
    let new_origin: [usize; D] = std::array::from_fn(|k| iterations * origin[k]);
    let result = iterate_structure(structure, iterations);
    (result, new_origin)
}

/// Stamp `src` into `dst` at position `pos`. `src` must fit inside `dst` at
/// `pos`. Used internally by `iterate_structure` to place the input
/// structure in the output canvas before dilating.
fn stamp<const D: usize>(dst: &mut BoolStructure<D>, pos: &[usize; D], src: &BoolStructure<D>) {
    let dst_shape = *dst.shape();
    for (src_flat, &src_val) in src.as_slice().iter().enumerate() {
        if !src_val {
            continue;
        }
        let src_multi = src.flat_to_multi(src_flat);
        let dst_multi: [usize; D] = std::array::from_fn(|k| pos[k] + src_multi[k]);
        assert!(
            (0..D).all(|k| dst_multi[k] < dst_shape[k]),
            "BoolStructure::stamp: src at pos {:?} does not fit in dst shape {:?}",
            pos,
            dst_shape,
        );
        let dst_flat = dst.multi_to_flat(&dst_multi);
        dst.as_mut_slice()[dst_flat] = true;
    }
}

#[cfg(test)]
mod tests;
