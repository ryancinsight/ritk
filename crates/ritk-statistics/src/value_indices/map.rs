use super::key::F32Key;
use std::collections::HashMap;

/// Per-value index map returned by [`value_indices`](super::value_indices).
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
