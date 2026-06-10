//! Per-value index map: for each distinct voxel value, the multi-indices
//! where it occurs.
//!
//! # Mathematical Specification
//!
//! Given a D-dimensional image `I` of shape `[d_0, d_1, …, d_{D-1}]`
//! stored in row-major order, the **value-index map** is a dictionary
//!
//! VI = { v : M_v : v ∈ distinct(I) }
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
//! p = i_0 · (d_1 · d_2 · …) + i_1 · (d_2 · …) + … + i_{D-1}.
//!
//! # SciPy Reference
//!
//! \[`scipy.ndimage.value_indices`\] (added in scipy 1.10.0). The default
//! `ignore_value=None` includes every distinct value in the output.

pub mod compute;
pub mod key;
pub mod map;

#[cfg(test)]
mod tests;

pub use compute::value_indices;
pub use key::F32Key;
pub use map::ValueIndices;
