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
//! Both filters use `select_nth_unstable_by` (introselect) for `O(n)` per
//! voxel selection. This avoids a full `O(n log n)` sort and is asymptotically
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

pub mod percentile_filter;
pub mod rank_filter;

pub use percentile_filter::PercentileFilter;
pub use rank_filter::RankFilter;

#[cfg(test)]
mod tests;
