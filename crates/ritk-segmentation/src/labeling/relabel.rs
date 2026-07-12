//! Re-label connected components in order of decreasing size.
//!
//! # Mathematical Specification
//!
//! Given a label image L where L\[v\] ∈ {0, 1, …, K} (0 = background, 1…K are
//! component indices from `ConnectedComponentsFilter`), the relabeling mapping
//! ρ: {0…K} → {0…K'} is defined as:
//!
//! 1. For each component k ∈ {1…K}, compute size(k) = |{v : L\[v\] = k}|.
//! 2. Sort components by (size(k) descending, k ascending) to obtain a
//!    permutation π of the non-background labels.
//! 3. Let K' = |{k : size(k) ≥ τ}| where τ = `minimum_object_size`.
//! 4. ρ(k) = position of k in π restricted to {k : size(k) ≥ τ},
//!    using 1-based indexing.  If size(k) < τ then ρ(k) = 0.
//! 5. ρ(0) = 0 (background is preserved).
//!
//! The output O\[v\] = ρ(L\[v\]) has at most K' non-zero labels, each ≥ 1, with
//! label 1 assigned to the component with the most voxels.
//!
//! # ITK parity
//! Matches `itk::RelabelComponentImageFilter` semantics:
//! - `SetMinimumObjectSize(τ)` removes components with fewer than τ voxels.
//! - Default τ = 0 (no removal).
//! - Components are always renumbered 1…K' in descending-size order.
//!
//! # Complexity
//! - Pass 1 (count):  O(n) voxels.
//! - Sorting:         O(K log K) where K = number of unique non-zero labels.
//! - Pass 2 (remap):  O(n).
//! - Space:           O(K) auxiliary.

use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-component statistics produced by `RelabelComponentFilter`.
#[derive(Debug, Clone, PartialEq)]
pub struct RelabelStatistics {
    /// Original label in the input image (before relabeling).
    pub original_label: u32,
    /// New label in the output image (1-based, 0 if discarded).
    pub new_label: u32,
    /// Number of voxels carrying this label in the *input* image.
    pub voxel_count: usize,
}

/// Re-label connected components in order of decreasing voxel count.
///
/// Wraps a label image (output of `ConnectedComponentsFilter`) and reassigns
/// component indices such that label 1 = largest component, label 2 = second
/// largest, and so on.  Components with fewer than `minimum_object_size` voxels
/// are merged into the background (set to 0.0).
///
/// # ITK parity
/// Matches `itk::RelabelComponentImageFilter::SetMinimumObjectSize`.
/// Default `minimum_object_size = 0` retains all components.
pub struct RelabelComponentFilter {
    /// Minimum number of voxels a component must have to survive relabeling.
    ///
    /// Components with `voxel_count < minimum_object_size` are removed
    /// (set to background 0.0 in the output).  Default: 0 (retain all).
    minimum_object_size: usize,
}

impl RelabelComponentFilter {
    /// Create a filter that retains all components (ITK default: no size threshold).
    pub fn new() -> Self {
        Self {
            minimum_object_size: 0,
        }
    }

    /// Create a filter that discards components smaller than `minimum_object_size` voxels.
    ///
    /// Matches `itk::RelabelComponentImageFilter::SetMinimumObjectSize`.
    pub fn with_minimum_object_size(minimum_object_size: usize) -> Self {
        Self {
            minimum_object_size,
        }
    }

    /// Return the minimum component size retained by the filter.
    pub fn minimum_object_size(&self) -> usize {
        self.minimum_object_size
    }

    /// Apply relabeling to a Coeus-native label image.
    ///
    /// # Errors
    ///
    /// Returns an error when backend storage is not host-addressable or the
    /// native output image cannot be constructed.
    pub fn apply_native<B>(
        &self,
        label_image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<(ritk_image::native::Image<f32, B, 3>, Vec<RelabelStatistics>)>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (values, statistics) =
            relabel_values(label_image.data_slice()?, self.minimum_object_size)?;
        Ok((
            crate::native_output::from_values(label_image, values, backend)?,
            statistics,
        ))
    }

    /// Apply relabeling to an integer label image (output of `ConnectedComponentsFilter`).
    ///
    /// # Precondition
    /// - `label_image` voxels are non-negative f32 integers (0 = background, k ≥ 1 = component).
    /// - Voxel values must be representable as `u32` (integers in [0, 2^24]).
    ///
    /// # Postcondition
    /// - Output voxels are in {0, 1, …, K'} where K' ≤ K.
    /// - Label 1 corresponds to the component with the largest voxel count in the input.
    /// - Components with `voxel_count < self.minimum_object_size` are mapped to 0.
    /// - Spatial metadata (origin, spacing, direction) is preserved unchanged.
    ///
    /// Returns `(output_image, statistics)` where `statistics` has one entry per
    /// surviving component in ascending new-label order.
    ///
    /// # Errors
    ///
    /// Returns an error if any input label is non-finite, negative,
    /// fractional, or outside the exact integer range of `f32`.
    pub fn apply<B: Backend>(
        &self,
        label_image: &Image<B, 3>,
    ) -> anyhow::Result<(Image<B, 3>, Vec<RelabelStatistics>)> {
        let (data_vals, shape) = extract_vec_infallible(label_image);
        let device = label_image.data().device();
        let flat: &[f32] = &data_vals;

        let (output_vec, stats) = relabel_values(flat, self.minimum_object_size)?;

        let td = TensorData::new(output_vec, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let out_image = Image::new(
            tensor,
            *label_image.origin(),
            *label_image.spacing(),
            *label_image.direction(),
        );

        Ok((out_image, stats))
    }
}

impl Default for RelabelComponentFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Core relabeling algorithm.
///
/// Returns `(output_vec, statistics)` for the flat ZYX label slice.
///
/// # Algorithm
/// 1. Count voxels per non-zero label in O(n): `counts[label] = count`.
/// 2. Sort (label, count) pairs by (count desc, label asc) in O(K log K).
/// 3. Assign new labels 1…K' to entries with count ≥ `min_size`.
/// 4. Build remap table old_label → new_label in O(K).
/// 5. Remap the flat slice in O(n).
pub(crate) fn relabel_values(
    flat: &[f32],
    min_size: usize,
) -> anyhow::Result<(Vec<f32>, Vec<RelabelStatistics>)> {
    // Step 1 — Count voxels per label.
    const MAX_EXACT_LABEL: f32 = 16_777_216.0;
    let mut counts: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::with_capacity(flat.len() / 4 + 1);
    for &v in flat {
        anyhow::ensure!(
            v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v <= MAX_EXACT_LABEL,
            "label values must be finite non-negative integers exactly representable as f32, got {v}"
        );
        let label = v as u32;
        if label != 0 {
            *counts.entry(label).or_insert(0) += 1;
        }
    }

    if counts.is_empty() {
        return Ok((flat.iter().map(|_| 0.0_f32).collect(), Vec::new()));
    }

    // Step 2 — Sort by (count desc, label asc) for deterministic tie-breaking.
    let mut sorted: Vec<(u32, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Step 3 — Assign new labels; build statistics.
    let mut remap = std::collections::HashMap::with_capacity(sorted.len());
    let mut stats: Vec<RelabelStatistics> = Vec::with_capacity(sorted.len());
    let mut new_label: u32 = 0;

    for (old_label, count) in sorted {
        // Retain if count ≥ min_size.  When min_size = 0 (ITK default),
        // count ≥ 0 is always true so all components are retained.
        if count >= min_size {
            new_label += 1;
            remap.insert(old_label, new_label);
            stats.push(RelabelStatistics {
                original_label: old_label,
                new_label,
                voxel_count: count,
            });
        }
        // count < min_size: remap entry remains 0 (background); no stats entry.
    }

    // Step 5 — Apply remap O(n).
    let output: Vec<f32> = flat
        .iter()
        .map(|&v| {
            let old = v as u32;
            remap.get(&old).copied().unwrap_or(0) as f32
        })
        .collect();

    Ok((output, stats))
}

/// Relabel non-zero labels to consecutive integers `1, 2, …, K` in **ascending
/// original-label order** (background 0 unchanged).
///
/// Matches `sitk.RelabelLabelMap` (via the LabelMap round-trip): unlike
/// [`RelabelComponentFilter`] (size-descending), the LabelMap relabeling assigns
/// new labels in the order of the existing (ascending) label values. Values are
/// rounded to the nearest integer.
pub fn relabel_consecutive<B: Backend>(label_image: &Image<B, 3>) -> Image<B, 3> {
    let (vals, dims) = extract_vec_infallible(label_image);
    let mut uniq: Vec<u32> = vals
        .iter()
        .map(|&v| v.round().max(0.0) as u32)
        .filter(|&l| l != 0)
        .collect();
    uniq.sort_unstable();
    uniq.dedup();
    // old label → new consecutive label (1-based, ascending original order).
    let max_label = uniq.last().copied().unwrap_or(0) as usize;
    let mut remap = vec![0u32; max_label + 1];
    for (new_minus_1, &old) in uniq.iter().enumerate() {
        remap[old as usize] = new_minus_1 as u32 + 1;
    }
    let out: Vec<f32> = vals
        .iter()
        .map(|&v| {
            let l = v.round().max(0.0) as usize;
            if l == 0 || l > max_label {
                0.0
            } else {
                remap[l] as f32
            }
        })
        .collect();
    rebuild(out, dims, label_image)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_relabel.rs"]
mod tests;
