//! Re-label connected components in order of decreasing size.
//!
//! # Mathematical Specification
//!
//! Given a label image L where L[v] ∈ {0, 1, …, K} (0 = background, 1…K are
//! component indices from `ConnectedComponentsFilter`), the relabeling mapping
//! ρ: {0…K} → {0…K'} is defined as:
//!
//! 1. For each component k ∈ {1…K}, compute size(k) = |{v : L[v] = k}|.
//! 2. Sort components by (size(k) descending, k ascending) to obtain a
//!    permutation π of the non-background labels.
//! 3. Let K' = |{k : size(k) ≥ τ}| where τ = `minimum_object_size`.
//! 4. ρ(k) = position of k in π restricted to {k : size(k) ≥ τ},
//!           using 1-based indexing.  If size(k) < τ then ρ(k) = 0.
//! 5. ρ(0) = 0 (background is preserved).
//!
//! The output O[v] = ρ(L[v]) has at most K' non-zero labels, each ≥ 1, with
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

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

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
    pub minimum_object_size: usize,
}

impl RelabelComponentFilter {
    /// Create a filter that retains all components (ITK default: no size threshold).
    pub fn new() -> Self {
        Self { minimum_object_size: 0 }
    }

    /// Create a filter that discards components smaller than `minimum_object_size` voxels.
    ///
    /// Matches `itk::RelabelComponentImageFilter::SetMinimumObjectSize`.
    pub fn with_minimum_object_size(minimum_object_size: usize) -> Self {
        Self { minimum_object_size }
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
    pub fn apply<B: Backend>(&self, label_image: &Image<B, 3>) -> (Image<B, 3>, Vec<RelabelStatistics>) {
        let shape = label_image.shape();
        let device = label_image.data().device();

        let data = label_image.data().clone().into_data();
        let flat = data.as_slice::<f32>().expect("f32 label tensor");

        let (output_vec, stats) = relabel_impl(flat, self.minimum_object_size);

        let td = TensorData::new(output_vec, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let out_image = Image::new(
            tensor,
            *label_image.origin(),
            *label_image.spacing(),
            *label_image.direction(),
        );

        (out_image, stats)
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
fn relabel_impl(flat: &[f32], min_size: usize) -> (Vec<f32>, Vec<RelabelStatistics>) {
    // Step 1 — Count voxels per label.
    // Labels are stored as f32 integers; convert via round then clamp to u32.
    let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &v in flat {
        let label = v as u32;
        if label != 0 {
            *counts.entry(label).or_insert(0) += 1;
        }
    }

    if counts.is_empty() {
        return (flat.iter().map(|_| 0.0_f32).collect(), Vec::new());
    }

    // Step 2 — Sort by (count desc, label asc) for deterministic tie-breaking.
    let mut sorted: Vec<(u32, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Step 3 — Assign new labels; build statistics.
    // Maximum label value in the input — used to size the remap table.
    let max_old_label = sorted.iter().map(|&(l, _)| l).max().unwrap_or(0) as usize;
    let mut remap: Vec<u32> = vec![0u32; max_old_label + 1];
    let mut stats: Vec<RelabelStatistics> = Vec::new();
    let mut new_label: u32 = 0;

    for (old_label, count) in sorted {
        // Retain if count ≥ min_size.  When min_size = 0 (ITK default),
        // count ≥ 0 is always true so all components are retained.
        if count >= min_size {
            new_label += 1;
            remap[old_label as usize] = new_label;
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
            let old = v as usize;
            if old == 0 || old > max_old_label {
                0.0_f32
            } else {
                remap[old] as f32
            }
        })
        .collect();

    (output, stats)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_label_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn flat(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// Single component, no size threshold → relabeled as 1, count preserved.
    #[test]
    fn single_component_identity() {
        // 2×1×1 image: both voxels are component 1.
        let img = make_label_image(vec![1.0, 1.0], [2, 1, 1]);
        let (out, stats) = RelabelComponentFilter::new().apply(&img);
        assert_eq!(flat(&out), vec![1.0, 1.0]);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].new_label, 1);
        assert_eq!(stats[0].voxel_count, 2);
        assert_eq!(stats[0].original_label, 1);
    }

    /// Three components with distinct sizes → sorted by descending count.
    ///
    /// Input labels and voxel counts: {1:5, 2:15, 3:3}.
    /// Expected new labels: 2→1 (15), 1→2 (5), 3→3 (3).
    #[test]
    fn three_components_sorted_descending() {
        // 1×1×23 flat image: label 1 appears 5×, label 2 appears 15×, label 3 appears 3×.
        let mut vals = vec![1.0_f32; 5];
        vals.extend(vec![2.0_f32; 15]);
        vals.extend(vec![3.0_f32; 3]);
        let n = vals.len();
        let img = make_label_image(vals.clone(), [1, 1, n]);

        let (out, stats) = RelabelComponentFilter::new().apply(&img);
        let out_flat = flat(&out);

        // stats should be sorted by descending count: 15, 5, 3.
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].new_label, 1);
        assert_eq!(stats[0].voxel_count, 15);
        assert_eq!(stats[0].original_label, 2);
        assert_eq!(stats[1].new_label, 2);
        assert_eq!(stats[1].voxel_count, 5);
        assert_eq!(stats[1].original_label, 1);
        assert_eq!(stats[2].new_label, 3);
        assert_eq!(stats[2].voxel_count, 3);
        assert_eq!(stats[2].original_label, 3);

        // Voxels that were 2 should now be 1, 1→2, 3→3.
        let expected: Vec<f32> = vals
            .iter()
            .map(|&v| match v as u32 { 2 => 1.0, 1 => 2.0, 3 => 3.0, _ => 0.0 })
            .collect();
        assert_eq!(out_flat, expected);
    }

    /// `minimum_object_size` removes components below threshold.
    ///
    /// Components: {1: 3 voxels, 2: 10 voxels}. Threshold = 5.
    /// Expected: component 1 removed (→0), component 2 relabeled to 1.
    #[test]
    fn minimum_object_size_removes_small() {
        let mut vals = vec![1.0_f32; 3]; // label 1, count=3 (small)
        vals.extend(vec![2.0_f32; 10]); // label 2, count=10 (large)
        let n = vals.len();
        let img = make_label_image(vals, [1, 1, n]);

        let (out, stats) =
            RelabelComponentFilter::with_minimum_object_size(5).apply(&img);
        let out_flat = flat(&out);

        // Only the large component survives as label 1.
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].original_label, 2);
        assert_eq!(stats[0].new_label, 1);
        assert_eq!(stats[0].voxel_count, 10);

        // First 3 voxels (label 1) → 0; last 10 voxels (label 2) → 1.
        let mut expected = vec![0.0_f32; 3];
        expected.extend(vec![1.0_f32; 10]);
        assert_eq!(out_flat, expected);
    }

    /// All components below minimum_object_size → all-zero output.
    #[test]
    fn all_below_threshold_gives_all_zero() {
        let vals: Vec<f32> = (1..=4).map(|v| v as f32).collect(); // labels 1,2,3,4 each with 1 voxel
        let img = make_label_image(vals, [1, 1, 4]);

        let (out, stats) =
            RelabelComponentFilter::with_minimum_object_size(2).apply(&img);

        assert!(stats.is_empty());
        assert!(flat(&out).iter().all(|&v| v == 0.0));
    }

    /// Background voxels (0.0) are preserved as 0.0 after relabeling.
    #[test]
    fn background_preserved() {
        // Pattern: bg, comp1, bg, comp1, bg
        let vals = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let img = make_label_image(vals, [1, 1, 5]);

        let (out, stats) = RelabelComponentFilter::new().apply(&img);
        let out_flat = flat(&out);

        assert_eq!(stats.len(), 1);
        assert_eq!(out_flat, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    /// All-background input produces all-zero output with empty statistics.
    #[test]
    fn all_background_produces_empty_stats() {
        let img = make_label_image(vec![0.0, 0.0, 0.0], [1, 1, 3]);
        let (out, stats) = RelabelComponentFilter::new().apply(&img);
        assert!(stats.is_empty());
        assert_eq!(flat(&out), vec![0.0, 0.0, 0.0]);
    }

    /// Spatial metadata (origin, spacing, direction) is preserved unchanged.
    #[test]
    fn spatial_metadata_preserved() {
        use crate::spatial::Direction;
        let device = Default::default();
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.75, 1.25]);
        let direction = Direction::identity();

        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(vec![1.0_f32; 8], Shape::new([2, 2, 2])), &device);
        let img = Image::new(tensor, origin, spacing, direction);

        let (out, _) = RelabelComponentFilter::new().apply(&img);

        assert_eq!(*out.origin(), origin);
        assert_eq!(*out.spacing(), spacing);
        assert_eq!(*out.direction(), direction);
    }

    /// Two equal-size components → sorted by original label ascending (tie-break).
    ///
    /// Both label 1 and label 2 have 4 voxels.
    /// Tie-break by ascending label: 1 → new 1, 2 → new 2.
    #[test]
    fn equal_size_tiebreak_by_label() {
        let vals = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let img = make_label_image(vals, [1, 1, 8]);

        let (out, stats) = RelabelComponentFilter::new().apply(&img);

        // Tie-break: label 1 comes first (ascending original label).
        assert_eq!(stats[0].original_label, 1);
        assert_eq!(stats[0].new_label, 1);
        assert_eq!(stats[1].original_label, 2);
        assert_eq!(stats[1].new_label, 2);
        assert_eq!(flat(&out), vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
    }

    /// minimum_object_size = 1 is equivalent to no filtering (retain all components).
    ///
    /// This verifies the boundary condition: min_size=1 means "at least 1 voxel",
    /// which matches all non-background labels by definition.
    #[test]
    fn minimum_object_size_one_retains_all() {
        let vals = vec![1.0, 2.0, 3.0]; // three single-voxel components
        let img = make_label_image(vals, [1, 1, 3]);

        let (_, stats) = RelabelComponentFilter::with_minimum_object_size(1).apply(&img);
        assert_eq!(stats.len(), 3);
        assert!(stats.iter().all(|s| s.new_label > 0));
    }
}
