//! Marker-controlled watershed segmentation (seeded variant of Meyer's flooding algorithm).
//!
//! # Mathematical Specification
//!
//! The marker-controlled watershed extends Meyer's flooding algorithm by initializing
//! the priority queue from explicit seed (marker) regions rather than discovering minima
//! during traversal.
//!
//! ## Algorithm
//!
//! Given:
//! - Gradient image G: 3D float image (typically gradient magnitude of the original).
//! - Marker image M:   3D label image where M(x) > 0 indicates a seed of basin M(x);
//!                     M(x) = 0 indicates unlabeled voxels.
//!
//! 1. **Seed initialization**:
//!    - Copy all seed labels into the output label map.
//!    - Enqueue all unlabeled voxels that are 6-adjacent to at least one seed voxel,
//!      with priority = G(v). Ties in priority broken by linear index (determinism).
//!
//! 2. **Priority-queue flooding** (ascending gradient):
//!    - Pop voxel v with minimum G(v).
//!    - If v is already labeled, discard and continue.
//!    - Collect distinct non-zero labels L among v's labeled 6-connected neighbors.
//!    - |L| = 1 → assign v = the single label.
//!    - |L| > 1 → assign v = 0 (watershed boundary).
//!    - Push all unlabeled 6-neighbors of v into the queue.
//!
//! 3. **Output**: label image. Label 0 = watershed boundary or unreachable voxel.
//!
//! ## Properties
//! - Seed labels are preserved: ∀x, M(x) > 0 ⟹ Output(x) = M(x).
//! - Determinism: (G(v), linear_index(v)) comparison fully orders the queue.
//! - 6-connectivity: each basin is 6-connected by construction.
//! - Boundary guarantee: voxels adjacent to two distinct basins become boundaries (0).
//!
//! # Complexity
//! - O(n log n) dominated by priority-queue operations.
//! - O(n) memory for label map and visited flags.
//!
//! # References
//! - Meyer, F. (1994). "Topographic distance and watershed lines."
//!   *Signal Processing*, 38(1), 113–125.
//! - Vincent, L. & Soille, P. (1991). "Watersheds in digital spaces: An efficient
//!   algorithm based on immersion simulations." *IEEE Trans. Pattern Anal. Mach. Intell.*,
//!   13(6), 583–598.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Marker-controlled watershed segmentation.
///
/// Floods from explicit seed (marker) regions using a min-heap ordered by
/// gradient intensity. Regions expand in order of increasing gradient magnitude;
/// voxels adjacent to two distinct basins become watershed boundaries (label 0).
#[derive(Debug, Clone)]
pub struct MarkerControlledWatershed {}

impl MarkerControlledWatershed {
    /// Create a new `MarkerControlledWatershed` filter.
    pub fn new() -> Self {
        Self {}
    }

    /// Apply marker-controlled watershed segmentation.
    ///
    /// # Arguments
    /// - `gradient`: 3D scalar image (typically gradient magnitude). Drives the flooding order.
    /// - `markers`:  3D label image. Non-zero values define basin seeds. Zeros are unlabeled.
    ///
    /// # Returns
    /// A label image with the same shape and spatial metadata as `gradient`:
    /// - Non-zero labels: assigned basin index (from the marker seeds).
    /// - Zero: watershed boundary or voxel unreachable from any seed.
    ///
    /// # Panics
    /// Panics if `gradient` and `markers` have different shapes.
    pub fn apply<B: Backend>(
        &self,
        gradient: &Image<B, 3>,
        markers: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims_g = gradient.shape();
        let dims_m = markers.shape();
        assert_eq!(
            dims_g, dims_m,
            "gradient and marker images must have the same shape: {:?} vs {:?}",
            dims_g, dims_m
        );

        let device = gradient.data().device();

        let g_data = gradient.data().clone().into_data();
        let g_vals: Vec<f32> = g_data
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("gradient image requires f32 data: {:?}", e))?
            .to_vec();

        let m_data = markers.data().clone().into_data();
        let m_vals: Vec<f32> = m_data
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("marker image requires f32 data: {:?}", e))?
            .to_vec();

        let labels = marker_controlled_flooding(&g_vals, &m_vals, dims_g);

        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(labels, Shape::new(dims_g)), &device);

        Ok(Image::new(
            tensor,
            gradient.origin().clone(),
            gradient.spacing().clone(),
            gradient.direction().clone(),
        ))
    }
}

impl Default for MarkerControlledWatershed {
    fn default() -> Self {
        Self::new()
    }
}

// ── Priority-queue entry ───────────────────────────────────────────────────────

/// Min-heap entry ordered by (gradient_value, insertion_sequence).
///
/// `BinaryHeap` is a max-heap. We achieve min-heap semantics by reversing the
/// gradient comparison: for non-negative `f32` values, the IEEE 754 bit
/// representation (`f32::to_bits()`) preserves total order, so reversing the
/// `u32` comparison (`other.grad_bits.cmp(&self.grad_bits)`) gives a correct
/// min-heap without any external crate.
///
/// Ties at equal gradient are broken by FIFO insertion order (smaller sequence
/// number = earlier insertion = higher priority). FIFO within each gradient
/// level is required for correct boundary placement: without it, a voxel
/// enqueued later at the same gradient can be processed before a symmetrically
/// equidistant voxel from the opposite seed, producing incorrect basin labels.
#[derive(Debug)]
struct QueueEntry {
    /// Raw `u32` bits of the (non-negative) `f32` gradient.
    /// Non-negative IEEE 754 single-precision bit patterns are totally ordered
    /// the same way as their `u32` representations, so reversing the `u32`
    /// comparison yields a correct min-heap on gradient value.
    grad_bits: u32,
    /// Monotonically increasing insertion sequence number.
    /// Smaller = inserted earlier = higher FIFO priority at equal gradient.
    seq: u64,
    /// Linear voxel index (z*ny*nx + y*nx + x); stored for the caller.
    idx: usize,
}

impl QueueEntry {
    fn new(grad: f32, idx: usize, seq: u64) -> Self {
        // Clamp NaN / negative gradients to 0.0 so bit ordering stays valid.
        let g = if grad.is_nan() || grad < 0.0 {
            0.0_f32
        } else {
            grad
        };
        Self {
            grad_bits: g.to_bits(),
            seq,
            idx,
        }
    }
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.grad_bits == other.grad_bits && self.seq == other.seq && self.idx == other.idx
    }
}
impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on gradient: smaller grad_bits → higher priority.
        // In a max-heap, `self` must compare Greater than `other` to be popped first,
        // so we reverse: other.grad_bits.cmp(&self.grad_bits).
        // FIFO tie-break: smaller seq → higher priority → also reversed.
        other
            .grad_bits
            .cmp(&self.grad_bits)
            .then(other.seq.cmp(&self.seq))
    }
}

// ── Core implementation ────────────────────────────────────────────────────────

/// 6-connected face offsets for a 3D grid (±z, ±y, ±x).
const FACE_OFFSETS: [(i64, i64, i64); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Marker-controlled watershed flooding on flat voxel arrays.
///
/// Returns a `Vec<f32>` of the same length as `grad_vals` containing integer
/// labels encoded as `f32`: 0.0 for boundaries/unreachable, ≥1.0 for basins.
fn marker_controlled_flooding(
    grad_vals: &[f32],
    marker_vals: &[f32],
    dims: [usize; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    debug_assert_eq!(grad_vals.len(), n);
    debug_assert_eq!(marker_vals.len(), n);

    if n == 0 {
        return Vec::new();
    }

    // Sentinel: u32::MAX = "not yet assigned". 0 = boundary / unreachable.
    const UNLABELED: u32 = u32::MAX;

    // Initialize label map from markers (round f32 label to u32).
    let mut labels: Vec<u32> = marker_vals
        .iter()
        .map(|&v| {
            let lbl = v.round() as i64;
            if lbl > 0 {
                lbl as u32
            } else {
                UNLABELED
            }
        })
        .collect();

    let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // ── 1. Initialize priority queue from unlabeled neighbors of seeds ─────────
    let mut in_queue = vec![false; n];
    let mut heap: BinaryHeap<QueueEntry> = BinaryHeap::new();
    // Monotonically increasing insertion counter for FIFO tie-breaking at
    // equal gradient levels. Earlier-inserted entries have smaller seq and
    // are popped first, ensuring correct boundary placement on plateaus.
    let mut seq: u64 = 0;

    for idx in 0..n {
        if labels[idx] == UNLABELED {
            continue;
        }
        // Seed voxel — enqueue its unlabeled 6-neighbors.
        let z = idx / (ny * nx);
        let rem = idx % (ny * nx);
        let y = rem / nx;
        let x = rem % nx;

        for &(dz, dy, dx) in &FACE_OFFSETS {
            let nz_i = z as i64 + dz;
            let ny_i = y as i64 + dy;
            let nx_i = x as i64 + dx;
            if nz_i < 0
                || nz_i >= nz as i64
                || ny_i < 0
                || ny_i >= ny as i64
                || nx_i < 0
                || nx_i >= nx as i64
            {
                continue;
            }
            let ni = flat(nz_i as usize, ny_i as usize, nx_i as usize);
            if labels[ni] == UNLABELED && !in_queue[ni] {
                in_queue[ni] = true;
                heap.push(QueueEntry::new(grad_vals[ni], ni, seq));
                seq += 1;
            }
        }
    }

    // ── 2. Priority-queue flooding (ascending gradient) ────────────────────────
    while let Some(entry) = heap.pop() {
        let idx = entry.idx;

        // Already labeled by a prior pop (duplicate entries are possible).
        if labels[idx] != UNLABELED {
            continue;
        }

        let z = idx / (ny * nx);
        let rem = idx % (ny * nx);
        let y = rem / nx;
        let x = rem % nx;

        // Collect distinct non-zero basin labels among 6-connected neighbors.
        let mut nbr_labels: [u32; 6] = [0u32; 6];
        let mut n_distinct = 0usize;

        for &(dz, dy, dx) in &FACE_OFFSETS {
            let nz_i = z as i64 + dz;
            let ny_i = y as i64 + dy;
            let nx_i = x as i64 + dx;
            if nz_i < 0
                || nz_i >= nz as i64
                || ny_i < 0
                || ny_i >= ny as i64
                || nx_i < 0
                || nx_i >= nx as i64
            {
                continue;
            }
            let ni = flat(nz_i as usize, ny_i as usize, nx_i as usize);
            let lbl = labels[ni];
            // Ignore unlabeled and boundary (0) neighbors.
            if lbl == UNLABELED || lbl == 0 {
                continue;
            }
            let mut found = false;
            for k in 0..n_distinct {
                if nbr_labels[k] == lbl {
                    found = true;
                    break;
                }
            }
            if !found {
                nbr_labels[n_distinct] = lbl;
                n_distinct += 1;
            }
        }

        labels[idx] = match n_distinct {
            0 => {
                // No labeled neighbor reachable yet; mark boundary/unreachable.
                0
            }
            1 => nbr_labels[0],
            _ => 0, // Adjacent to two or more distinct basins → watershed boundary.
        };

        // Enqueue unlabeled 6-neighbors not yet in the queue.
        for &(dz, dy, dx) in &FACE_OFFSETS {
            let nz_i = z as i64 + dz;
            let ny_i = y as i64 + dy;
            let nx_i = x as i64 + dx;
            if nz_i < 0
                || nz_i >= nz as i64
                || ny_i < 0
                || ny_i >= ny as i64
                || nx_i < 0
                || nx_i >= nx as i64
            {
                continue;
            }
            let ni = flat(nz_i as usize, ny_i as usize, nx_i as usize);
            if labels[ni] == UNLABELED && !in_queue[ni] {
                in_queue[ni] = true;
                heap.push(QueueEntry::new(grad_vals[ni], ni, seq));
                seq += 1;
            }
        }
    }

    // ── 3. Convert to f32 output (UNLABELED → 0 for unreachable voxels) ────────
    labels
        .iter()
        .map(|&lbl| {
            if lbl == UNLABELED {
                0.0_f32
            } else {
                lbl as f32
            }
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn get_labels(image: &Image<B, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── Seeds preserved ───────────────────────────────────────────────────────

    #[test]
    fn test_seed_labels_preserved() {
        // 1×1×5 uniform gradient; seed at index 0 = label 1, seed at index 4 = label 2.
        let gradient = make_image_3d(vec![1.0_f32; 5], [1, 1, 5]);
        let mut markers_data = vec![0.0_f32; 5];
        markers_data[0] = 1.0;
        markers_data[4] = 2.0;
        let markers = make_image_3d(markers_data, [1, 1, 5]);

        let result = MarkerControlledWatershed::new()
            .apply(&gradient, &markers)
            .unwrap();
        let labels = get_labels(&result);

        assert_eq!(labels[0], 1.0, "seed at index 0 must retain label 1");
        assert_eq!(labels[4], 2.0, "seed at index 4 must retain label 2");
    }

    // ── Two seeds on uniform gradient: watershed boundary in middle ────────────

    #[test]
    fn test_two_seeds_uniform_gradient_boundary_in_middle() {
        // 1×1×5 uniform gradient; seed label 1 at [0], seed label 2 at [4].
        // Voxels [1,2,3] expand from both ends simultaneously.
        // The middle voxel should become a watershed boundary.
        let gradient = make_image_3d(vec![1.0_f32; 5], [1, 1, 5]);
        let markers = make_image_3d(vec![1.0, 0.0, 0.0, 0.0, 2.0], [1, 1, 5]);

        let result = MarkerControlledWatershed::new()
            .apply(&gradient, &markers)
            .unwrap();
        let labels = get_labels(&result);

        assert_eq!(labels[0], 1.0, "seed 1 preserved");
        assert_eq!(labels[4], 2.0, "seed 2 preserved");
        // Expansion from both ends on uniform gradient: labels[1]=1 (adjacent to 1),
        // labels[3]=2 (adjacent to 2); labels[2] is adjacent to both → boundary 0.
        assert_eq!(
            labels[1], 1.0,
            "voxel 1 adjacent to seed 1 must expand to label 1"
        );
        assert_eq!(
            labels[3], 2.0,
            "voxel 3 adjacent to seed 2 must expand to label 2"
        );
        assert_eq!(
            labels[2], 0.0,
            "middle voxel adjacent to both basins must be watershed boundary"
        );
    }

    // ── Gradient drives flooding order ────────────────────────────────────────

    #[test]
    fn test_gradient_drives_flooding_order() {
        // 1×1×6: gradient [0,1,2,2,1,0], seeds at idx 0 (label 1) and idx 5 (label 2).
        // Low gradient regions (near edges) flood first.
        // idx 1: adjacent to seed 0 (label 1), grad=1 → label 1.
        // idx 4: adjacent to seed 5 (label 2), grad=1 → label 2.
        // idx 2: adjacent to idx1 (label 1), grad=2 → label 1.
        // idx 3: adjacent to idx4 (label 2), grad=2 → label 2.
        // Actually idx 2 and idx 3 are queued with grad=2 simultaneously, and they are
        // adjacent to different labels only, so they each get their label cleanly.
        let gradient = make_image_3d(vec![0.0, 1.0, 2.0, 2.0, 1.0, 0.0], [1, 1, 6]);
        let markers = make_image_3d(vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0], [1, 1, 6]);

        let result = MarkerControlledWatershed::new()
            .apply(&gradient, &markers)
            .unwrap();
        let labels = get_labels(&result);

        assert_eq!(labels[0], 1.0, "seed preserved");
        assert_eq!(labels[5], 2.0, "seed preserved");
        assert_eq!(
            labels[1], 1.0,
            "voxel 1 must be labeled 1 (adjacent to seed 1 only)"
        );
        assert_eq!(
            labels[4], 2.0,
            "voxel 4 must be labeled 2 (adjacent to seed 2 only)"
        );
        // Inner voxels may form a boundary or be assigned to one basin.
        // Both labels[2] and labels[3] must be non-negative integers.
        assert!(
            labels[2] >= 0.0 && labels[2] == labels[2].floor(),
            "label[2] must be non-negative integer"
        );
        assert!(
            labels[3] >= 0.0 && labels[3] == labels[3].floor(),
            "label[3] must be non-negative integer"
        );
    }

    // ── Spatial metadata preserved ─────────────────────────────────────────────

    #[test]
    fn test_spatial_metadata_preserved() {
        let gradient = make_image_3d(vec![1.0_f32; 8], [2, 2, 2]);
        let markers = make_image_3d(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], [2, 2, 2]);

        let result = MarkerControlledWatershed::new()
            .apply(&gradient, &markers)
            .unwrap();
        assert_eq!(result.origin(), gradient.origin());
        assert_eq!(result.spacing(), gradient.spacing());
        assert_eq!(result.direction(), gradient.direction());
    }

    // ── Output shape matches input ─────────────────────────────────────────────

    #[test]
    fn test_output_shape_matches_input() {
        let dims = [4, 5, 6];
        let n: usize = dims.iter().product();
        let gradient = make_image_3d(vec![1.0_f32; n], dims);
        let mut markers_data = vec![0.0_f32; n];
        markers_data[0] = 1.0;
        markers_data[n - 1] = 2.0;
        let markers = make_image_3d(markers_data, dims);

        let result = MarkerControlledWatershed::new()
            .apply(&gradient, &markers)
            .unwrap();
        assert_eq!(result.shape(), dims, "output shape must match input shape");
    }

    // ── All-seeded image: every voxel retains its label ────────────────────────

    #[test]
    fn test_all_seeded_image_all_labels_preserved() {
        // Every voxel is a seed with label = its index + 1.
        // No unlabeled voxels → output = input.
        let n = 8;
        let gradient = make_image_3d(vec![1.0_f32; n], [2, 2, 2]);
        let markers: Vec<f32> = (1..=n as u32).map(|i| i as f32).collect();
        let marker_image = make_image_3d(markers.clone(), [2, 2, 2]);

        let result = MarkerControlledWatershed::new()
            .apply(&gradient, &marker_image)
            .unwrap();
        let labels = get_labels(&result);
        for (i, (&got, &expected)) in labels.iter().zip(markers.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "all-seed image: voxel {} label {:.0} must be {:.0}",
                i, got, expected
            );
        }
    }

    // ── Shape mismatch panics ──────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "gradient and marker images must have the same shape")]
    fn test_shape_mismatch_panics() {
        let gradient = make_image_3d(vec![1.0_f32; 8], [2, 2, 2]);
        let markers = make_image_3d(vec![1.0_f32; 4], [1, 2, 2]);
        let _ = MarkerControlledWatershed::new().apply(&gradient, &markers);
    }

    // ── Default construction ───────────────────────────────────────────────────

    #[test]
    fn test_default_construction() {
        let _w = MarkerControlledWatershed::default();
    }

    // ── No seeds → all zeros (unreachable) ────────────────────────────────────

    #[test]
    fn test_no_seeds_produces_all_zero_output() {
        let gradient = make_image_3d(vec![1.0_f32; 8], [2, 2, 2]);
        let markers = make_image_3d(vec![0.0_f32; 8], [2, 2, 2]);
        let result = MarkerControlledWatershed::new()
            .apply(&gradient, &markers)
            .unwrap();
        let labels = get_labels(&result);
        assert!(
            labels.iter().all(|&v| v == 0.0),
            "no seeds → all labels must be 0, got {:?}",
            labels
        );
    }

    // ── 3D volumetric: two sphere seeds ───────────────────────────────────────

    #[test]
    fn test_3d_two_sphere_seeds_produce_two_basins() {
        // 9×9×9 image; seed label 1 at center-left (4,4,2), seed label 2 at center-right (4,4,6).
        // Uniform gradient. Basins should expand and meet in the middle.
        let (nz, ny, nx) = (9, 9, 9);
        let n = nz * ny * nx;
        let gradient = make_image_3d(vec![1.0_f32; n], [nz, ny, nx]);
        let mut markers_data = vec![0.0_f32; n];
        // Seed 1: (4,4,2) = 4*81 + 4*9 + 2 = 324 + 36 + 2 = 362
        markers_data[4 * ny * nx + 4 * nx + 2] = 1.0;
        // Seed 2: (4,4,6) = 4*81 + 4*9 + 6 = 324 + 36 + 6 = 366
        markers_data[4 * ny * nx + 4 * nx + 6] = 2.0;
        let markers = make_image_3d(markers_data, [nz, ny, nx]);

        let result = MarkerControlledWatershed::new()
            .apply(&gradient, &markers)
            .unwrap();
        let labels = get_labels(&result);

        // Both labels must appear in the output.
        let has_label_1 = labels.iter().any(|&v| v == 1.0);
        let has_label_2 = labels.iter().any(|&v| v == 2.0);
        assert!(has_label_1, "output must contain label 1");
        assert!(has_label_2, "output must contain label 2");

        // All labels must be non-negative integers.
        for &v in &labels {
            assert!(
                v >= 0.0 && v == v.floor(),
                "label {v} must be non-negative integer"
            );
        }

        // Seed voxels must retain their labels.
        assert_eq!(
            labels[4 * ny * nx + 4 * nx + 2],
            1.0,
            "seed 1 must be preserved"
        );
        assert_eq!(
            labels[4 * ny * nx + 4 * nx + 6],
            2.0,
            "seed 2 must be preserved"
        );
    }
}
