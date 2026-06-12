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
//!   M(x) = 0 indicates unlabeled voxels.
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

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
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
        let (g_vals, dims_g) = extract_vec_infallible(gradient);
        let (m_vals, dims_m) = extract_vec_infallible(markers);
        assert_eq!(
            dims_g, dims_m,
            "gradient and marker images must have the same shape: {:?} vs {:?}",
            dims_g, dims_m
        );

        let device = gradient.data().device();

        let labels = marker_controlled_flooding(&g_vals, &m_vals, dims_g);

        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(labels, Shape::new(dims_g)), &device);

        Ok(Image::new(
            tensor,
            *gradient.origin(),
            *gradient.spacing(),
            *gradient.direction(),
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
            if !nbr_labels[..n_distinct].contains(&lbl) {
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
#[path = "tests_marker_controlled.rs"]
mod tests;
