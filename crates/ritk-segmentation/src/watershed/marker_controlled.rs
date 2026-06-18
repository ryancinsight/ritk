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
use std::collections::{BTreeMap, VecDeque};

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

// ── Core implementation ────────────────────────────────────────────────────────

/// Totally-ordered key for an `f32` gradient. Non-negative IEEE-754 single-
/// precision bit patterns sort identically to their `u32` representation, so the
/// raw bits give an ascending gray-level key. NaN / negative gradients clamp to
/// `0.0` so the order stays valid.
#[inline]
fn gray_key(grad: f32) -> u32 {
    let g = if grad.is_nan() || grad < 0.0 {
        0.0_f32
    } else {
        grad
    };
    g.to_bits()
}

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

    // ── 1. Hierarchical FIFO (`fah`): one FIFO queue per gray level, processed
    //    in ascending gray order. A voxel's `status` (in_queue) is set when it is
    //    first enqueued, so each voxel is queued exactly once. Mirrors ITK's
    //    `itkMorphologicalWatershedFromMarkersImageFilter` priority structure.
    let mut in_queue = vec![false; n];
    let mut fah: BTreeMap<u32, VecDeque<usize>> = BTreeMap::new();

    // Helper to evaluate a voxel's 6-connected in-bounds neighbours.
    let neighbors = |idx: usize| -> [Option<usize>; 6] {
        let z = idx / (ny * nx);
        let rem = idx % (ny * nx);
        let y = rem / nx;
        let x = rem % nx;
        let mut out = [None; 6];
        for (k, &(dz, dy, dx)) in FACE_OFFSETS.iter().enumerate() {
            let (zi, yi, xi) = (z as i64 + dz, y as i64 + dy, x as i64 + dx);
            if zi >= 0
                && zi < nz as i64
                && yi >= 0
                && yi < ny as i64
                && xi >= 0
                && xi < nx as i64
            {
                out[k] = Some(flat(zi as usize, yi as usize, xi as usize));
            }
        }
        out
    };

    // Seed the queue with the unlabeled neighbours of every marker voxel.
    for idx in 0..n {
        if labels[idx] == UNLABELED {
            continue;
        }
        for ni in neighbors(idx).into_iter().flatten() {
            if labels[ni] == UNLABELED && !in_queue[ni] {
                in_queue[ni] = true;
                fah.entry(gray_key(grad_vals[ni]))
                    .or_default()
                    .push_back(ni);
            }
        }
    }

    // ── 2. Flood ascending gray levels, FIFO within each level. ────────────────
    while let Some(&current_key) = fah.keys().next() {
        let mut current_queue = fah.remove(&current_key).unwrap();
        while let Some(idx) = current_queue.pop_front() {
            // Distinct non-zero basin labels among the 6-connected neighbours.
            let mut nbr_labels: [u32; 6] = [0u32; 6];
            let mut n_distinct = 0usize;
            for ni in neighbors(idx).into_iter().flatten() {
                let lbl = labels[ni];
                if lbl == UNLABELED || lbl == 0 {
                    continue;
                }
                if !nbr_labels[..n_distinct].contains(&lbl) {
                    nbr_labels[n_distinct] = lbl;
                    n_distinct += 1;
                }
            }

            labels[idx] = match n_distinct {
                0 => 0,
                1 => nbr_labels[0],
                _ => 0,
            };

            // ITK collision rule: a watershed-line voxel (≥2 distinct basin
            // neighbours) does NOT propagate — the line is a flooding barrier.
            if n_distinct >= 2 {
                continue;
            }

            // Propagate: a neighbour at gray ≤ the current level is processed in
            // this level's FIFO (ITK's `GrayVal <= currentValue → currentQueue`);
            // otherwise it waits in its own (higher) level.
            for ni in neighbors(idx).into_iter().flatten() {
                if labels[ni] == UNLABELED && !in_queue[ni] {
                    in_queue[ni] = true;
                    let k = gray_key(grad_vals[ni]);
                    if k <= current_key {
                        current_queue.push_back(ni);
                    } else {
                        fah.entry(k).or_default().push_back(ni);
                    }
                }
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
