//! Watershed segmentation (Meyer 1994 flooding algorithm).
//!
//! # Mathematical Specification
//!
//! The watershed transform partitions a grayscale image (typically a gradient
//! magnitude image) into labelled catchment basins separated by watershed lines.
//!
//! ## Algorithm (Immersion Simulation)
//!
//! Given a scalar image I defined on a 3D grid with 6-connected neighbourhood:
//!
//! 1. **Sort** all voxels by intensity in ascending order (ties broken by
//!    linear index for determinism).
//! 2. **Process** each voxel v in sorted order:
//!    a. Collect the set L of distinct labels among v's 6-connected neighbours
//!    that have already been labelled (label > 0).
//!    b. If L = ∅ → assign v a new unique label (next_label += 1).
//!    c. If |L| = 1 → assign v the single label in L.
//!    d. If |L| > 1 → mark v as a watershed boundary (label = 0).
//! 3. **Output**: a label image where each voxel holds its basin label
//!    (1, 2, …, K) or 0 for watershed boundaries.
//!
//! ## Properties
//!
//! - **Determinism**: voxel processing order is fully determined by
//!   (intensity, linear_index) sort key.
//! - **Connectedness**: each basin is 6-connected by construction.
//! - **Boundary completeness**: every pair of adjacent basins is separated
//!   by at least one watershed voxel (label 0).
//!
//! # Complexity
//!
//! - Sorting:    O(n log n) where n = total voxels.
//! - Labelling:  O(n) with constant-time (≤6) neighbour lookups per voxel.
//! - Total:      O(n log n).
//! - Memory:     O(n) for the label map and sorted index array.
//!
//! # References
//!
//! - Meyer, F. (1994). "Topographic distance and watershed lines."
//!   *Signal Processing*, 38(1), 113–125.

pub mod marker_controlled;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
pub use marker_controlled::MarkerControlledWatershed;
use ritk_tensor_ops::extract_vec;
use ritk_image::Image;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Watershed segmentation via Meyer's flooding algorithm.
///
/// Operates on 3D images (typically gradient magnitude). Each voxel is assigned
/// a basin label (≥ 1) or marked as a watershed boundary (0).
#[derive(Debug, Clone)]
pub struct WatershedSegmentation {}

impl WatershedSegmentation {
    /// Create a new `WatershedSegmentation` filter.
    pub fn new() -> Self {
        Self {}
    }

    /// Apply watershed segmentation to a 3D image.
    ///
    /// The input should be a scalar 3D image (e.g. gradient magnitude).
    /// Returns a label image of the same shape where:
    /// - Label 0 = watershed boundary.
    /// - Label 1, 2, … K = catchment basin indices (encoded as `f32`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let device = image.data().device();
        let (vals, _) = extract_vec(image)?;

        let labels = watershed_flooding(&vals, dims);

        let tensor = Tensor::<B, 3>::from_data(TensorData::new(labels, Shape::new(dims)), &device);

        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

impl Default for WatershedSegmentation {
    fn default() -> Self {
        Self::new()
    }
}

// ── Core implementation ────────────────────────────────────────────────────────

/// 6-connected neighbour offsets for a 3D grid (±z, ±y, ±x).
const NEIGHBOUR_OFFSETS: [(i64, i64, i64); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Meyer's flooding watershed on a flat voxel array with shape `[nz, ny, nx]`.
///
/// Returns a `Vec<f32>` of the same length as `vals`, containing integer labels
/// encoded as `f32`: 0.0 for watershed boundaries, 1.0, 2.0, … for basins.
fn watershed_flooding(vals: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    debug_assert_eq!(vals.len(), n, "vals length must equal product of dims");

    if n == 0 {
        return Vec::new();
    }

    // ── 1. Sort voxels by (intensity, linear_index) ────────────────────────────
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        vals[a]
            .partial_cmp(&vals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // ── 2. Process voxels in ascending intensity order ─────────────────────────
    // Label 0 is reserved for "unlabelled" during processing, and later also
    // used for watershed boundaries. We use a separate sentinel (u32::MAX) to
    // distinguish "not yet visited" from "watershed boundary (0)".
    const UNVISITED: u32 = u32::MAX;
    const WATERSHED: u32 = 0;

    let mut labels = vec![UNVISITED; n];
    let mut next_label: u32 = 1;

    for &idx in &indices {
        // Decompose linear index → (z, y, x).
        let z = idx / (ny * nx);
        let rem = idx % (ny * nx);
        let y = rem / nx;
        let x = rem % nx;

        // Collect distinct non-zero, non-unvisited labels among 6-neighbours.
        let mut neighbour_labels: [u32; 6] = [UNVISITED; 6];
        let mut n_distinct = 0usize;

        for &(dz, dy, dx) in &NEIGHBOUR_OFFSETS {
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

            let ni = nz_i as usize * ny * nx + ny_i as usize * nx + nx_i as usize;
            let lbl = labels[ni];

            // Only consider assigned basin labels (> 0, not UNVISITED, not WATERSHED).
            if lbl == UNVISITED || lbl == WATERSHED {
                continue;
            }

            // Check if this label is already recorded.
            let mut found = false;
            for &nl in neighbour_labels.iter().take(n_distinct) {
                if nl == lbl {
                    found = true;
                    break;
                }
            }
            if !found {
                neighbour_labels[n_distinct] = lbl;
                n_distinct += 1;
            }
        }

        // Assign label according to Meyer's rules.
        labels[idx] = match n_distinct {
            0 => {
                // No labelled neighbours → new basin.
                let lbl = next_label;
                next_label = next_label.saturating_add(1);
                lbl
            }
            1 => {
                // Single neighbouring basin → extend it.
                neighbour_labels[0]
            }
            _ => {
                // Multiple distinct neighbouring basins → watershed boundary.
                WATERSHED
            }
        };
    }

    // ── 3. Convert to f32 label image ──────────────────────────────────────────
    // UNVISITED should not remain if n > 0, but defensively map it to 0.
    labels
        .iter()
        .map(|&lbl| {
            if lbl == UNVISITED {
                0.0_f32
            } else {
                lbl as f32
            }
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_watershed.rs"]
mod tests;
