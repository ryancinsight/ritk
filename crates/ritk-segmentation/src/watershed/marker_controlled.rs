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
//! Flooding follows `itk::MorphologicalWatershedFromMarkersImageFilter` exactly:
//! a hierarchical FIFO (`fah`), one FIFO queue per gray level, processed in
//! ascending gray order, with a neighbour at gray ≤ the current level pushed into
//! the current level's queue. This is bit-exact to
//! `sitk.MorphologicalWatershedFromMarkers` across all
//! (`mark_watershed_line`, `fully_connected`) combinations.
//!
//! 1. **With watershed lines** (`mark_watershed_line = true`, ITK default): the
//!    queue holds *unlabelled* voxels (the seed neighbours). When popped, a voxel
//!    takes the single distinct basin label among its neighbours; if two or more
//!    distinct basins meet it stays label 0 (a watershed line) and does **not**
//!    propagate, so basins cannot leak across the line.
//! 2. **Without watershed lines** (`mark_watershed_line = false`): the queue
//!    holds *labelled* voxels (the seeds). When popped, a voxel propagates its own
//!    label to unlabelled neighbours — first front to arrive claims a voxel, and
//!    there are no lines.
//!
//! Connectivity is face (6-/4-) by default, or full (26-/8-) when
//! `fully_connected`. Output label 0 = watershed line or unreachable voxel; seed
//! labels are preserved.
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

use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
use std::collections::{BTreeMap, VecDeque};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Marker-controlled watershed segmentation.
///
/// Floods from explicit seed (marker) regions using a min-heap ordered by
/// gradient intensity. Regions expand in order of increasing gradient magnitude;
/// voxels adjacent to two distinct basins become watershed boundaries (label 0).
#[derive(Debug, Clone, Copy)]
pub struct MarkerControlledWatershed {
    /// Use 26- (3-D) / 8- (2-D) connectivity instead of face connectivity.
    /// ITK `FullyConnected`, default `false`.
    pub fully_connected: bool,
    /// Mark voxels between two basins as watershed lines (label 0). ITK
    /// `MarkWatershedLine`, default `true`.
    pub mark_watershed_line: bool,
}

impl MarkerControlledWatershed {
    /// Create a new `MarkerControlledWatershed` filter (face connectivity,
    /// watershed lines on).
    pub fn new() -> Self {
        Self {
            fully_connected: false,
            mark_watershed_line: true,
        }
    }

    /// Set 26-/8-connectivity (ITK `FullyConnected`).
    pub fn with_fully_connected(mut self, fully_connected: bool) -> Self {
        self.fully_connected = fully_connected;
        self
    }

    /// Set whether to mark watershed-line voxels (ITK `MarkWatershedLine`).
    pub fn with_mark_watershed_line(mut self, mark: bool) -> Self {
        self.mark_watershed_line = mark;
        self
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

        let labels = marker_controlled_flooding(
            &g_vals,
            &m_vals,
            dims_g,
            self.fully_connected,
            self.mark_watershed_line,
        );

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

/// All 26 neighbour offsets for a 3D grid (full connectivity); for a `z = 1`
/// image the `±z` rows are simply out of bounds, reducing to 8-connectivity.
fn full_offsets() -> Vec<(i64, i64, i64)> {
    let mut v = Vec::with_capacity(26);
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if (dz, dy, dx) != (0, 0, 0) {
                    v.push((dz, dy, dx));
                }
            }
        }
    }
    v
}

/// Marker-controlled watershed flooding on flat voxel arrays.
///
/// Returns a `Vec<f32>` of the same length as `grad_vals` containing integer
/// labels encoded as `f32`: 0.0 for boundaries/unreachable, ≥1.0 for basins.
///
/// Mirrors `itkMorphologicalWatershedFromMarkersImageFilter`: a hierarchical
/// FIFO (`fah`) per gray level, flooded in ascending gray order. With
/// `mark_line` the queue holds unlabelled voxels that derive their label from
/// their neighbours and become a watershed line on collision (and do not
/// propagate); without it, the queue holds labelled voxels that propagate their
/// own label, first-front-wins (no lines).
fn marker_controlled_flooding(
    grad_vals: &[f32],
    marker_vals: &[f32],
    dims: [usize; 3],
    fully_connected: bool,
    mark_line: bool,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    debug_assert_eq!(grad_vals.len(), n);
    debug_assert_eq!(marker_vals.len(), n);

    if n == 0 {
        return Vec::new();
    }

    const UNLABELED: u32 = u32::MAX;

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
    let face;
    let full;
    let offsets: &[(i64, i64, i64)] = if fully_connected {
        full = full_offsets();
        &full
    } else {
        face = FACE_OFFSETS;
        &face
    };
    let for_neighbors = |idx: usize, f: &mut dyn FnMut(usize)| {
        let z = idx / (ny * nx);
        let rem = idx % (ny * nx);
        let (y, x) = (rem / nx, rem % nx);
        for &(dz, dy, dx) in offsets {
            let (zi, yi, xi) = (z as i64 + dz, y as i64 + dy, x as i64 + dx);
            if zi >= 0 && zi < nz as i64 && yi >= 0 && yi < ny as i64 && xi >= 0 && xi < nx as i64 {
                f(flat(zi as usize, yi as usize, xi as usize));
            }
        }
    };

    let mut fah: BTreeMap<u32, VecDeque<usize>> = BTreeMap::new();

    if mark_line {
        // Seed the queue with unlabelled neighbours of every marker voxel.
        let mut in_queue = vec![false; n];
        for idx in 0..n {
            if labels[idx] == UNLABELED {
                continue;
            }
            for_neighbors(idx, &mut |ni| {
                if labels[ni] == UNLABELED && !in_queue[ni] {
                    in_queue[ni] = true;
                    fah.entry(gray_key(grad_vals[ni]))
                        .or_default()
                        .push_back(ni);
                }
            });
        }

        while let Some(&current_key) = fah.keys().next() {
            let mut current_queue = fah.remove(&current_key).unwrap();
            while let Some(idx) = current_queue.pop_front() {
                let mut nbr_labels: [u32; 26] = [0u32; 26];
                let mut n_distinct = 0usize;
                for_neighbors(idx, &mut |ni| {
                    let lbl = labels[ni];
                    if lbl != UNLABELED && lbl != 0 && !nbr_labels[..n_distinct].contains(&lbl) {
                        nbr_labels[n_distinct] = lbl;
                        n_distinct += 1;
                    }
                });

                labels[idx] = if n_distinct == 1 { nbr_labels[0] } else { 0 };

                // Collision (≥2 basins) → watershed line, does NOT propagate.
                if n_distinct >= 2 {
                    continue;
                }
                for_neighbors(idx, &mut |ni| {
                    if labels[ni] == UNLABELED && !in_queue[ni] {
                        in_queue[ni] = true;
                        let k = gray_key(grad_vals[ni]);
                        if k <= current_key {
                            current_queue.push_back(ni);
                        } else {
                            fah.entry(k).or_default().push_back(ni);
                        }
                    }
                });
            }
        }
    } else {
        // No watershed lines: each marker voxel with an unlabelled neighbour
        // seeds the queue and propagates its own label; the first front to reach
        // a voxel claims it.
        for idx in 0..n {
            if labels[idx] == UNLABELED {
                continue;
            }
            let mut has_unlabeled = false;
            for_neighbors(idx, &mut |ni| {
                if labels[ni] == UNLABELED {
                    has_unlabeled = true;
                }
            });
            if has_unlabeled {
                fah.entry(gray_key(grad_vals[idx]))
                    .or_default()
                    .push_back(idx);
            }
        }

        while let Some(&current_key) = fah.keys().next() {
            let mut current_queue = fah.remove(&current_key).unwrap();
            while let Some(idx) = current_queue.pop_front() {
                let marker = labels[idx];
                let mut to_push: Vec<usize> = Vec::new();
                for_neighbors(idx, &mut |ni| {
                    if labels[ni] == UNLABELED {
                        labels[ni] = marker;
                        to_push.push(ni);
                    }
                });
                for ni in to_push {
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
