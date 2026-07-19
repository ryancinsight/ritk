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
//! ([`WatershedLinePolicy`], [`FloodConnectivity`]) combinations.
//!
//! 1. **With watershed lines** ([`WatershedLinePolicy::Mark`], ITK default): the
//!    queue holds *unlabelled* voxels (the seed neighbours). When popped, a voxel
//!    takes the single distinct basin label among its neighbours; if two or more
//!    distinct basins meet it stays label 0 (a watershed line) and does **not**
//!    propagate, so basins cannot leak across the line.
//! 2. **Without watershed lines** ([`WatershedLinePolicy::Omit`]): the queue
//!    holds *labelled* voxels (the seeds). When popped, a voxel propagates its own
//!    label to unlabelled neighbours — first front to arrive claims a voxel, and
//!    there are no lines.
//!
//! Connectivity is face (6-/4-) by default, or full (26-/8-) when
//! [`FloodConnectivity::Full`]. Output label 0 = watershed line or unreachable voxel; seed
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

use ritk_image::tensor::{Backend, Tensor};
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
    connectivity: FloodConnectivity,
    watershed_lines: WatershedLinePolicy,
}

/// Neighborhood policy for marker-controlled flooding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloodConnectivity {
    /// Face connectivity: 6-neighbor in 3-D and 4-neighbor when `z == 1`.
    Face,
    /// Full connectivity: 26-neighbor in 3-D and 8-neighbor when `z == 1`.
    Full,
}

/// Boundary policy when two flooding fronts meet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WatershedLinePolicy {
    /// Emit label-zero watershed lines.
    Mark,
    /// Assign every reachable voxel to the first arriving basin.
    Omit,
}

impl MarkerControlledWatershed {
    /// Create a new `MarkerControlledWatershed` filter (face connectivity,
    /// watershed lines on).
    pub fn new() -> Self {
        Self {
            connectivity: FloodConnectivity::Face,
            watershed_lines: WatershedLinePolicy::Mark,
        }
    }

    /// Set the flooding neighborhood policy.
    #[must_use]
    pub fn with_connectivity(mut self, connectivity: FloodConnectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Set the watershed-line policy.
    #[must_use]
    pub fn with_watershed_lines(mut self, watershed_lines: WatershedLinePolicy) -> Self {
        self.watershed_lines = watershed_lines;
        self
    }

    /// Return the flooding neighborhood policy.
    pub fn connectivity(&self) -> FloodConnectivity {
        self.connectivity
    }

    /// Return the watershed-line policy.
    pub fn watershed_lines(&self) -> WatershedLinePolicy {
        self.watershed_lines
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
    /// # Errors
    ///
    /// Returns an error for shape or geometry mismatch, zero dimensions,
    /// non-finite/negative gradient samples, or marker labels that are
    /// non-finite, negative, fractional, or not exactly representable as f32.
    pub fn apply<B: Backend>(
        &self,
        gradient: &Image<f32, B, 3>,
        markers: &Image<f32, B, 3>,
    ) -> anyhow::Result<Image<f32, B, 3>> {
        let (g_vals, dims_g) = extract_vec_infallible(gradient);
        let (m_vals, dims_m) = extract_vec_infallible(markers);
        validate_inputs(
            &g_vals,
            &m_vals,
            dims_g,
            dims_m,
            gradient.origin() == markers.origin(),
            gradient.spacing() == markers.spacing(),
            gradient.direction() == markers.direction(),
        )?;

        let device = B::default();

        let labels = marker_controlled_flooding(
            &g_vals,
            &m_vals,
            dims_g,
            self.connectivity,
            self.watershed_lines,
        );

        let tensor = Tensor::<f32, B>::from_slice_on(dims_g, &labels, &device);

        Image::new(
            tensor,
            *gradient.origin(),
            *gradient.spacing(),
            *gradient.direction(),
        )
    }

    /// Apply marker-controlled flooding to Coeus-native images.
    ///
    /// # Errors
    ///
    /// Returns the validation errors documented by [`Self::apply`], or a
    /// backend storage/output construction error.
    pub fn apply_native<B>(
        &self,
        gradient: &ritk_image::Image<f32, B, 3>,
        markers: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let gradient_values = gradient.data_slice()?;
        let marker_values = markers.data_slice()?;
        validate_inputs(
            gradient_values,
            marker_values,
            gradient.shape(),
            markers.shape(),
            gradient.origin() == markers.origin(),
            gradient.spacing() == markers.spacing(),
            gradient.direction() == markers.direction(),
        )?;
        crate::native_output::from_values(
            gradient,
            marker_controlled_flooding(
                gradient_values,
                marker_values,
                gradient.shape(),
                self.connectivity,
                self.watershed_lines,
            ),
            backend,
        )
    }
}

impl Default for MarkerControlledWatershed {
    fn default() -> Self {
        Self::new()
    }
}

// ── Core implementation ────────────────────────────────────────────────────────

const MAX_EXACT_LABEL: f32 = (1u32 << f32::MANTISSA_DIGITS) as f32;

fn validate_inputs(
    gradient: &[f32],
    markers: &[f32],
    gradient_shape: [usize; 3],
    marker_shape: [usize; 3],
    same_origin: bool,
    same_spacing: bool,
    same_direction: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        gradient_shape == marker_shape,
        "gradient and marker shapes must match: {gradient_shape:?} vs {marker_shape:?}"
    );
    anyhow::ensure!(
        gradient_shape.iter().all(|&extent| extent > 0),
        "marker watershed requires nonzero dimensions, got {gradient_shape:?}"
    );
    let expected = gradient_shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| {
            anyhow::anyhow!("marker watershed shape product overflows usize: {gradient_shape:?}")
        })?;
    anyhow::ensure!(
        gradient.len() == expected && markers.len() == expected,
        "marker watershed shape {gradient_shape:?} requires {expected} samples, got gradient={} markers={}",
        gradient.len(),
        markers.len()
    );
    anyhow::ensure!(same_origin, "gradient and marker origins must match");
    anyhow::ensure!(same_spacing, "gradient and marker spacing must match");
    anyhow::ensure!(same_direction, "gradient and marker directions must match");
    if let Some((index, value)) = gradient
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value < 0.0)
    {
        anyhow::bail!("marker watershed gradient at flat index {index} must be finite and nonnegative, got {value}");
    }
    if let Some((index, value)) = markers.iter().copied().enumerate().find(|(_, value)| {
        !value.is_finite() || *value < 0.0 || value.fract() != 0.0 || *value > MAX_EXACT_LABEL
    }) {
        anyhow::bail!("marker watershed label at flat index {index} must be a finite nonnegative integer no greater than {MAX_EXACT_LABEL}, got {value}");
    }
    Ok(())
}

/// Totally-ordered key for a validated non-negative finite `f32` gradient.
#[inline]
fn gray_key(grad: f32) -> u32 {
    grad.to_bits()
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

/// All 26 neighbour offsets for full 3-D connectivity.
const FULL_OFFSETS: [(i64, i64, i64); 26] = [
    (-1, -1, -1),
    (-1, -1, 0),
    (-1, -1, 1),
    (-1, 0, -1),
    (-1, 0, 0),
    (-1, 0, 1),
    (-1, 1, -1),
    (-1, 1, 0),
    (-1, 1, 1),
    (0, -1, -1),
    (0, -1, 0),
    (0, -1, 1),
    (0, 0, -1),
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
];

#[inline]
fn visit_neighbors<F>(
    index: usize,
    dimensions: [usize; 3],
    connectivity: FloodConnectivity,
    mut visit: F,
) where
    F: FnMut(usize),
{
    let [depth, height, width] = dimensions;
    let plane = height * width;
    let z = index / plane;
    let remainder = index % plane;
    let (y, x) = (remainder / width, remainder % width);
    let offsets: &[(i64, i64, i64)] = match connectivity {
        FloodConnectivity::Face => &FACE_OFFSETS,
        FloodConnectivity::Full => &FULL_OFFSETS,
    };
    for &(dz, dy, dx) in offsets {
        let (neighbor_z, neighbor_y, neighbor_x) = (z as i64 + dz, y as i64 + dy, x as i64 + dx);
        if neighbor_z >= 0
            && neighbor_z < depth as i64
            && neighbor_y >= 0
            && neighbor_y < height as i64
            && neighbor_x >= 0
            && neighbor_x < width as i64
        {
            visit(neighbor_z as usize * plane + neighbor_y as usize * width + neighbor_x as usize);
        }
    }
}

/// Marker-controlled watershed flooding on flat voxel arrays.
///
/// Returns a `Vec<f32>` of the same length as `grad_vals` containing integer
/// labels encoded as `f32`: 0.0 for boundaries/unreachable, ≥1.0 for basins.
///
/// Mirrors `itkMorphologicalWatershedFromMarkersImageFilter`: a hierarchical
/// FIFO (`fah`) per gray level, flooded in ascending gray order. With
/// [`WatershedLinePolicy::Mark`] makes the queue hold unlabelled voxels that derive their label from
/// their neighbours and become a watershed line on collision (and do not
/// propagate); without it, the queue holds labelled voxels that propagate their
/// own label, first-front-wins (no lines).
fn marker_controlled_flooding(
    grad_vals: &[f32],
    marker_vals: &[f32],
    dims: [usize; 3],
    connectivity: FloodConnectivity,
    watershed_lines: WatershedLinePolicy,
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
        .map(|&v| if v > 0.0 { v as u32 } else { UNLABELED })
        .collect();

    let mut fah: BTreeMap<u32, VecDeque<usize>> = BTreeMap::new();

    if watershed_lines == WatershedLinePolicy::Mark {
        // Seed the queue with unlabelled neighbours of every marker voxel.
        let mut in_queue = vec![false; n];
        for idx in 0..n {
            if labels[idx] == UNLABELED {
                continue;
            }
            visit_neighbors(idx, dims, connectivity, |ni| {
                if labels[ni] == UNLABELED && !in_queue[ni] {
                    in_queue[ni] = true;
                    fah.entry(gray_key(grad_vals[ni]))
                        .or_default()
                        .push_back(ni);
                }
            });
        }

        while let Some(&current_key) = fah.keys().next() {
            let mut current_queue = fah
                .remove(&current_key)
                .expect("invariant: selected gray level exists");
            while let Some(idx) = current_queue.pop_front() {
                let mut nbr_labels: [u32; 26] = [0u32; 26];
                let mut n_distinct = 0usize;
                visit_neighbors(idx, dims, connectivity, |ni| {
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
                visit_neighbors(idx, dims, connectivity, |ni| {
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
            visit_neighbors(idx, dims, connectivity, |ni| {
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
            let mut current_queue = fah
                .remove(&current_key)
                .expect("invariant: selected gray level exists");
            while let Some(idx) = current_queue.pop_front() {
                let marker = labels[idx];
                let mut to_push: Vec<usize> = Vec::new();
                visit_neighbors(idx, dims, connectivity, |ni| {
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
