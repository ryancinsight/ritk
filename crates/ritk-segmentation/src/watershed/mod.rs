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
//! 1. **Sort** all voxels by intensity in ascending order, treating both signed
//!    zero representations as the same relief level.
//! 2. **Process each equal-height level as one batch**. Mask its voxels, then
//!    propagate already-labelled lower basins across the plateau by geodesic
//!    distance. Equidistant collisions become watershed lines.
//! 3. **Label remaining components** at the level as new regional-minimum
//!    basins before advancing to the next height.
//! 4. **Output**: a label image where each voxel holds its basin label
//!    (1, 2, …, K) or 0 for watershed boundaries.
//!
//! ## Properties
//!
//! - **Determinism**: relief levels are totally ordered and plateau queues use
//!   linear-index discovery order without changing geodesic distances.
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
//! - Vincent, L. and Soille, P. (1991). "Watersheds in digital spaces: an
//!   efficient algorithm based on immersion simulations." *IEEE TPAMI*,
//!   13(6), 583–598. DOI: 10.1109/34.87344.

pub mod isolated;
pub mod marker_controlled;
pub mod morphological;
pub mod toboggan;
pub use isolated::IsolatedWatershed;
pub use marker_controlled::{FloodConnectivity, MarkerControlledWatershed, WatershedLinePolicy};
pub use morphological::MorphologicalWatershed;
use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;
use std::collections::VecDeque;
pub use toboggan::toboggan;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Watershed segmentation via Meyer's flooding algorithm.
///
/// Operates on 3D images (typically gradient magnitude). Each voxel is assigned
/// a basin label (≥ 1) or marked as a watershed boundary (0).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WatershedSegmentation;

impl WatershedSegmentation {
    /// Create a new `WatershedSegmentation` filter.
    pub fn new() -> Self {
        Self
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
    /// Returns an error if storage cannot be read, any extent is zero, shape
    /// cardinality overflows, the storage length differs from the shape, the
    /// relief contains a non-finite sample, or the volume can produce more
    /// basin labels than `f32` represents exactly.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let device = image.data().device();
        let (vals, _) = extract_vec(image)?;
        validate_relief(&vals, dims)?;
        let labels = watershed_flooding(&vals, dims);

        let tensor = Tensor::<B, 3>::from_data(TensorData::new(labels, Shape::new(dims)), &device);

        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }

    /// Apply Meyer flooding to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns the validation errors documented by [`Self::apply`], or a
    /// backend storage/output construction error.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = image.data_slice()?;
        validate_relief(values, image.shape())?;
        crate::native_output::from_values(image, watershed_flooding(values, image.shape()), backend)
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

const MAX_EXACT_LABEL_COUNT: usize = 1usize << f32::MANTISSA_DIGITS;

fn validate_relief(values: &[f32], dimensions: [usize; 3]) -> anyhow::Result<()> {
    anyhow::ensure!(
        dimensions.iter().all(|&extent| extent > 0),
        "Meyer watershed requires nonzero dimensions, got {dimensions:?}"
    );
    let expected = dimensions
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| {
            anyhow::anyhow!("Meyer watershed shape product overflows usize: {dimensions:?}")
        })?;
    anyhow::ensure!(
        expected <= MAX_EXACT_LABEL_COUNT,
        "Meyer watershed supports at most {MAX_EXACT_LABEL_COUNT} samples for exact f32 labels, got {expected}"
    );
    anyhow::ensure!(
        values.len() == expected,
        "Meyer watershed shape {dimensions:?} requires {expected} samples, got {}",
        values.len()
    );
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("Meyer watershed relief at flat index {index} must be finite, got {value}");
    }
    Ok(())
}

/// Meyer's flooding watershed on a flat voxel array with shape `[nz, ny, nx]`.
///
/// Returns a `Vec<f32>` of the same length as `vals`, containing integer labels
/// encoded as `f32`: 0.0 for watershed boundaries, 1.0, 2.0, … for basins.
fn watershed_flooding(vals: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let n = vals.len();
    debug_assert_eq!(
        dims.iter().product::<usize>(),
        n,
        "validated shape cardinality must equal relief length"
    );

    // ── 1. Sort voxels by canonical relief level and linear index ──────────────
    let mut indices: Vec<usize> = (0..n).collect();
    let canonical_level = |value: f32| if value == 0.0 { 0.0 } else { value };
    indices.sort_by(|&a, &b| {
        canonical_level(vals[a])
            .total_cmp(&canonical_level(vals[b]))
            .then(a.cmp(&b))
    });

    // Vincent-Soille/Meyer immersion states. MASK marks the active level;
    // WATERSHED is the final boundary label.
    const INIT: u32 = u32::MAX;
    const MASK: u32 = u32::MAX - 1;
    const WATERSHED: u32 = 0;
    let mut labels = vec![INIT; n];
    let mut distance = vec![0usize; n];
    let mut queue = VecDeque::new();
    let mut next_label = 0u32;
    let mut level_start = 0usize;

    while level_start < n {
        let level = canonical_level(vals[indices[level_start]]);
        let mut level_end = level_start + 1;
        while level_end < n && canonical_level(vals[indices[level_end]]) == level {
            level_end += 1;
        }

        for &index in &indices[level_start..level_end] {
            labels[index] = MASK;
            visit_face_neighbors(index, dims, |neighbor| {
                if labels[neighbor] != INIT && labels[neighbor] != MASK {
                    distance[index] = 1;
                    queue.push_back(index);
                }
            });
        }

        // Propagate lower-basin labels through this plateau in geodesic layers.
        let mut current_distance = 1usize;
        queue.push_back(n);
        while let Some(index) = queue.pop_front() {
            if index == n {
                if queue.is_empty() {
                    break;
                }
                queue.push_back(n);
                current_distance += 1;
                continue;
            }
            visit_face_neighbors(index, dims, |neighbor| {
                let neighbor_label = labels[neighbor];
                if distance[neighbor] < current_distance
                    && neighbor_label != INIT
                    && neighbor_label != MASK
                {
                    if neighbor_label > WATERSHED {
                        if labels[index] == MASK || labels[index] == WATERSHED {
                            labels[index] = neighbor_label;
                        } else if labels[index] != neighbor_label {
                            labels[index] = WATERSHED;
                        }
                    } else if neighbor_label == WATERSHED && labels[index] == MASK {
                        labels[index] = WATERSHED;
                    }
                } else if neighbor_label == MASK && distance[neighbor] == 0 {
                    distance[neighbor] = current_distance + 1;
                    queue.push_back(neighbor);
                }
            });
        }

        // Any MASK component is a regional minimum at this level.
        for &index in &indices[level_start..level_end] {
            distance[index] = 0;
            if labels[index] != MASK {
                continue;
            }
            next_label += 1;
            labels[index] = next_label;
            queue.push_back(index);
            while let Some(component_index) = queue.pop_front() {
                visit_face_neighbors(component_index, dims, |neighbor| {
                    if labels[neighbor] == MASK {
                        labels[neighbor] = next_label;
                        queue.push_back(neighbor);
                    }
                });
            }
        }
        level_start = level_end;
    }

    // ── 3. Convert to f32 label image ──────────────────────────────────────────
    debug_assert!(labels.iter().all(|&label| label != INIT && label != MASK));
    labels.into_iter().map(|label| label as f32).collect()
}

#[inline]
fn visit_face_neighbors<F>(index: usize, dimensions: [usize; 3], mut visit: F)
where
    F: FnMut(usize),
{
    let [depth, height, width] = dimensions;
    let plane = height * width;
    let z = index / plane;
    let remainder = index % plane;
    let (y, x) = (remainder / width, remainder % width);
    for &(dz, dy, dx) in &NEIGHBOUR_OFFSETS {
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

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_watershed.rs"]
mod tests;
