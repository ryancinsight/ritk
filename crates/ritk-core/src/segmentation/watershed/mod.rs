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
//!       that have already been labelled (label > 0).
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
pub use marker_controlled::MarkerControlledWatershed;

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

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

        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("WatershedSegmentation requires f32 data: {:?}", e))?
            .to_vec();

        let labels = watershed_flooding(&vals, dims);

        let tensor = Tensor::<B, 3>::from_data(TensorData::new(labels, Shape::new(dims)), &device);

        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
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
            for k in 0..n_distinct {
                if neighbour_labels[k] == lbl {
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

    // ── Constant / uniform image ───────────────────────────────────────────────

    #[test]
    fn test_constant_image_single_basin() {
        // All voxels have the same intensity → processed in index order.
        // The first voxel gets label 1; every subsequent voxel is 6-adjacent
        // to an already-labelled voxel with label 1 → all get label 1.
        let dims = [3, 3, 3];
        let n: usize = dims.iter().product();
        let data = vec![5.0_f32; n];
        let image = make_image_3d(data, dims);
        let result = WatershedSegmentation::new().apply(&image).unwrap();
        let labels = get_labels(&result);

        // Every voxel should have the same non-zero label.
        assert!(
            labels.iter().all(|&v| v == 1.0),
            "constant image must produce a single basin (label 1), got labels: {:?}",
            labels
        );
    }

    // ── Two separated minima → two basins + watershed boundary ─────────────────

    #[test]
    fn test_two_minima_produce_two_basins_with_boundary() {
        // 1×1×5 image: [0, 10, 100, 10, 0]
        // Two local minima at index 0 and 4. The ridge voxel at index 2
        // (intensity 100) should become a watershed boundary (label 0) because
        // when it is processed last, both basin labels are among its neighbours.
        let dims = [1, 1, 5];
        let data = vec![0.0, 10.0, 100.0, 10.0, 0.0];
        let image = make_image_3d(data, dims);
        let result = WatershedSegmentation::new().apply(&image).unwrap();
        let labels = get_labels(&result);

        // Voxels 0 and 4 have the lowest intensities; they get distinct labels.
        assert!(labels[0] > 0.0, "minimum voxel must have a basin label");
        assert!(labels[4] > 0.0, "minimum voxel must have a basin label");
        assert!(
            (labels[0] - labels[4]).abs() > f32::EPSILON,
            "two separated minima must get distinct labels: {} vs {}",
            labels[0],
            labels[4]
        );

        // The ridge voxel (index 2) should be a watershed boundary.
        assert!(
            labels[2] == 0.0,
            "ridge voxel between two basins must be watershed (0), got {}",
            labels[2]
        );
    }

    // ── Output shape matches input shape ───────────────────────────────────────

    #[test]
    fn test_output_shape_matches_input() {
        let dims = [4, 5, 6];
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 10.0).collect();
        let image = make_image_3d(data, dims);
        let result = WatershedSegmentation::new().apply(&image).unwrap();
        assert_eq!(result.shape(), dims, "output shape must match input shape");
    }

    // ── Spatial metadata preserved ─────────────────────────────────────────────

    #[test]
    fn test_spatial_metadata_preserved() {
        let dims = [2, 2, 2];
        let data = vec![0.0_f32; 8];
        let image = make_image_3d(data, dims);
        let result = WatershedSegmentation::new().apply(&image).unwrap();

        assert_eq!(result.origin(), image.origin());
        assert_eq!(result.spacing(), image.spacing());
        assert_eq!(result.direction(), image.direction());
    }

    // ── Labels are non-negative integers ───────────────────────────────────────

    #[test]
    fn test_labels_are_nonneg_integers() {
        let dims = [3, 3, 3];
        let n: usize = dims.iter().product();
        // Gradient-like image with a saddle to force watershed boundaries.
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let z = i / 9;
                let y = (i % 9) / 3;
                let x = i % 3;
                // Two minima at corners (0,0,0) and (2,2,2); ridge in between.
                let d0 = ((z * z + y * y + x * x) as f32).sqrt();
                let d1 =
                    (((2 - z) * (2 - z) + (2 - y) * (2 - y) + (2 - x) * (2 - x)) as f32).sqrt();
                d0.min(d1) * 50.0
            })
            .collect();
        let image = make_image_3d(data, dims);
        let result = WatershedSegmentation::new().apply(&image).unwrap();
        let labels = get_labels(&result);

        for (i, &v) in labels.iter().enumerate() {
            assert!(
                v >= 0.0 && v == v.floor(),
                "label at voxel {} must be a non-negative integer, got {}",
                i,
                v
            );
        }
    }

    // ── Default delegates to new ───────────────────────────────────────────────

    #[test]
    fn test_default_construction() {
        let _ws = WatershedSegmentation::default();
    }

    // ── Single voxel → single basin ────────────────────────────────────────────

    #[test]
    fn test_single_voxel_single_basin() {
        let dims = [1, 1, 1];
        let data = vec![42.0_f32];
        let image = make_image_3d(data, dims);
        let result = WatershedSegmentation::new().apply(&image).unwrap();
        let labels = get_labels(&result);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], 1.0, "single voxel must be labelled 1");
    }
}
