//! GrowCut interactive segmentation via cellular automaton.
//!
//! # Algorithm
//! Vezhnevets & Konouchine (2005) "GrowCut — Interactive Multi-Label
//! N-D Image Segmentation by Cellular Automata", GRAPHITE 2005.
//!
//! Each voxel i maintains a label L[i] ∈ {0,1,…,K} and a strength C[i] ∈ [0,1].
//! Seeds are initialized with L[i] = seed, C[i] = 1.0. Unlabeled voxels start
//! with L[i] = 0, C[i] = 0.0.  Seed voxels are immutable.
//!
//! At each iteration a labeled neighbor j "attacks" i if:
//!   C[j] · g(j,i) > C[i]    where g(j,i) = 1 − |I[j]−I[i]| / max_diff
//! On a successful attack L[i] ← L[j], C[i] ← C[j]·g(j,i).
//!
//! Terminates when no voxel changes label or `max_iter` is reached.
//!
//! # Complexity
//! O(max_iter · N · 6) = O(max_iter · N). Each iteration is fully data-parallel.
//!
//! # ITK Parity
//! `itk::FastMarchingSegmentationModule` / GrowCut filter (3D Slicer extension).

use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use rayon::prelude::*;

// ── Public types ──────────────────────────────────────────────────────────────

/// GrowCut interactive segmentation filter for 3-D images.
pub struct GrowCutFilter {
    /// Maximum number of automaton iterations. Default 200.
    pub max_iter: usize,
}

impl Default for GrowCutFilter {
    fn default() -> Self {
        Self { max_iter: 200 }
    }
}

impl GrowCutFilter {
    /// Create a `GrowCutFilter` with a specified iteration limit.
    pub fn new(max_iter: usize) -> Self {
        Self { max_iter }
    }

    /// Apply GrowCut to `image` using `seeds` as the initial label map.
    ///
    /// `seeds` must have the same shape as `image`.  Voxels with `seeds[i] > 0.5`
    /// are treated as seeds with their integer label; all others are unlabeled (0).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>, seeds: &Image<B, 3>) -> Image<B, 3> {
        growcut(image, seeds, self.max_iter)
    }
}

// ── Public function ───────────────────────────────────────────────────────────

/// Run GrowCut segmentation on a 3-D image.
///
/// `seeds` carries integer labels (stored as f32).  Voxels with seeds > 0.5
/// are treated as immutable seed voxels.  The returned image contains the final
/// integer label map (as f32), preserving the spatial metadata of `image`.
///
/// # Panics
/// Panics if `image` and `seeds` have different shapes.
pub fn growcut<B: Backend>(
    image: &Image<B, 3>,
    seeds: &Image<B, 3>,
    max_iter: usize,
) -> Image<B, 3> {
    assert_eq!(
        image.shape(),
        seeds.shape(),
        "image and seeds must have identical shapes"
    );
    let shape = image.shape();
    let (nz, ny, nx) = (shape[0], shape[1], shape[2]);

    let (img_vals, _) = extract_vec_infallible(image);
    let (seed_vals, _) = extract_vec_infallible(seeds);

    let result = growcut_slice(&img_vals, &seed_vals, [nz, ny, nx], max_iter);

    let device = image.data().device();
    let td = TensorData::new(result, Shape::new([nz, ny, nx]));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
    )
}

// ── Core automaton ────────────────────────────────────────────────────────────

/// Run GrowCut automaton on flat slices.
///
/// `image_slice`:  flat f32 voxel intensities [nz*ny*nx]
/// `seed_slice`:   flat f32 label map (values cast to u32; 0 = unlabeled)
/// Returns flat f32 label map after convergence or `max_iter` iterations.
pub fn growcut_slice(
    image_slice: &[f32],
    seed_slice: &[f32],
    dims: [usize; 3],
    max_iter: usize,
) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = nz * ny * nx;

    // Compute intensity range for adjacency weight denominator.
    let (i_min, i_max) = image_slice
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    let max_diff = (i_max - i_min).max(f32::EPSILON) as f64;

    // Initialize state: (label u32, strength f64).
    let mut labels: Vec<u32> = seed_slice.iter().map(|&v| v as u32).collect();
    let mut strengths: Vec<f64> = labels
        .iter()
        .map(|&l| if l > 0 { 1.0_f64 } else { 0.0_f64 })
        .collect();

    // Mark which voxels are immutable seeds.
    let is_seed: Vec<bool> = labels.iter().map(|&l| l > 0).collect();

    let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // 6-connectivity face offsets.
    const OFFSETS: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    let img = image_slice; // convenience alias

    for _ in 0..max_iter {
        // Parallel computation of (new_label, new_strength) per voxel.
        let updates: Vec<(u32, f64)> = (0..n)
            .into_par_iter()
            .map(|idx| {
                // If seed, keep immutable.
                if is_seed[idx] {
                    return (labels[idx], strengths[idx]);
                }

                let iz = idx / (ny * nx);
                let rem = idx % (ny * nx);
                let iy = rem / nx;
                let ix = rem % nx;

                let mut best_label = labels[idx];
                let mut best_strength = strengths[idx];

                for &(dz, dy, dx) in &OFFSETS {
                    let jz = iz as isize + dz;
                    let jy = iy as isize + dy;
                    let jx = ix as isize + dx;
                    if jz < 0
                        || jz >= nz as isize
                        || jy < 0
                        || jy >= ny as isize
                        || jx < 0
                        || jx >= nx as isize
                    {
                        continue;
                    }
                    let j = flat(jz as usize, jy as usize, jx as usize);
                    if labels[j] == 0 {
                        continue; // unlabeled neighbor cannot attack
                    }
                    let diff = (img[j] as f64 - img[idx] as f64).abs();
                    let g = 1.0 - diff / max_diff;
                    let attack = strengths[j] * g;
                    if attack > best_strength {
                        best_strength = attack;
                        best_label = labels[j];
                    }
                }
                (best_label, best_strength)
            })
            .collect();

        // Check convergence and apply updates.
        let mut changed = false;
        for (idx, (new_l, new_c)) in updates.into_iter().enumerate() {
            if !is_seed[idx] && new_l != labels[idx] {
                changed = true;
                labels[idx] = new_l;
                strengths[idx] = new_c;
            }
        }
        if !changed {
            break;
        }
    }

    labels.into_iter().map(|l| l as f32).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        image::Image,
        spatial::{Direction, Point, Spacing},
    };
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image(values: Vec<f32>, shape: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let td = TensorData::new(values, Shape::new(shape));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn get_labels(image: &Image<TestBackend, 3>) -> Vec<u32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }

    // ── Positive tests ────────────────────────────────────────────────────────

    /// Seeds are never overwritten. If image is uniform and seed has two labels,
    /// the seed positions must keep their original labels.
    #[test]
    fn test_seeds_are_immutable() {
        // 1x1x4: uniform intensities, seed=[1,0,0,2]
        let image = make_image(vec![100.0_f32; 4], [1, 1, 4]);
        let seeds = make_image(vec![1.0_f32, 0.0, 0.0, 2.0], [1, 1, 4]);
        let result = growcut(&image, &seeds, 100);
        let labels = get_labels(&result);
        assert_eq!(labels[0], 1, "seed at index 0 must remain label 1");
        assert_eq!(labels[3], 2, "seed at index 3 must remain label 2");
    }

    /// On a uniform intensity image with seeds at both ends, GrowCut partitions
    /// the volume by proximity (Voronoi-like).
    #[test]
    fn test_uniform_intensity_voronoi_split() {
        // 1x1x8: uniform intensities, seed1 at 0, seed2 at 7
        let image = make_image(vec![100.0_f32; 8], [1, 1, 8]);
        let mut seed_vals = vec![0.0_f32; 8];
        seed_vals[0] = 1.0;
        seed_vals[7] = 2.0;
        let seeds = make_image(seed_vals, [1, 1, 8]);
        let result = growcut(&image, &seeds, 200);
        let labels = get_labels(&result);
        // All voxels must be labeled (no zeros after convergence)
        for (i, &l) in labels.iter().enumerate() {
            assert!(
                l > 0,
                "voxel {i} must be labeled after convergence, got {l}"
            );
        }
        // Symmetry: label 1 should be in indices 0..=3 and label 2 in 4..=7
        // (or 0..=3 and 4..=7 with the boundary at 3.5)
        assert_eq!(labels[0], 1, "index 0 must be label 1");
        assert_eq!(labels[7], 2, "index 7 must be label 2");
        assert_eq!(
            labels[1], 1,
            "index 1 should be label 1 (closest to seed 1)"
        );
        assert_eq!(
            labels[6], 2,
            "index 6 should be label 2 (closest to seed 2)"
        );
    }

    /// High-contrast barrier separates the two seeds. The high-intensity wall
    /// between seeds greatly reduces g(j,i) so seeds cannot cross it.
    #[test]
    fn test_high_contrast_barrier_limits_propagation() {
        // 1x1x7: [low, low, low, HIGH, low, low, low]
        // seed1 at 0, seed2 at 6. Barrier at index 3.
        // With barrier intensity >> background, g across barrier ≈ 0.
        let mut img = vec![10.0_f32; 7];
        img[3] = 1000.0; // barrier
        let image = make_image(img, [1, 1, 7]);
        let mut seed_vals = vec![0.0_f32; 7];
        seed_vals[0] = 1.0;
        seed_vals[6] = 2.0;
        let seeds = make_image(seed_vals, [1, 1, 7]);
        let result = growcut(&image, &seeds, 200);
        let labels = get_labels(&result);
        // Seed positions must be preserved
        assert_eq!(labels[0], 1);
        assert_eq!(labels[6], 2);
        // Low-intensity region left of barrier belongs to label 1
        assert_eq!(labels[1], 1, "left of barrier must be label 1");
        assert_eq!(labels[2], 1, "left of barrier must be label 1");
        // Low-intensity region right of barrier belongs to label 2
        assert_eq!(labels[4], 2, "right of barrier must be label 2");
        assert_eq!(labels[5], 2, "right of barrier must be label 2");
    }

    /// Single label seed fills the entire volume when all voxels are uniform.
    #[test]
    fn test_single_seed_fills_entire_volume() {
        let n = 27;
        let image = make_image(vec![50.0_f32; n], [3, 3, 3]);
        let mut seed_vals = vec![0.0_f32; n];
        seed_vals[0] = 1.0; // single seed at corner
        let seeds = make_image(seed_vals, [3, 3, 3]);
        let result = growcut(&image, &seeds, 500);
        let labels = get_labels(&result);
        assert!(
            labels.iter().all(|&l| l == 1),
            "all voxels must be label 1 after fill"
        );
    }

    /// Spatial metadata (origin, spacing, direction) is preserved.
    #[test]
    fn test_spatial_metadata_preserved() {
        let image = make_image(vec![1.0_f32; 8], [2, 2, 2]);
        let seeds = make_image(vec![1.0_f32; 8], [2, 2, 2]);
        let result = growcut(&image, &seeds, 10);
        assert_eq!(result.origin(), image.origin());
        assert_eq!(result.spacing(), image.spacing());
        assert_eq!(result.direction(), image.direction());
    }

    // ── Negative / boundary tests ─────────────────────────────────────────────

    /// Mismatched shapes between image and seeds must panic.
    #[test]
    #[should_panic(expected = "identical shapes")]
    fn test_shape_mismatch_panics() {
        let image = make_image(vec![1.0_f32; 8], [2, 2, 2]);
        let seeds = make_image(vec![1.0_f32; 27], [3, 3, 3]);
        let _ = growcut(&image, &seeds, 10);
    }

    /// max_iter=0 leaves the label map unchanged (just seed initialization, no iterations).
    #[test]
    fn test_zero_iterations_returns_seed_map() {
        let image = make_image(vec![1.0_f32; 8], [2, 2, 2]);
        let seed_vals = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let seeds = make_image(seed_vals.clone(), [2, 2, 2]);
        let result = growcut(&image, &seeds, 0);
        let labels = get_labels(&result);
        // Without any iteration only the seed at index 0 is labeled.
        assert_eq!(labels[0], 1, "seed must remain labeled");
        for &l in labels.iter().skip(1) {
            assert_eq!(l, 0, "unlabeled voxels remain 0 with 0 iterations");
        }
    }

    /// GrowCutFilter struct and growcut function produce identical results.
    #[test]
    fn test_filter_struct_matches_function() {
        let image = make_image(vec![1.0_f32; 8], [2, 2, 2]);
        let seed_vals = vec![1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let seeds = make_image(seed_vals, [2, 2, 2]);
        let via_fn = growcut(&image, &seeds, 50);
        let via_struct = GrowCutFilter::new(50).apply(&image, &seeds);
        let fn_labels = get_labels(&via_fn);
        let struct_labels = get_labels(&via_struct);
        assert_eq!(fn_labels, struct_labels, "function and struct must match");
    }
}
