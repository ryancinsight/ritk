//! GrowCut interactive segmentation via cellular automaton.
//!
//! # Algorithm
//! Vezhnevets & Konouchine (2005) "GrowCut — Interactive Multi-Label
//! N-D Image Segmentation by Cellular Automata", GRAPHITE 2005.
//!
//! Each voxel i maintains a label L\[i\] ∈ {0,1,…,K} and a strength C\[i\] ∈ \[0,1\].
//! Seeds are initialized with L\[i\] = seed, C\[i\] = 1.0. Unlabeled voxels start
//! with L\[i\] = 0, C\[i\] = 0.0. Seed voxels are immutable.
//!
//! At each iteration a labeled neighbor j "attacks" i if:
//! C\[j\] · g(j,i) > C\[i\] where g(j,i) = 1 − |I\[j\]−I\[i\]| / max_diff
//! On a successful attack L\[i\] ← L\[j\], C\[i\] ← C\[j\]·g(j,i).
//!
//! Terminates when no voxel changes label or `max_iter` is reached.
//!
//! # Complexity
//! O(max_iter · N · 6) = O(max_iter · N). Each iteration is fully data-parallel.
//!
//! # ITK Parity
//! `itk::FastMarchingSegmentationModule` / GrowCut filter (3D Slicer extension).

use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_tensor_ops::extract_vec_infallible;
use ritk_image::Image;

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
/// `image_slice`: flat f32 voxel intensities \[nz\*ny\*nx\]
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
        let updates: Vec<(u32, f64)> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |idx| {
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
            });

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
#[path = "tests_growcut.rs"]
mod tests;
