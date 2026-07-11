//! Confidence-connected region growing for 3-D images.
//!
//! # Mathematical Specification
//!
//! Given an intensity image I and a seed voxel s ∈ ℤ³, the confidence-connected
//! region growing algorithm iteratively expands based on adaptive intensity
//! statistics computed from the current region.
//!
//! ## Theorem (Yanowitz/Bruckstein Adaptive Region Growing)
//!
//! For iteration i with region Rᵢ:
//! - μᵢ = (1/|Rᵢ|) Σ_{p∈Rᵢ} I(p)        (sample mean)
//! - σᵢ = √[(1/|Rᵢ|) Σ_{p∈Rᵢ} (I(p) - μᵢ)²]  (sample standard deviation)
//!
//! The inclusion predicate for voxel q ∈ N₆(p) where p ∈ Rᵢ:
//!
//! P(q ∈ Rᵢ₊₁) ≡ μᵢ - k·σᵢ ≤ I(q) ≤ μᵢ + k·σᵢ
//!
//! where k is the confidence interval multiplier.
//!
//! ## Algorithm — Iterative Re-Flood (ITK `ConfidenceConnectedImageFilter`)
//!
//! 1. Flood-fill the **entire** 6-connected region reachable from the seed whose
//!    intensity lies in the initial interval `[lower₀, upper₀]`.
//! 2. For each of `max_iterations` passes:
//!    a. Recompute μ and σ over **all** voxels currently in the region, using the
//!       sample (N − 1) variance estimator.
//!    b. Set the interval to `[μ − k·σ, μ + k·σ]`, widened if necessary to contain
//!       the seed intensity.
//!    c. Discard the region and **re-flood from scratch** from the seed with the
//!       new interval (a complete connected-threshold flood, not one BFS ring).
//!    d. Stop early once the interval is a fixed point (the next flood would be
//!       identical).
//!
//! The earlier implementation advanced exactly **one BFS wavefront per iteration**
//! and recomputed statistics after each ring, so `max_iterations` capped growth at
//! that many rings (≈ a tiny diamond around the seed) instead of the full region —
//! the source of a ~180× under-segmentation versus ITK. Each iteration is now a
//! complete flood, matching ITK's "clear output, re-flood from seeds" loop.
//!
//! # Complexity
//! - Time: O(|R| · iter) — one full flood per iteration, each visiting the region
//!   plus its boundary once.
//! - Space: O(n) for the visited buffer and BFS queue.
//!
//! # References
//! Yanowitz, S.D., & Bruckstein, A.M. (1989). "A New Method for Image
//! Segmentation." *Computer Vision, Graphics, and Image Processing*, 46(1), 82-95.
//! ITK `itkConfidenceConnectedImageFilter.hxx` (N−1 variance, per-iteration
//! re-flood, seed-clamped inclusive bounds).

use ritk_core::spatial::VoxelIndex;
use ritk_image::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
use std::collections::VecDeque;

// ── Public types ─────────────────────────────────────────────────────────────

/// Confidence-connected region-growing filter (Yanowitz/Bruckstein variant).
///
/// Segments region by iteratively adapting inclusion interval based on
/// running mean and standard deviation of the current region.
pub struct ConfidenceConnectedFilter {
    /// Seed voxel in [z, y, x] index space.
    pub seed: VoxelIndex,
    /// Initial lower bound for first iteration when σ = 0.
    pub initial_lower: f32,
    /// Initial upper bound for first iteration when σ = 0.
    pub initial_upper: f32,
    /// Multiplier k for k·σ interval expansion (typically 2.5).
    pub multiplier: f32,
    /// Maximum number of iterations before forced termination.
    pub max_iterations: usize,
}

impl ConfidenceConnectedFilter {
    /// Create a `ConfidenceConnectedFilter` with required parameters.
    ///
    /// # Arguments
    /// * `seed` — starting voxel in [z, y, x] index space.
    /// * `initial_lower` — inclusive lower bound when σ = 0 (first iteration).
    /// * `initial_upper` — inclusive upper bound when σ = 0 (first iteration).
    ///
    /// # Panics
    /// Panics if `initial_lower > initial_upper`.
    pub fn new(seed: impl Into<VoxelIndex>, initial_lower: f32, initial_upper: f32) -> Self {
        assert!(
            initial_lower <= initial_upper,
            "initial_lower {initial_lower} must be ≤ initial_upper {initial_upper}"
        );
        Self {
            seed: seed.into(),
            initial_lower,
            initial_upper,
            multiplier: 2.5,
            max_iterations: 15,
        }
    }

    /// Set the confidence interval multiplier (k for k·σ interval).
    ///
    /// Default: 2.5. Larger values produce more permissive region growing.
    pub fn with_multiplier(mut self, multiplier: f32) -> Self {
        self.multiplier = multiplier;
        self
    }

    /// Set the maximum number of iterations before termination.
    ///
    /// Default: 15. Prevents infinite loops on ambiguous boundaries.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Apply confidence-connected region growing to `image`.
    ///
    /// Returns a binary mask (values in {0.0, 1.0}) with the same shape and
    /// spatial metadata as `image`. Region adapts iteratively based on
    /// statistics of voxels already included.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        confidence_connected(
            image,
            self.seed,
            self.initial_lower,
            self.initial_upper,
            self.multiplier,
            self.max_iterations,
        )
    }
}

// ── Public function ───────────────────────────────────────────────────────────

/// Confidence-connected region growing starting from `seed`.
///
/// Returns a binary mask whose shape and spatial metadata match `image`.
/// Voxels included in the grown region are set to 1.0; all others to 0.0.
///
/// The algorithm iteratively updates intensity statistics (mean, std deviation)
/// from the current region and uses [μ - k·σ, μ + k·σ] as the inclusion interval.
/// For the first iteration, σ = 0, so the initial bounds are used instead.
///
/// # Arguments
/// * `image` — input intensity image (3-D).
/// * `seed` — starting voxel [z, y, x].
/// * `initial_lower` — lower bound for first iteration.
/// * `initial_upper` — upper bound for first iteration.
/// * `multiplier` — k value for confidence interval scaling.
/// * `max_iterations` — hard limit on iteration count.
///
/// # Panics
/// Panics if `initial_lower > initial_upper` or if `seed` is out of bounds.
pub fn confidence_connected<B: Backend>(
    image: &Image<B, 3>,
    seed: impl Into<VoxelIndex>,
    initial_lower: f32,
    initial_upper: f32,
    multiplier: f32,
    max_iterations: usize,
) -> Image<B, 3> {
    assert!(
        initial_lower <= initial_upper,
        "initial_lower {initial_lower} must be ≤ initial_upper {initial_upper}"
    );
    let seed = seed.into();
    let shape = image.shape();
    let (nz, ny, nx) = (shape[0], shape[1], shape[2]);
    assert!(
        seed[0] < nz && seed[1] < ny && seed[2] < nx,
        "seed {:?} is out of bounds for image shape {:?}",
        seed.as_array(),
        shape
    );

    let device = image.data().device();
    let (img_slice_vec, _) = extract_vec_infallible(image);
    let img_slice: &[f32] = &img_slice_vec;

    let result = grow_region(
        img_slice,
        shape,
        seed,
        initial_lower,
        initial_upper,
        multiplier,
        max_iterations,
    );
    let td = TensorData::new(result, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
    )
}

// ── Core iterative growing algorithm ─────────────────────────────────────────

/// Perform iterative confidence-connected region growing on flat `[nz × ny × nx]` data.
///
/// Returns a flat binary `Vec<f32>` of the same length as `data`.
pub(crate) fn grow_region(
    data: &[f32],
    dims: [usize; 3],
    seed: VoxelIndex,
    initial_lower: f32,
    initial_upper: f32,
    multiplier: f32,
    max_iterations: usize,
) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let n = nz * ny * nx;
    let seed_flat = seed[0] * ny * nx + seed[1] * nx + seed[2];
    let seed_val = data[seed_flat];

    let mut output = vec![0.0_f32; n];

    // Seed must lie in the initial interval; otherwise the region is empty.
    if seed_val < initial_lower || seed_val > initial_upper {
        return output;
    }
    // The seed is always part of the region (covers the max_iterations == 0 case).
    output[seed_flat] = 1.0;

    // Reusable flood buffers.
    let mut visited = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(1024);
    let mut region: Vec<usize> = Vec::with_capacity(n / 16);

    let mut lower = initial_lower;
    let mut upper = initial_upper;

    for _ in 0..max_iterations {
        // Full connected-threshold flood from the seed with the current interval.
        let (sum, sum_sq) = flood_region(
            data,
            dims,
            seed_flat,
            lower,
            upper,
            &mut visited,
            &mut queue,
            &mut region,
        );

        // Recompute the interval from the whole region using the sample (N-1)
        // variance estimator, matching ITK. A 1-voxel region has σ = 0.
        let count = region.len();
        let (mut new_lower, mut new_upper) = if count <= 1 {
            (seed_val, seed_val)
        } else {
            let n_f = count as f64;
            let mean = sum / n_f;
            let variance = ((sum_sq - mean * mean * n_f) / (n_f - 1.0)).max(0.0);
            let delta = multiplier as f64 * variance.sqrt();
            ((mean - delta) as f32, (mean + delta) as f32)
        };
        // Bounds are widened to always contain the seed intensity (ITK clamps the
        // threshold to include every seed value).
        new_lower = new_lower.min(seed_val);
        new_upper = new_upper.max(seed_val);

        // Fixed point: the next flood would reproduce the same interval and region.
        if new_lower == lower && new_upper == upper {
            break;
        }
        lower = new_lower;
        upper = new_upper;
    }

    // Mark the final region (the last completed flood) into the output mask.
    for &idx in &region {
        output[idx] = 1.0;
    }
    output
}

/// Flood the full 6-connected region reachable from `seed_flat` whose intensity
/// lies in `[lower, upper]` (inclusive). The `visited`/`queue`/`region` buffers
/// are cleared and reused across calls. Returns the region's intensity `sum` and
/// `sum_sq` for statistics.
#[allow(clippy::too_many_arguments)]
fn flood_region(
    data: &[f32],
    dims: [usize; 3],
    seed_flat: usize,
    lower: f32,
    upper: f32,
    visited: &mut [bool],
    queue: &mut VecDeque<usize>,
    region: &mut Vec<usize>,
) -> (f64, f64) {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    visited.iter_mut().for_each(|v| *v = false);
    queue.clear();
    region.clear();

    let seed_val = data[seed_flat] as f64;
    visited[seed_flat] = true;
    queue.push_back(seed_flat);
    region.push(seed_flat);
    let mut sum = seed_val;
    let mut sum_sq = seed_val * seed_val;

    const FACE_OFFSETS: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    while let Some(curr) = queue.pop_front() {
        let iz = curr / (ny * nx);
        let rem = curr % (ny * nx);
        let iy = rem / nx;
        let ix = rem % nx;

        for &(dz, dy, dx) in &FACE_OFFSETS {
            let zz = iz as isize + dz;
            let yy = iy as isize + dy;
            let xx = ix as isize + dx;
            if zz < 0
                || zz >= nz as isize
                || yy < 0
                || yy >= ny as isize
                || xx < 0
                || xx >= nx as isize
            {
                continue;
            }
            let n_flat = zz as usize * ny * nx + yy as usize * nx + xx as usize;
            if visited[n_flat] {
                continue;
            }
            let intensity = data[n_flat];
            if intensity < lower || intensity > upper {
                continue;
            }
            visited[n_flat] = true;
            queue.push_back(n_flat);
            region.push(n_flat);
            let v = intensity as f64;
            sum += v;
            sum_sq += v * v;
        }
    }

    (sum, sum_sq)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_confidence_connected.rs"]
mod tests_confidence_connected;
