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
//! ## Algorithm — Iterative Adaptive Flood Fill
//!
//! 1. Initialize R₀ = {s}, μ₀ = I(s), σ₀ = 0
//! 2. For iteration i until convergence:
//!    a. Compute interval [μᵢ - k·σᵢ, μᵢ + k·σᵢ]
//!    b. Use initial bounds [lower, upper] when i = 0 and σ₀ = 0
//!    c. Find all 6-connected neighbors outside Rᵢ with I ∈ interval
//!    d. Add qualifying voxels to Rᵢ₊₁
//!    e. Recompute μᵢ₊₁, σᵢ₊₁ from all voxels in Rᵢ₊₁
//! 3. Termination when |Rᵢ₊₁| = |Rᵢ| (no growth) or max iterations reached
//!
//! # Complexity
//! - Time: O(|R| · iter) — each voxel visited once per iteration, statistics
//!   computed from accumulating sums for O(1) amortized update.
//! - Space: O(|R|) for visited set, queue, and region voxel tracking.
//!
//! # References
//! Yanowitz, S.D., & Bruckstein, A.M. (1989). "A New Method for Image
//! Segmentation." *Computer Vision, Graphics, and Image Processing*, 46(1), 82-95.

use ritk_tensor_ops::extract_vec_infallible;
use ritk_image::Image;
use ritk_core::spatial::VoxelIndex;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
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
fn grow_region(
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
    let flat = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;

    // Check seed intensity against initial bounds (otherwise empty region).
    let seed_val = data[flat(seed[0], seed[1], seed[2])];
    if seed_val < initial_lower || seed_val > initial_upper {
        return vec![0.0_f32; n];
    }

    // Output binary mask.
    let mut output = vec![0.0_f32; n];
    // Visited tracking (includes both queued and processed).
    let mut visited = vec![false; n];
    // BFS frontier.
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(1024);
    // Track region voxels for statistics recomputation: stores flat indices.
    // Heuristic: region typically covers a small fraction of the volume.
    let mut region_voxels: Vec<usize> = Vec::with_capacity(n / 16);
    // Buffer for collecting new voxels each iteration (hoisted to avoid per-iteration heap allocation).
    let mut new_voxels: Vec<usize> = Vec::with_capacity(queue.capacity().min(n / 16));

    let seed_flat = flat(seed[0], seed[1], seed[2]);
    visited[seed_flat] = true;
    output[seed_flat] = 1.0;
    queue.push_back(seed_flat);
    region_voxels.push(seed_flat);

    // Running statistics: sum and sum of squares for O(1) updates.
    let mut sum: f64 = seed_val as f64;
    let mut sum_sq: f64 = (seed_val as f64) * (seed_val as f64);

    // 6-connectivity face offsets.
    let face_offsets: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    // Iterative growing.
    for _iteration in 0..max_iterations {
        // Compute current statistics.
        let count = region_voxels.len() as f64;
        let mean = sum / count;
        let variance = (sum_sq / count) - (mean * mean);
        let std_dev = variance.max(0.0).sqrt() as f32;

        // Determine intensity interval.
        let (lower, upper) = if std_dev == 0.0 {
            (initial_lower, initial_upper)
        } else {
            let delta = multiplier * std_dev;
            ((mean as f32) - delta, (mean as f32) + delta)
        };

        // Collect new voxels to add this iteration.
        new_voxels.clear();
        // Process BFS queue with current interval criteria.
        let mut _frontier_processed = false;

        while let Some(curr_flat) = queue.pop_front() {
            _frontier_processed = true;

            // Decode flat index to (z, y, x).
            let iz = curr_flat / (ny * nx);
            let rem = curr_flat % (ny * nx);
            let iy = rem / nx;
            let ix = rem % nx;

            // Check 6-connected neighbors.
            for &(dz, dy, dx) in &face_offsets {
                let nz_i = iz as isize + dz;
                let ny_i = iy as isize + dy;
                let nx_i = ix as isize + dx;

                // Bounds check.
                if nz_i < 0
                    || nz_i >= nz as isize
                    || ny_i < 0
                    || ny_i >= ny as isize
                    || nx_i < 0
                    || nx_i >= nx as isize
                {
                    continue;
                }

                let n_flat = flat(nz_i as usize, ny_i as usize, nx_i as usize);
                if visited[n_flat] {
                    continue;
                }

                let intensity = data[n_flat];
                if intensity < lower || intensity > upper {
                    continue;
                }

                // Voxel qualifies: mark visited, add to output, track as new region member.
                visited[n_flat] = true;
                output[n_flat] = 1.0;
                new_voxels.push(n_flat);
            }
        }

        // If no new voxels added, convergence reached.
        if new_voxels.is_empty() {
            break;
        }

        // Update region tracking and statistics for next iteration.
        for &voxel_flat in &new_voxels {
            region_voxels.push(voxel_flat);
            let val = data[voxel_flat] as f64;
            sum += val;
            sum_sq += val * val;
            // Add to BFS queue for next iteration's expansion.
            queue.push_back(voxel_flat);
        }

        // If queue is empty after processing, no further growth possible.
        if queue.is_empty() {
            break;
        }
    }

    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_confidence_connected.rs"]
mod tests_confidence_connected;
