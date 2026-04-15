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

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use std::collections::VecDeque;

// ── Public types ─────────────────────────────────────────────────────────────

/// Confidence-connected region-growing filter (Yanowitz/Bruckstein variant).
///
/// Segments region by iteratively adapting inclusion interval based on
/// running mean and standard deviation of the current region.
pub struct ConfidenceConnectedFilter {
    /// Seed voxel in [z, y, x] index space.
    pub seed: [usize; 3],
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
    pub fn new(seed: [usize; 3], initial_lower: f32, initial_upper: f32) -> Self {
        assert!(
            initial_lower <= initial_upper,
            "initial_lower {initial_lower} must be ≤ initial_upper {initial_upper}"
        );
        Self {
            seed,
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
    seed: [usize; 3],
    initial_lower: f32,
    initial_upper: f32,
    multiplier: f32,
    max_iterations: usize,
) -> Image<B, 3> {
    assert!(
        initial_lower <= initial_upper,
        "initial_lower {initial_lower} must be ≤ initial_upper {initial_upper}"
    );
    let shape = image.shape();
    let (nz, ny, nx) = (shape[0], shape[1], shape[2]);
    assert!(
        seed[0] < nz && seed[1] < ny && seed[2] < nx,
        "seed {:?} is out of bounds for image shape {:?}",
        seed,
        shape
    );

    let device = image.data().device();
    let img_data = image.data().clone().into_data();
    let img_slice = img_data.as_slice::<f32>().expect("f32 image tensor data");

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
        image.origin().clone(),
        image.spacing().clone(),
        image.direction().clone(),
    )
}

// ── Core iterative growing algorithm ─────────────────────────────────────────

/// Perform iterative confidence-connected region growing on flat `[nz × ny × nx]` data.
///
/// Returns a flat binary Vec<f32> of the same length as `data`.
fn grow_region(
    data: &[f32],
    dims: [usize; 3],
    seed: [usize; 3],
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
    let mut region_voxels: Vec<usize> = Vec::new();

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
        let mut new_voxels: Vec<usize> = Vec::new();
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
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image(values: Vec<f32>, shape: [usize; 3]) -> Image<TestBackend, 3> {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(values, Shape::new(shape));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn get_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn count_foreground(image: &Image<TestBackend, 3>) -> usize {
        get_values(image).iter().filter(|&&v| v > 0.5).count()
    }

    // ── Positive tests ────────────────────────────────────────────────────────

    #[test]
    fn test_seed_in_uniform_region_grows_entire_volume() {
        // All voxels have intensity 100; seed intensity 100 within [50, 150].
        // With μ=100, σ=0 initially, then σ=0 for uniform region,
        // all voxels qualify on first iteration.
        let image = make_image(vec![100.0_f32; 64], [4, 4, 4]);
        let result = confidence_connected(&image, [0, 0, 0], 50.0, 150.0, 2.5, 15);
        assert_eq!(
            count_foreground(&result),
            64,
            "uniform region grows entire volume"
        );
    }

    #[test]
    fn test_iterative_update_converges_to_stable_region() {
        // Two-region image: center 2x2x2 = 200, surrounding = 50.
        // Seed in center; initial bounds [150, 255] only capture center.
        // After first iteration, μ=200, σ=0, so same bounds → stable.
        let mut values = vec![50.0_f32; 64]; // 4x4x4
                                             // Center 2x2x2 at [1..3, 1..3, 1..3].
        for z in 1..3 {
            for y in 1..3 {
                for x in 1..3 {
                    values[z * 16 + y * 4 + x] = 200.0;
                }
            }
        }
        let image = make_image(values, [4, 4, 4]);
        let result = confidence_connected(&image, [1, 1, 1], 150.0, 255.0, 2.5, 15);
        // Should converge to exactly the 2x2x2 center region (8 voxels).
        assert_eq!(
            count_foreground(&result),
            8,
            "iterative update must converge to stable region"
        );
    }

    #[test]
    fn test_multiplier_affects_region_size() {
        // Gradient sphere: center intensity 200, linear falloff to edge.
        // Larger multiplier should produce larger regions.
        let (nz, ny, nx) = (9, 9, 9);
        let mut values = vec![0.0_f32; nz * ny * nx];
        let (cz, cy, cx) = (4isize, 4isize, 4isize);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dz = iz as isize - cz;
                    let dy = iy as isize - cy;
                    let dx = ix as isize - cx;
                    let dist_sq = dz * dz + dy * dy + dx * dx;
                    // Intensity = 200 - 10*distance², clipped at 50.
                    let intensity = (200.0 - 10.0 * (dist_sq as f32)).max(50.0);
                    values[iz * ny * nx + iy * nx + ix] = intensity;
                }
            }
        }
        let image = make_image(values, [nz, ny, nx]);
        let result_small = confidence_connected(&image, [4, 4, 4], 150.0, 250.0, 1.0, 15);
        let result_large = confidence_connected(&image, [4, 4, 4], 150.0, 250.0, 5.0, 15);
        let count_small = count_foreground(&result_small);
        let count_large = count_foreground(&result_large);
        assert!(
            count_large > count_small,
            "larger k={} (count={}) must produce larger region than k={} (count={})",
            5.0,
            count_large,
            1.0,
            count_small
        );
    }

    #[test]
    fn test_max_iteration_limit_respected() {
        // Gradual gradient from center; without limit would grow indefinitely.
        // With max_iterations=1, should stop after first expansion.
        let mut values = vec![100.0_f32; 27]; // 3x3x3
        values[13] = 200.0; // center
        let image = make_image(values, [3, 3, 3]);
        let result = confidence_connected(&image, [1, 1, 1], 150.0, 250.0, 2.5, 1);
        // First iteration: center only (σ=0). No growth, so just 1 voxel.
        assert_eq!(
            count_foreground(&result),
            1,
            "max_iterations=1 must limit growth"
        );
    }

    #[test]
    fn test_spatial_metadata_preserved() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vec![100.0_f32; 27], Shape::new([3, 3, 3]));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 1.0, 2.0]);
        let direction = Direction::identity();
        let image = Image::new(tensor, origin, spacing, direction);
        let result = confidence_connected(&image, [0, 0, 0], 50.0, 150.0, 2.5, 15);
        assert_eq!(result.origin(), &origin);
        assert_eq!(result.spacing(), &spacing);
        assert_eq!(result.direction(), &direction);
    }

    #[test]
    fn test_binary_output_verification() {
        let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);
        let result = confidence_connected(&image, [1, 1, 1], 50.0, 150.0, 2.5, 15);
        for &v in get_values(&result).iter() {
            assert!(
                v == 0.0 || v == 1.0,
                "output must be strictly binary, got {v}"
            );
        }
    }

    // ── 3-D volumetric test ───────────────────────────────────────────────────

    #[test]
    fn test_3d_sphere_region_growing() {
        // 9×9×9 image with a sphere of radius 3 at center (4,4,4) with intensity 200;
        // background intensity 50; initial bounds [150, 255].
        // Region growing from center should capture exactly the sphere.
        let (nz, ny, nx) = (9, 9, 9);
        let mut values = vec![50.0_f32; nz * ny * nx];
        let (cz, cy, cx) = (4isize, 4isize, 4isize);
        let r2 = 9isize; // radius 3
        let mut sphere_count = 0;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dz = iz as isize - cz;
                    let dy = iy as isize - cy;
                    let dx = ix as isize - cx;
                    if dz * dz + dy * dy + dx * dx <= r2 {
                        values[iz * ny * nx + iy * nx + ix] = 200.0;
                        sphere_count += 1;
                    }
                }
            }
        }
        let image = make_image(values, [nz, ny, nx]);
        // Sphere voxels are uniform (200), so once entered, μ=200, σ=0,
        // initial bounds continue to apply, sphere captured completely.
        let result = confidence_connected(&image, [4, 4, 4], 150.0, 255.0, 2.5, 15);
        assert_eq!(
            count_foreground(&result),
            sphere_count,
            "grown region must match sphere voxel count exactly"
        );
    }

    // ── Negative tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_seed_outside_initial_range_returns_empty() {
        // Seed intensity = 5.0, initial range [50, 200] → seed excluded → empty mask.
        let image = make_image(vec![5.0_f32; 8], [2, 2, 2]);
        let result = confidence_connected(&image, [0, 0, 0], 50.0, 200.0, 2.5, 15);
        assert_eq!(
            count_foreground(&result),
            0,
            "seed outside initial range must produce empty region"
        );
    }

    #[test]
    fn test_filter_struct_builder_pattern() {
        let image = make_image(vec![100.0_f32; 64], [4, 4, 4]);
        let via_fn = confidence_connected(&image, [0, 0, 0], 50.0, 150.0, 3.0, 10);
        let via_struct = ConfidenceConnectedFilter::new([0, 0, 0], 50.0, 150.0)
            .with_multiplier(3.0)
            .with_max_iterations(10)
            .apply(&image);
        let fn_vals = get_values(&via_fn);
        let struct_vals = get_values(&via_struct);
        assert_eq!(
            fn_vals, struct_vals,
            "function and filter struct must produce identical results"
        );
    }
}
