//! K-Means clustering segmentation (Lloyd's algorithm with k-means++ initialization).
//!
//! # Mathematical Specification
//!
//! K-Means partitions the intensity space of an image into K clusters by
//! minimising the within-cluster sum of squared distances (WCSS):
//!
//!   J = Σ_{k=1}^{K} Σ_{x ∈ C_k} (x − μ_k)²
//!
//! where μ_k is the centroid (mean intensity) of cluster C_k.
//!
//! ## Initialization (k-means++)
//!
//! Arthur & Vassilvitskii (2007) initialization:
//! 1. Choose the first centroid c₁ uniformly at random from the data points.
//! 2. For each subsequent centroid c_j (j = 2..K):
//!    - For each data point x, compute D(x) = min_{i<j} |x − c_i|.
//!    - Choose c_j with probability proportional to D(x)².
//! 3. This yields O(log K)-competitive initial centroids in expectation.
//!
//! ## Lloyd's Iteration
//!
//! Repeat until convergence or `max_iterations`:
//! 1. **Assignment**: For each voxel x, assign label(x) = argmin_k |x − μ_k|.
//! 2. **Update**: μ_k = (1/|C_k|) Σ_{x ∈ C_k} x.
//! 3. **Convergence**: max_k |μ_k^{new} − μ_k^{old}| < tolerance.
//!
//! ## Output
//!
//! A label image where each voxel contains its cluster index (0..K−1) as `f32`.
//!
//! # Complexity
//!
//! Initialization:  O(n·K) distance computations.
//! Each iteration:  O(n·K) assignments + O(n) centroid updates.
//! Total:           O(n·K·(I+1)) where I = number of iterations.
//!
//! # References
//!
//! - Lloyd, S.P. (1982). "Least squares quantization in PCM." *IEEE Trans.
//!   Information Theory*, 28(2):129–137.
//! - Arthur, D. & Vassilvitskii, S. (2007). "k-means++: The Advantages of
//!   Careful Seeding." *Proc. 18th ACM-SIAM Symposium on Discrete Algorithms*.

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

// ── Deterministic PRNG (xorshift64) ────────────────────────────────────────────

/// Minimal xorshift64 PRNG to avoid external dependencies.
///
/// Period: 2^64 − 1. Not cryptographically secure; sufficient for
/// reproducible k-means++ initialization.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG with the given seed.
    ///
    /// # Panics
    /// Panics if `seed == 0` (xorshift64 has a fixed point at 0).
    fn new(seed: u64) -> Self {
        assert!(seed != 0, "xorshift64 seed must be non-zero");
        Self { state: seed }
    }

    /// Return the next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s
    }

    /// Return a pseudo-random `f64` in [0, 1).
    fn sample_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Return a pseudo-random index in [0, n).
    fn next_index(&mut self, n: usize) -> usize {
        (self.sample_unit() * n as f64) as usize % n
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// K-Means clustering segmentation.
///
/// Partitions an image's intensity space into `k` clusters using Lloyd's
/// algorithm with k-means++ initialization. The output is a label image
/// where each voxel contains its cluster index (0..K−1) as `f32`.
#[derive(Debug, Clone)]
pub struct KMeansSegmentation {
    /// Number of clusters. Must be ≥ 1.
    pub k: usize,
    /// Maximum number of Lloyd iterations. Default 100.
    pub max_iterations: usize,
    /// Convergence tolerance on centroid displacement. Default 1e-6.
    pub tolerance: f64,
    /// Deterministic seed for k-means++ initialization. Default 42.
    pub seed: u64,
}

impl KMeansSegmentation {
    /// Create a `KMeansSegmentation` with default parameters.
    ///
    /// Defaults: k=2, max_iterations=100, tolerance=1e-6, seed=42.
    pub fn new(k: usize) -> Self {
        assert!(k >= 1, "k must be ≥ 1");
        Self {
            k,
            max_iterations: 100,
            tolerance: 1e-6,
            seed: 42,
        }
    }

    /// Apply K-Means clustering to `image`, returning a label image.
    ///
    /// Each voxel in the output contains its assigned cluster index (0..K−1)
    /// as `f32`. Spatial metadata (origin, spacing, direction) is preserved.
    ///
    /// For a constant image (zero range), all voxels are assigned label 0.0.
    /// For k=1, all voxels are assigned label 0.0.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let (vals, shape) = extract_vec_infallible(image);
        let device = image.data().device();
        let slice: &[f32] = &vals;

        let labels = kmeans_impl(
            slice,
            self.k,
            self.max_iterations,
            self.tolerance,
            self.seed,
        );

        let tensor = Tensor::<B, D>::from_data(TensorData::new(labels, Shape::new(shape)), &device);

        Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
    }
}

impl Default for KMeansSegmentation {
    fn default() -> Self {
        Self::new(2)
    }
}

/// Convenience function: apply K-Means with k clusters and default parameters.
pub fn kmeans_segment<B: Backend, const D: usize>(image: &Image<B, D>, k: usize) -> Image<B, D> {
    KMeansSegmentation::new(k).apply(image)
}

// ── Core implementation ────────────────────────────────────────────────────────

/// Core K-Means implementation operating on a flat f32 slice.
///
/// Returns a `Vec<f32>` of cluster labels (0..K−1).
fn kmeans_impl(
    data: &[f32],
    k: usize,
    max_iterations: usize,
    tolerance: f64,
    seed: u64,
) -> Vec<f32> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    // Degenerate: k=1 or k >= n, assign trivially.
    if k <= 1 {
        return vec![0.0_f32; n];
    }

    // Effective k is min(k, number of distinct values) but for correctness
    // we proceed with k and handle empty clusters in the update step.
    let effective_k = k.min(n);

    // ── k-means++ initialization ───────────────────────────────────────────────
    let mut rng = Xorshift64::new(seed);
    let mut centroids = Vec::with_capacity(effective_k);

    // First centroid: random data point.
    let first_idx = rng.next_index(n);
    centroids.push(data[first_idx] as f64);

    // Distance-squared buffer for weighted sampling.
    let mut dist_sq = vec![f64::MAX; n];

    for _ in 1..effective_k {
        // Update minimum distances to the most recently added centroid.
        let last_centroid = *centroids.last().expect("centroids must not be empty");
        for i in 0..n {
            let d = data[i] as f64 - last_centroid;
            let d2 = d * d;
            if d2 < dist_sq[i] {
                dist_sq[i] = d2;
            }
        }

        // Compute cumulative distribution for weighted sampling.
        let total: f64 = dist_sq.iter().sum();
        if total < 1e-30 {
            // All remaining points are at existing centroids; duplicate last.
            centroids.push(last_centroid);
            continue;
        }

        let target = rng.sample_unit() * total;
        let mut cumulative = 0.0_f64;
        let mut chosen = n - 1;
        for (i, &d) in dist_sq.iter().enumerate() {
            cumulative += d;
            if cumulative >= target {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen] as f64);
    }

    // ── Lloyd's iterations ─────────────────────────────────────────────────────
    let mut labels = vec![0u32; n];

    for _iter in 0..max_iterations {
        // ── Assignment step ─────────────────────────────────────────────────
        for i in 0..n {
            let val = data[i] as f64;
            let mut best_k = 0u32;
            let mut best_dist = f64::MAX;
            for (ci, &centroid) in centroids.iter().enumerate() {
                let d = (val - centroid).abs();
                if d < best_dist {
                    best_dist = d;
                    best_k = ci as u32;
                }
            }
            labels[i] = best_k;
        }

        // ── Update step ────────────────────────────────────────────────────
        let mut sums = vec![0.0_f64; effective_k];
        let mut counts = vec![0u64; effective_k];

        for i in 0..n {
            let c = labels[i] as usize;
            sums[c] += data[i] as f64;
            counts[c] += 1;
        }

        let mut max_shift = 0.0_f64;
        for ci in 0..effective_k {
            if counts[ci] > 0 {
                let new_centroid = sums[ci] / counts[ci] as f64;
                let shift = (new_centroid - centroids[ci]).abs();
                if shift > max_shift {
                    max_shift = shift;
                }
                centroids[ci] = new_centroid;
            }
            // Empty clusters retain their previous centroid position.
        }

        // ── Convergence check ──────────────────────────────────────────────
        if max_shift < tolerance {
            break;
        }
    }

    // ── Final assignment (in case we broke before the last assignment) ──────
    for i in 0..n {
        let val = data[i] as f64;
        let mut best_k = 0u32;
        let mut best_dist = f64::MAX;
        for (ci, &centroid) in centroids.iter().enumerate() {
            let d = (val - centroid).abs();
            if d < best_dist {
                best_dist = d;
                best_k = ci as u32;
            }
        }
        labels[i] = best_k;
    }

    labels.iter().map(|&l| l as f32).collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_kmeans.rs"]
mod tests_kmeans;
