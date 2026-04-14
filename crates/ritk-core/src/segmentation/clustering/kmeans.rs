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

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

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
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Return a pseudo-random index in [0, n).
    fn next_index(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize % n
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
        let device = image.data().device();
        let shape: [usize; D] = image.shape();

        let tensor_data = image.data().clone().into_data();
        let slice = tensor_data
            .as_slice::<f32>()
            .expect("f32 image tensor data");

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
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
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
        let last_centroid = *centroids.last().unwrap();
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

        let target = rng.next_f64() * total;
        let mut cumulative = 0.0_f64;
        let mut chosen = n - 1;
        for i in 0..n {
            cumulative += dist_sq[i];
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
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image_1d(data: Vec<f32>) -> Image<B, 1> {
        let n = data.len();
        let device = Default::default();
        let tensor = Tensor::<B, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

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

    fn get_slice_1d(image: &Image<B, 1>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── Degenerate / constant image ────────────────────────────────────────────

    #[test]
    fn test_constant_image_all_same_label() {
        let data = vec![42.0_f32; 100];
        let image = make_image_1d(data);
        let result = KMeansSegmentation::new(3).apply(&image);
        let labels = get_slice_1d(&result);

        // All voxels must have the same label (cluster).
        let first = labels[0];
        for &l in &labels {
            assert!(
                (l - first).abs() < f32::EPSILON,
                "constant image must yield uniform labels, found {} and {}",
                first,
                l
            );
        }
    }

    // ── Bimodal image produces two clusters ────────────────────────────────────

    #[test]
    fn test_bimodal_two_clusters() {
        // 50 voxels at 10.0 and 50 voxels at 200.0, k=2.
        // All low-intensity voxels must share one label, all high another.
        let mut data = vec![10.0_f32; 50];
        data.extend(vec![200.0_f32; 50]);
        let image = make_image_1d(data);
        let result = KMeansSegmentation::new(2).apply(&image);
        let labels = get_slice_1d(&result);

        // Labels in [0, K-1].
        for &l in &labels {
            assert!(l >= 0.0 && l < 2.0, "label must be in [0, 2), got {}", l);
        }

        // The first 50 must share a label, the last 50 must share a different label.
        let low_label = labels[0];
        let high_label = labels[50];
        assert!(
            (low_label - high_label).abs() > 0.5,
            "two distinct modes must get different labels: {} vs {}",
            low_label,
            high_label
        );

        for i in 0..50 {
            assert!(
                (labels[i] - low_label).abs() < f32::EPSILON,
                "low-mode voxel {} has inconsistent label {} (expected {})",
                i,
                labels[i],
                low_label
            );
        }
        for i in 50..100 {
            assert!(
                (labels[i] - high_label).abs() < f32::EPSILON,
                "high-mode voxel {} has inconsistent label {} (expected {})",
                i,
                labels[i],
                high_label
            );
        }
    }

    // ── Output shape matches input shape ───────────────────────────────────────

    #[test]
    fn test_apply_output_shape_matches_input() {
        let dims = [4, 5, 6];
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i % 3) as f32 * 50.0).collect();
        let image = make_image_3d(data, dims);
        let result = KMeansSegmentation::new(3).apply(&image);
        assert_eq!(result.shape(), dims);
    }

    // ── Labels in valid range ──────────────────────────────────────────────────

    #[test]
    fn test_labels_in_valid_range() {
        let k = 4;
        let mut data = Vec::new();
        for c in 0..k {
            data.extend(vec![c as f32 * 80.0; 25]);
        }
        let image = make_image_1d(data);
        let result = KMeansSegmentation::new(k).apply(&image);
        let labels = get_slice_1d(&result);

        for &l in &labels {
            let li = l as usize;
            assert!(li < k, "label {} must be in [0, {})", l, k);
            assert!(
                (l - li as f32).abs() < f32::EPSILON,
                "label must be an integer, got {}",
                l
            );
        }
    }

    // ── Spatial metadata preserved ─────────────────────────────────────────────

    #[test]
    fn test_apply_preserves_spatial_metadata() {
        let dims = [2, 3, 4];
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 10.0).collect();
        let image = make_image_3d(data, dims);
        let result = KMeansSegmentation::new(2).apply(&image);

        assert_eq!(result.origin(), image.origin());
        assert_eq!(result.spacing(), image.spacing());
        assert_eq!(result.direction(), image.direction());
    }

    // ── Convenience function ───────────────────────────────────────────────────

    #[test]
    fn test_convenience_fn_produces_valid_output() {
        let mut data = vec![10.0_f32; 30];
        data.extend(vec![200.0_f32; 30]);
        let image = make_image_1d(data);
        let result = kmeans_segment(&image, 2);
        let labels = get_slice_1d(&result);
        assert_eq!(labels.len(), 60);
        for &l in &labels {
            assert!(
                l == 0.0 || l == 1.0,
                "must produce binary labels, got {}",
                l
            );
        }
    }

    // ── Determinism ────────────────────────────────────────────────────────────

    #[test]
    fn test_deterministic_with_same_seed() {
        let mut data = vec![10.0_f32; 50];
        data.extend(vec![200.0_f32; 50]);
        let image = make_image_1d(data);

        let r1 = KMeansSegmentation::new(2).apply(&image);
        let r2 = KMeansSegmentation::new(2).apply(&image);

        let l1 = get_slice_1d(&r1);
        let l2 = get_slice_1d(&r2);
        assert_eq!(l1, l2, "same seed must produce identical results");
    }

    // ── k=1 assigns all label 0 ────────────────────────────────────────────────

    #[test]
    fn test_k1_all_zero() {
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let image = make_image_1d(data);
        let result = KMeansSegmentation::new(1).apply(&image);
        let labels = get_slice_1d(&result);
        for &l in &labels {
            assert!(
                (l - 0.0).abs() < f32::EPSILON,
                "k=1 must assign all label 0, got {}",
                l
            );
        }
    }

    // ── Default trait ──────────────────────────────────────────────────────────

    #[test]
    fn test_default_k2() {
        let d = KMeansSegmentation::default();
        assert_eq!(d.k, 2);
        assert_eq!(d.max_iterations, 100);
        assert!((d.tolerance - 1e-6).abs() < 1e-15);
        assert_eq!(d.seed, 42);
    }
}
