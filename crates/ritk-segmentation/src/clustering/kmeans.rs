//! K-Means clustering segmentation (Lloyd's algorithm with k-means++ initialization).
//!
//! # Mathematical Specification
//!
//! K-Means partitions the intensity space of an image into K clusters by
//! minimising the within-cluster sum of squared distances (WCSS):
//!
//!   J = Î£_{k=1}^{K} Î£_{x âˆˆ C_k} (x âˆ’ Î¼_k)Â²
//!
//! where Î¼_k is the centroid (mean intensity) of cluster C_k.
//!
//! ## Initialization (k-means++)
//!
//! Arthur & Vassilvitskii (2007) initialization:
//! 1. Choose the first centroid câ‚ uniformly at random from the data points.
//! 2. For each subsequent centroid c_j (j = 2..K):
//!    - For each data point x, compute D(x) = min_{i<j} |x âˆ’ c_i|.
//!    - Choose c_j with probability proportional to D(x)Â².
//! 3. This yields O(log K)-competitive initial centroids in expectation.
//!
//! ## Lloyd's Iteration
//!
//! Repeat until convergence or `max_iterations`:
//! 1. **Assignment**: For each voxel x, assign label(x) = argmin_k |x âˆ’ Î¼_k|.
//! 2. **Update**: Î¼_k = (1/|C_k|) Î£_{x âˆˆ C_k} x.
//! 3. **Convergence**: max_k |Î¼_k^{new} âˆ’ Î¼_k^{old}| < tolerance.
//!
//! ## Output
//!
//! A label image where each voxel contains its cluster index (0..Kâˆ’1) as `f32`.
//!
//! # Complexity
//!
//! Initialization:  O(nÂ·K) distance computations.
//! Each iteration:  O(nÂ·K) assignments + O(n) centroid updates.
//! Total:           O(nÂ·KÂ·(I+1)) where I = number of iterations.
//!
//! # References
//!
//! - Lloyd, S.P. (1982). "Least squares quantization in PCM." *IEEE Trans.
//!   Information Theory*, 28(2):129â€“137.
//! - Arthur, D. & Vassilvitskii, S. (2007). "k-means++: The Advantages of
//!   Careful Seeding." *Proc. 18th ACM-SIAM Symposium on Discrete Algorithms*.

use ritk_image::tensor::{Backend, Tensor};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

const MAX_EXACT_LABELS: usize = 1 << f32::MANTISSA_DIGITS;

// â”€â”€ Deterministic PRNG (xorshift64) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Minimal xorshift64 PRNG to avoid external dependencies.
///
/// Period: 2^64 âˆ’ 1. Not cryptographically secure; sufficient for
/// reproducible k-means++ initialization.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG with the given seed.
    ///
    /// xorshift64 has a fixed point at 0, so a zero seed is remapped to the
    /// 64-bit golden-ratio constant. This keeps every user-supplied seed valid
    /// and deterministic (in particular `seed == 0`, a natural default) rather
    /// than panicking on input.
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
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

    /// Return a pseudo-random `f32` in [0, 1).
    fn sample_unit(&mut self) -> f32 {
        // The shift bounds the extracted bit field to 24 bits.
        let mantissa = (self.next_u64() >> 40) as u32;
        mantissa as f32 / (1u32 << 24) as f32
    }

    /// Return a pseudo-random index in [0, n).
    fn next_index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// K-Means clustering segmentation.
///
/// Partitions an image's intensity space into `k` clusters using Lloyd's
/// algorithm with k-means++ initialization. The output is a label image
/// where each voxel contains its cluster index (0..Kâˆ’1) as `f32`.
/// Arithmetic and centroid accumulation execute in `f32`. Inputs must be
/// finite and their total range must be representable in `f32`; shifted means
/// and normalized initialization weights prevent overflow within that domain.
#[derive(Debug, Clone)]
pub struct KMeansSegmentation {
    /// Number of clusters. Must be â‰¥ 1.
    k: usize,
    /// Maximum number of Lloyd iterations. Default 100.
    max_iterations: usize,
    /// Convergence tolerance on centroid displacement. Default 1e-6.
    tolerance: f32,
    /// Deterministic seed for k-means++ initialization. Default 42.
    seed: u64,
}

impl KMeansSegmentation {
    /// Create a `KMeansSegmentation` with default parameters.
    ///
    /// Defaults: k=2, max_iterations=100, tolerance=1e-6, seed=42.
    /// # Errors
    ///
    /// Returns an error when `k` is zero.
    pub fn new(k: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(k >= 1, "k must be at least 1, got {k}");
        anyhow::ensure!(
            k <= MAX_EXACT_LABELS,
            "k must not exceed {MAX_EXACT_LABELS} because output labels are f32, got {k}"
        );
        Ok(Self {
            k,
            max_iterations: 100,
            tolerance: 1e-6,
            seed: 42,
        })
    }

    /// Set the maximum number of Lloyd iterations.
    ///
    /// # Errors
    ///
    /// Returns an error when `max_iterations` is zero.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            max_iterations >= 1,
            "k-means maximum iterations must be at least 1, got {max_iterations}"
        );
        self.max_iterations = max_iterations;
        Ok(self)
    }

    /// Set the convergence tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error when `tolerance` is negative or non-finite.
    pub fn with_tolerance(mut self, tolerance: f32) -> anyhow::Result<Self> {
        anyhow::ensure!(
            tolerance.is_finite() && tolerance >= 0.0,
            "k-means tolerance must be finite and nonnegative, got {tolerance}"
        );
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Set the deterministic initialization seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Return the number of clusters.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Return the maximum iteration count.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Return the convergence tolerance.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Return the deterministic initialization seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Apply K-Means clustering to `image`, returning a label image.
    ///
    /// Each voxel in the output contains its assigned cluster index (0..Kâˆ’1)
    /// as `f32`. Spatial metadata (origin, spacing, direction) is preserved.
    ///
    /// For a constant image (zero range), all voxels are assigned label 0.0.
    /// For k=1, all voxels are assigned label 0.0.
    ///
    /// # Errors
    ///
    /// Returns an error when an input sample is non-finite.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<f32, B, D>,
    ) -> anyhow::Result<Image<f32, B, D>> {
        let (vals, shape) = extract_vec_infallible(image);
        validate_samples(&vals)?;
        let device = B::default();
        let slice: &[f32] = &vals;

        let labels = kmeans_impl(
            slice,
            self.k,
            self.max_iterations,
            self.tolerance,
            self.seed,
        );

        let tensor = Tensor::<f32, B>::from_slice_on(shape, &labels, &device);

        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }

    /// Apply K-means clustering to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error when a sample is non-finite, backend storage is not
    /// host-addressable, or the output image cannot be constructed.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::native::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let samples = image.data_slice()?;
        validate_samples(samples)?;
        crate::native_output::from_values(
            image,
            kmeans_impl(
                samples,
                self.k,
                self.max_iterations,
                self.tolerance,
                self.seed,
            ),
            backend,
        )
    }
}

impl Default for KMeansSegmentation {
    fn default() -> Self {
        Self {
            k: 2,
            max_iterations: 100,
            tolerance: 1e-6,
            seed: 42,
        }
    }
}

/// Convenience function: apply K-Means with k clusters and default parameters.
pub fn kmeans_segment<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
    k: usize,
) -> anyhow::Result<Image<f32, B, D>> {
    KMeansSegmentation::new(k)?.apply(image)
}

// â”€â”€ Core implementation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Core K-Means implementation operating on a flat f32 slice.
///
/// Returns a `Vec<f32>` of cluster labels (0..Kâˆ’1).
fn kmeans_impl(
    data: &[f32],
    k: usize,
    max_iterations: usize,
    tolerance: f32,
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

    // â”€â”€ k-means++ initialization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let mut rng = Xorshift64::new(seed);
    let mut centroids = Vec::with_capacity(effective_k);

    // First centroid: random data point.
    let first_idx = rng.next_index(n);
    centroids.push(data[first_idx]);

    // Minimum-distance buffer. Sampling normalizes before squaring so finite
    // input magnitudes cannot overflow the concrete f32 arithmetic contract.
    let mut distances = vec![f32::MAX; n];

    for _ in 1..effective_k {
        // Update minimum distances to the most recently added centroid.
        let last_centroid = *centroids.last().expect("centroids must not be empty");
        for i in 0..n {
            let distance = (data[i] - last_centroid).abs();
            if distance < distances[i] {
                distances[i] = distance;
            }
        }

        let scale = distances.iter().copied().fold(0.0_f32, f32::max);
        if scale == 0.0 {
            // All remaining points are at existing centroids; duplicate last.
            centroids.push(last_centroid);
            continue;
        }
        let total: f32 = distances
            .iter()
            .map(|&distance| {
                let normalized = distance / scale;
                normalized * normalized
            })
            .sum();

        let target = rng.sample_unit() * total;
        let mut cumulative = 0.0_f32;
        let mut chosen = n - 1;
        for (i, &distance) in distances.iter().enumerate() {
            let normalized = distance / scale;
            cumulative += normalized * normalized;
            if cumulative >= target {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen]);
    }

    // â”€â”€ Lloyd's iterations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let mut labels = vec![0usize; n];
    // Pre-allocate accumulator buffers once; reset at the top of each iteration
    // to avoid effective_k Ã— 2 per-iteration heap allocations.
    let mut means = vec![0.0_f32; effective_k];
    let mut counts = vec![0.0_f32; effective_k];
    let mut anchors = vec![None; effective_k];

    for _iter in 0..max_iterations {
        // â”€â”€ Assignment step â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for i in 0..n {
            let val = data[i];
            let mut best_k = 0usize;
            let mut best_dist = f32::MAX;
            for (ci, &centroid) in centroids.iter().enumerate() {
                let d = (val - centroid).abs();
                if d < best_dist {
                    best_dist = d;
                    best_k = ci;
                }
            }
            labels[i] = best_k;
        }

        // â”€â”€ Update step â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        means.iter_mut().for_each(|value| *value = 0.0);
        counts.iter_mut().for_each(|value| *value = 0.0);
        anchors.iter_mut().for_each(|anchor| *anchor = None);

        for &label in &labels {
            counts[label] += 1.0;
        }
        for (&value, &label) in data.iter().zip(&labels) {
            anchors[label].get_or_insert(value);
        }
        for (&value, &label) in data.iter().zip(&labels) {
            let cluster = label;
            let anchor = anchors[cluster].expect("nonempty cluster has an anchor");
            means[cluster] += (value - anchor) / counts[cluster];
        }

        let mut max_shift = 0.0_f32;
        for ci in 0..effective_k {
            if counts[ci] > 0.0 {
                let new_centroid = anchors[ci].expect("nonempty cluster has an anchor") + means[ci];
                let shift = (new_centroid - centroids[ci]).abs();
                if shift > max_shift {
                    max_shift = shift;
                }
                centroids[ci] = new_centroid;
            }
            // Empty clusters retain their previous centroid position.
        }

        // â”€â”€ Convergence check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if max_shift < tolerance {
            break;
        }
    }

    // â”€â”€ Final assignment (in case we broke before the last assignment) â”€â”€â”€â”€â”€â”€
    for i in 0..n {
        let val = data[i];
        let mut best_k = 0usize;
        let mut best_dist = f32::MAX;
        for (ci, &centroid) in centroids.iter().enumerate() {
            let d = (val - centroid).abs();
            if d < best_dist {
                best_dist = d;
                best_k = ci;
            }
        }
        labels[i] = best_k;
    }

    // Construction bounds `k` to the exact-integer range of f32.
    labels.iter().map(|&label| label as f32).collect()
}

fn validate_samples(samples: &[f32]) -> anyhow::Result<()> {
    if let Some((index, value)) = samples
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("k-means sample at flat index {index} must be finite, got {value}");
    }
    if let Some((&first, rest)) = samples.split_first() {
        let (minimum, maximum) = rest
            .iter()
            .copied()
            .fold((first, first), |(low, high), value| {
                (low.min(value), high.max(value))
            });
        anyhow::ensure!(
            (maximum - minimum).is_finite(),
            "k-means sample range must be representable in f32, got [{minimum}, {maximum}]"
        );
    }
    Ok(())
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_kmeans.rs"]
mod tests_kmeans;
