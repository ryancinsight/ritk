//! Multi-Otsu threshold segmentation.
//!
//! # Mathematical Specification
//!
//! Multi-Otsu extends Otsu's method to K intensity classes (K − 1 thresholds).
//! Given threshold bin indices 0 = t₀ < t₁ < t₂ < … < t_{K−1} < t_K = N,
//! class k spans bins \[t_{k−1}, t_k − 1\].
//!
//! The objective is to maximise the total between-class variance:
//!
//!   σ²_B = Σ_{k=1}^{K}  P_k · (μ_k − μ_T)²
//!
//! where:
//! - h\[i\]  = count\[i\] / n_total                      (normalised histogram)
//! - P_k   = Σ_{i=t_{k−1}}^{t_k−1} h\[i\]            (class weight)
//! - μ_k   = (Σ_{i=t_{k−1}}^{t_k−1} i·h\[i\]) / P_k  (class mean bin index)
//! - μ_T   = Σ_{i=0}^{N−1} i·h\[i\]                  (global mean bin index)
//!
//! Algebraic identity: σ²_B = Σ_k P_k·μ_k² − μ_T²,  which equals the standard
//! Otsu between-class variance P₁·P₂·(μ₁−μ₂)² when K = 2.
//!
//! Efficient computation via prefix sums (both arrays of length N + 1):
//! - H\[t\] = Σ_{i=0}^{t−1} h\[i\]     (H\[0\] = 0, H\[N\] = 1)
//! - M\[t\] = Σ_{i=0}^{t−1} i·h\[i\]  (M\[0\] = 0)
//!
//! For class k spanning bins \[a, b\] = \[t_{k−1}, t_k − 1\]:
//!
//!   P_k         = H\[t_k\] − H\[t_{k−1}\]
//!   Σ i·h\[i\]  = M\[t_k\] − M\[t_{k−1}\]
//!
//! # Complexity
//! - Histogram construction:  O(n) voxels.
//! - Prefix-sum setup:        O(N) bins.
//! - Exhaustive search:       O(N^{K−1}) threshold combinations.
//!   For K = 3, N = 256:     ≈ 32 640 combinations (fast).
//!   For K = 2:              O(N) — degenerates to standard Otsu.
//!
//! # Threshold Conversion
//! Bin threshold index t maps to physical intensity:
//!
//!   t_intensity = x_min + t / (N − 1) · (x_max − x_min)

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

/// Multi-Otsu threshold segmentation into K intensity classes.
///
/// For K = 2 this degenerates to the standard Otsu method.
/// For K = 3 this finds the 2 thresholds that maximise total between-class variance.
pub struct MultiOtsuThreshold {
    /// Number of intensity classes to segment into. Must be ≥ 2.
    pub num_classes: usize,
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl MultiOtsuThreshold {
    /// Create a `MultiOtsuThreshold` with `num_classes` classes and 256 histogram bins.
    ///
    /// # Panics
    /// Panics if `num_classes < 2`.
    pub fn new(num_classes: usize) -> Self {
        assert!(num_classes >= 2, "num_classes must be ≥ 2");
        Self {
            num_classes,
            num_bins: 256,
        }
    }

    /// Create a `MultiOtsuThreshold` with a custom number of classes and bins.
    ///
    /// # Panics
    /// Panics if `num_classes < 2` or `num_bins < 2`.
    pub fn with_bins(num_classes: usize, num_bins: usize) -> Self {
        assert!(num_classes >= 2, "num_classes must be ≥ 2");
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self {
            num_classes,
            num_bins,
        }
    }

    /// Compute K − 1 Otsu thresholds for `image`.
    ///
    /// Returns a sorted `Vec<f32>` of K − 1 intensity thresholds.
    /// For K = 2, returns one threshold equivalent to standard Otsu.
    /// For a constant image, all thresholds are set to the constant intensity.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Vec<f32> {
        compute_multi_otsu_impl(image, self.num_classes, self.num_bins)
    }

    /// Apply the multi-Otsu thresholds to produce a label image.
    ///
    /// Pixel values are mapped to class indices {0, 1, …, K−1} as f32:
    /// - v < t₁              → 0.0
    /// - t₁ ≤ v < t₂         → 1.0
    /// - …
    /// - v ≥ t_{K−1}         → (K−1).0
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let thresholds = self.compute(image);
        apply_multi_otsu_labels(image, &thresholds)
    }
}

impl Default for MultiOtsuThreshold {
    fn default() -> Self {
        Self::new(3)
    }
}

/// Convenience function: compute K − 1 Multi-Otsu thresholds with 256 bins.
///
/// # Panics
/// Panics if `num_classes < 2`.
pub fn multi_otsu_threshold<B: Backend, const D: usize>(
    image: &Image<B, D>,
    num_classes: usize,
) -> Vec<f32> {
    assert!(num_classes >= 2, "num_classes must be ≥ 2");
    compute_multi_otsu_impl(image, num_classes, 256)
}

// ── Core implementation ───────────────────────────────────────────────────────

/// Compute K-1 Multi-Otsu thresholds directly from a flat `&[f32]` slice.
///
/// Zero-copy variant: accepts pre-extracted slice, eliminating `clone().into_data()`.
///
/// # Panics
/// Panics if `num_classes < 2`.
pub fn compute_multi_otsu_thresholds_from_slice(
    slice: &[f32],
    num_classes: usize,
    num_bins: usize,
) -> Vec<f32> {
    assert!(num_classes >= 2, "num_classes must be ≥ 2");
    let n = slice.len();
    let k_minus_1 = num_classes - 1;
    if n == 0 {
        return vec![0.0_f32; k_minus_1];
    }
    let x_min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if (x_max - x_min).abs() < f32::EPSILON {
        return vec![x_min; k_minus_1];
    }
    let range = x_max - x_min;
    let num_bins_m1 = (num_bins - 1) as f32;
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) / range * num_bins_m1).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
    let mut prefix_h = vec![0.0_f64; num_bins + 1];
    let mut prefix_m = vec![0.0_f64; num_bins + 1];
    for i in 0..num_bins {
        prefix_h[i + 1] = prefix_h[i] + h[i];
        prefix_m[i + 1] = prefix_m[i] + i as f64 * h[i];
    }
    let total_mu = prefix_m[num_bins];
    if num_bins < num_classes {
        return (1..num_classes)
            .map(|k| x_min + k as f32 / num_classes as f32 * range)
            .collect();
    }
    let mut current = Vec::with_capacity(k_minus_1);
    let mut best: (f64, Vec<usize>) = (f64::NEG_INFINITY, vec![0; k_minus_1]);
    search_thresholds(
        ThresholdSearchState {
            level: 0,
            k_minus_1,
            prev: 0,
            num_bins,
        },
        OtsuTables {
            prefix_h: &prefix_h,
            prefix_m: &prefix_m,
            total_mu,
        },
        &mut current,
        &mut best,
    );
    best.1
        .iter()
        .map(|&t| x_min + t as f32 / num_bins_m1 * range)
        .collect()
}

/// Delegates to [`compute_multi_otsu_thresholds_from_slice`] after extracting a
/// slice from the image tensor.
fn compute_multi_otsu_impl<B: Backend, const D: usize>(
    image: &Image<B, D>,
    num_classes: usize,
    num_bins: usize,
) -> Vec<f32> {
    let (vals, _) = extract_vec_infallible(image);
    let slice: &[f32] = &vals;
    compute_multi_otsu_thresholds_from_slice(slice, num_classes, num_bins)
}

/// Recursive exhaustive search over all valid K−1 threshold bin combinations.
///
/// Recursive traversal state for threshold placement.
#[derive(Debug, Clone, Copy)]
struct ThresholdSearchState {
    /// Current recursion depth (0-based threshold index).
    level: usize,
    /// K − 1 (total number of thresholds to place).
    k_minus_1: usize,
    /// Last placed threshold bin index (exclusive lower bound for next threshold).
    prev: usize,
    /// Total number of histogram bins.
    num_bins: usize,
}

/// Precomputed prefix-sum tables for the between-class variance objective.
#[derive(Debug, Clone, Copy)]
struct OtsuTables<'a> {
    /// Cumulative normalised histogram: `prefix_h[i] = Σ_{j=0}^{i} h[j]`.
    prefix_h: &'a [f64],
    /// Cumulative intensity-weighted histogram: `prefix_m[i] = Σ_{j=0}^{i} j·h[j]`.
    prefix_m: &'a [f64],
    /// Total mean of the distribution.
    total_mu: f64,
}

/// # Validity constraint
/// At depth `level` (0-based), given `prev` (the most recently placed threshold):
/// - lo = prev + 1                        (must strictly exceed prior threshold)
/// - hi_inclusive = N − k_minus_1 + level (must leave ≥ 1 bin per remaining class)
///
/// For K = 2 (k_minus_1 = 1), this reduces to a linear scan over [1, N−1].
fn search_thresholds(
    state: ThresholdSearchState,
    tables: OtsuTables<'_>,
    current: &mut Vec<usize>,
    best: &mut (f64, Vec<usize>),
) {
    let ThresholdSearchState {
        level,
        k_minus_1,
        prev,
        num_bins,
    } = state;
    let lo = prev + 1;
    // hi_inclusive guarantees each remaining class gets ≥ 1 bin.
    // Derivation: t + (k_minus_1 − level − 1) ≤ N − 1  →  t ≤ N − k_minus_1 + level.
    let hi_inclusive = num_bins - k_minus_1 + level;

    if lo > hi_inclusive {
        return;
    }

    for t in lo..=hi_inclusive {
        current.push(t);

        if level == k_minus_1 - 1 {
            // All K−1 thresholds have been placed; evaluate this combination.
            let sigma2 = between_class_variance(
                current,
                tables.prefix_h,
                tables.prefix_m,
                tables.total_mu,
                num_bins,
            );
            if sigma2 > best.0 {
                best.0 = sigma2;
                best.1 = current.clone();
            }
        } else {
            search_thresholds(
                ThresholdSearchState {
                    level: level + 1,
                    prev: t,
                    ..state
                },
                tables,
                current,
                best,
            );
        }

        current.pop();
    }
}

/// Compute the total between-class variance for a given set of threshold bin indices.
///
/// Uses prefix-sum arrays for O(K) evaluation per combination.
///
/// # Formula
/// σ²_B = Σ_{k=1}^{K} P_k · (μ_k − μ_T)²
///
/// # Class boundaries
/// With thresholds \[t₁, t₂, …, t_{K−1}\] and boundaries t₀ = 0, t_K = N:
/// Class k spans bins \[t_{k−1}, t_k − 1\], evaluated as prefix arrays \[t_{k−1}, t_k\).
///
/// # Equivalence with Otsu for K = 2
/// For K = 2: P₁·(μ₁−μ_T)² + P₂·(μ₂−μ_T)² = P₁·P₂·(μ₁−μ₂)² (proven by substituting
/// μ_T = P₁·μ₁ + P₂·μ₂ and simplifying).
fn between_class_variance(
    thresholds: &[usize],
    prefix_h: &[f64],
    prefix_m: &[f64],
    total_mu: f64,
    num_bins: usize,
) -> f64 {
    let k = thresholds.len() + 1; // Number of classes.
    let mut sigma2 = 0.0_f64;

    for class_idx in 0..k {
        // Lower boundary of class class_idx (inclusive bin index).
        let a = if class_idx == 0 {
            0
        } else {
            thresholds[class_idx - 1]
        };
        // Upper boundary of class class_idx (exclusive; maps to prefix index).
        let b = if class_idx == k - 1 {
            num_bins
        } else {
            thresholds[class_idx]
        };

        // P_k = H[b] - H[a]  (sum of h over bins [a, b−1]).
        let p = prefix_h[b] - prefix_h[a];
        if p < super::PROB_ZERO_GUARD {
            // Empty class: contributes zero.
            continue;
        }

        // μ_k = (M[b] - M[a]) / P_k.
        let mu = (prefix_m[b] - prefix_m[a]) / p;
        sigma2 += p * (mu - total_mu) * (mu - total_mu);
    }

    sigma2
}

/// Apply a sorted list of intensity thresholds to assign class labels.
///
/// For each pixel value v:
///   label(v) = |{t ∈ thresholds : v ≥ t}|
///
/// This maps pixels to {0, 1, …, K−1} as f32 values.
fn apply_multi_otsu_labels<B: Backend, const D: usize>(
    image: &Image<B, D>,
    thresholds: &[f32],
) -> Image<B, D> {
    let device = image.data().device();
    let shape: [usize; D] = image.shape();
    let (img_vals, _shape) = extract_vec_infallible(image);
    let slice: &[f32] = &img_vals;

    let output: Vec<f32> = slice
        .iter()
        .map(|&v| thresholds.iter().filter(|&&t| v >= t).count() as f32)
        .collect();

    let tensor = Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

    Image::new(
        tensor,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
    )
}

#[cfg(test)]
#[path = "tests_multi_otsu_base.rs"]
mod tests_multi_otsu_base;

#[cfg(test)]
#[path = "tests_multi_otsu_ext.rs"]
mod tests_multi_otsu_ext;

#[cfg(test)]
#[path = "tests_multi_otsu_k4k5.rs"]
mod tests_multi_otsu_k4k5;
