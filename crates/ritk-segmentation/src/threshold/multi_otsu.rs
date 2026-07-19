//! Multi-Otsu threshold segmentation.
//!
//! # Mathematical Specification
//!
//! Multi-Otsu extends Otsu's method to K intensity classes (K âˆ’ 1 thresholds).
//! Given threshold bin indices 0 = tâ‚€ < tâ‚ < tâ‚‚ < â€¦ < t_{Kâˆ’1} < t_K = N,
//! class k spans bins \[t_{kâˆ’1}, t_k âˆ’ 1\].
//!
//! The objective is to maximise the total between-class variance:
//!
//!   ÏƒÂ²_B = Î£_{k=1}^{K}  P_k Â· (Î¼_k âˆ’ Î¼_T)Â²
//!
//! where:
//! - h\[i\]  = count\[i\] / n_total                      (normalised histogram)
//! - P_k   = Î£_{i=t_{kâˆ’1}}^{t_kâˆ’1} h\[i\]            (class weight)
//! - Î¼_k   = (Î£_{i=t_{kâˆ’1}}^{t_kâˆ’1} iÂ·h\[i\]) / P_k  (class mean bin index)
//! - Î¼_T   = Î£_{i=0}^{Nâˆ’1} iÂ·h\[i\]                  (global mean bin index)
//!
//! Algebraic identity: ÏƒÂ²_B = Î£_k P_kÂ·Î¼_kÂ² âˆ’ Î¼_TÂ²,  which equals the standard
//! Otsu between-class variance Pâ‚Â·Pâ‚‚Â·(Î¼â‚âˆ’Î¼â‚‚)Â² when K = 2.
//!
//! Efficient computation via prefix sums (both arrays of length N + 1):
//! - H\[t\] = Î£_{i=0}^{tâˆ’1} h\[i\]     (H\[0\] = 0, H\[N\] = 1)
//! - M\[t\] = Î£_{i=0}^{tâˆ’1} iÂ·h\[i\]  (M\[0\] = 0)
//!
//! For class k spanning bins \[a, b\] = \[t_{kâˆ’1}, t_k âˆ’ 1\]:
//!
//!   P_k         = H\[t_k\] âˆ’ H\[t_{kâˆ’1}\]
//!   Î£ iÂ·h\[i\]  = M\[t_k\] âˆ’ M\[t_{kâˆ’1}\]
//!
//! # Complexity
//! - Histogram construction:  O(n) voxels.
//! - Prefix-sum setup:        O(N) bins.
//! - Exact dynamic program:   O(KÂ·NÂ²) time and O(KÂ·N) storage.
//!
//! # Threshold Conversion
//! Each boundary bin index t separates classes [.., tâˆ’1] | [t, ..]; ITK reports
//! the boundary intensity as the left edge of bin t under its histogram geometry
//! (see `auto_threshold`):
//!
//!   t_intensity = x_min + t Â· bin_width
//!
//! # Non-finite intensities
//!
//! NaN and Â±Inf samples are excluded from histogram statistics and always map
//! to background class `0.0`. An input with no finite samples returns Kâˆ’1 zero
//! thresholds and an all-background label image.

use ritk_image::tensor::{Backend, Tensor};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

use super::auto_threshold::{build_histogram, finite_bounds, itk_bin_width};

/// Multi-Otsu threshold segmentation into K intensity classes.
///
/// For K = 2 this degenerates to the standard Otsu method.
/// For K = 3 this finds the 2 thresholds that maximise total between-class variance.
pub struct MultiOtsuThreshold {
    /// Number of intensity classes to segment into. Must be â‰¥ 2.
    num_classes: usize,
    /// Number of equally-spaced histogram bins. Default 256.
    num_bins: usize,
}

impl MultiOtsuThreshold {
    /// Create a `MultiOtsuThreshold` with `num_classes` classes and 256 histogram bins.
    ///
    /// # Panics
    /// Panics unless `2 <= num_classes <= 256`.
    pub fn new(num_classes: usize) -> Self {
        assert!(num_classes >= 2, "num_classes must be ≥ 2");
        assert!(num_classes <= 256, "num_classes must not exceed num_bins");
        Self {
            num_classes,
            num_bins: 256,
        }
    }

    /// Create a `MultiOtsuThreshold` with a custom number of classes and bins.
    ///
    /// # Panics
    /// Panics unless `2 <= num_classes <= num_bins`.
    pub fn with_bins(num_classes: usize, num_bins: usize) -> Self {
        assert!(num_classes >= 2, "num_classes must be ≥ 2");
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        assert!(
            num_classes <= num_bins,
            "num_classes must not exceed num_bins"
        );
        Self {
            num_classes,
            num_bins,
        }
    }

    /// Return the number of output intensity classes.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Return the histogram bin count.
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Compute K âˆ’ 1 Otsu thresholds for `image`.
    ///
    /// Returns a sorted `Vec<f32>` of K âˆ’ 1 intensity thresholds.
    /// For K = 2, returns one threshold equivalent to standard Otsu.
    /// For a constant image, all thresholds are set to the constant intensity.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> Vec<f32> {
        compute_multi_otsu_impl(image, self.num_classes, self.num_bins)
    }

    /// Apply the multi-Otsu thresholds to produce a label image.
    ///
    /// Pixel values are mapped to class indices {0, 1, â€¦, Kâˆ’1} as f32:
    /// - v < tâ‚              â†’ 0.0
    /// - tâ‚ â‰¤ v < tâ‚‚         â†’ 1.0
    /// - â€¦
    /// - v â‰¥ t_{Kâˆ’1}         â†’ (Kâˆ’1).0
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        // Extract once and reuse the slice for both threshold search and labelling
        // (the previous compute()-then-label form cloned and copied the whole
        // volume twice per `apply`).
        let (vals, shape) = extract_vec_infallible(image);
        let (_, output) = multi_otsu_labels_from_slice(&vals, self.num_classes, self.num_bins);

        let device = B::default();
        let tensor = Tensor::<f32, B>::from_slice_on(shape, &output, &device);
        Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
        .expect("invariant: segmentation output tensor preserves the image rank")
    }

    /// Compute thresholds directly from a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error when backend storage is not host-addressable.
    pub fn compute_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
    ) -> anyhow::Result<Vec<f32>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        Ok(compute_multi_otsu_thresholds_from_slice(
            image.data_slice()?,
            self.num_classes,
            self.num_bins,
        ))
    }

    /// Apply Multi-Otsu labels to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error when backend storage is not host-addressable or the
    /// output image cannot be constructed.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        self.apply_native_with_thresholds(image, backend)
            .map(|(labels, _)| labels)
    }

    /// Compute thresholds and native labels from one host-slice extraction.
    ///
    /// # Errors
    ///
    /// Returns an error when backend storage is not host-addressable or the
    /// output image cannot be constructed.
    pub fn apply_native_with_thresholds<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<(ritk_image::Image<f32, B, D>, Vec<f32>)>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (thresholds, output) =
            multi_otsu_labels_from_slice(image.data_slice()?, self.num_classes, self.num_bins);
        let labels = ritk_image::Image::from_flat_on(
            output,
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )?;
        Ok((labels, thresholds))
    }
}

impl Default for MultiOtsuThreshold {
    fn default() -> Self {
        Self::new(3)
    }
}

/// Convenience function: compute K âˆ’ 1 Multi-Otsu thresholds with 256 bins.
///
/// # Panics
/// Panics unless `2 <= num_classes <= 256`.
pub fn multi_otsu_threshold<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
    num_classes: usize,
) -> Vec<f32> {
    assert!(num_classes >= 2, "num_classes must be ≥ 2");
    assert!(num_classes <= 256, "num_classes must not exceed num_bins");
    compute_multi_otsu_impl(image, num_classes, 256)
}

// â”€â”€ Core implementation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Compute K-1 Multi-Otsu thresholds directly from a flat `&[f32]` slice.
///
/// Zero-copy variant: accepts pre-extracted slice, eliminating `clone().into_data()`.
///
/// # Panics
/// Panics unless `2 <= num_classes <= num_bins`.
pub fn compute_multi_otsu_thresholds_from_slice(
    slice: &[f32],
    num_classes: usize,
    num_bins: usize,
) -> Vec<f32> {
    assert!(num_classes >= 2, "num_classes must be ≥ 2");
    assert!(num_bins >= 2, "num_bins must be ≥ 2");
    assert!(
        num_classes <= num_bins,
        "num_classes must not exceed num_bins"
    );
    let k_minus_1 = num_classes - 1;
    let Some((x_min, x_max, n)) = finite_bounds(slice) else {
        return vec![0.0_f32; k_minus_1];
    };
    if (x_max - x_min).abs() < f32::EPSILON {
        return vec![x_min; k_minus_1];
    }
    // ITK histogram geometry (shared with the single-threshold calculators).
    let bin_width = itk_bin_width(x_min, x_max, num_bins);
    let counts = build_histogram(slice, num_bins, x_min, x_max);
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
    let mut prefix_h = vec![0.0_f64; num_bins + 1];
    let mut prefix_m = vec![0.0_f64; num_bins + 1];
    for i in 0..num_bins {
        prefix_h[i + 1] = prefix_h[i] + h[i];
        prefix_m[i + 1] = prefix_m[i] + i as f64 * h[i];
    }
    let total_mu = prefix_m[num_bins];
    let best = optimal_threshold_bins(num_classes, &prefix_h, &prefix_m, total_mu, num_bins);
    // Each boundary bin index t separates classes [.., tâˆ’1] | [t, ..]; ITK reports
    // the boundary intensity = left edge of bin t = x_min + tÂ·bin_width.
    best.iter()
        .map(|&t| (x_min as f64 + t as f64 * bin_width) as f32)
        .collect()
}

fn multi_otsu_labels_from_slice(
    slice: &[f32],
    num_classes: usize,
    num_bins: usize,
) -> (Vec<f32>, Vec<f32>) {
    let thresholds = compute_multi_otsu_thresholds_from_slice(slice, num_classes, num_bins);
    let labels = slice
        .iter()
        .map(|&value| {
            if value.is_finite() {
                thresholds
                    .iter()
                    .filter(|&&threshold| value >= threshold)
                    .count() as f32
            } else {
                0.0
            }
        })
        .collect();
    (thresholds, labels)
}

/// Delegates to [`compute_multi_otsu_thresholds_from_slice`] after extracting a
/// slice from the image tensor.
fn compute_multi_otsu_impl<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
    num_classes: usize,
    num_bins: usize,
) -> Vec<f32> {
    let (vals, _) = extract_vec_infallible(image);
    let slice: &[f32] = &vals;
    compute_multi_otsu_thresholds_from_slice(slice, num_classes, num_bins)
}

/// Compute the exact optimal K-class partition with dynamic programming.
///
/// `score[c][b]` is the maximum additive between-class variance for partitioning
/// bins `[0, b)` into `c` non-empty contiguous classes:
///
/// `score[c][b] = max score[c-1][a] + variance(a, b)` for `c-1 <= a < b`.
///
/// Ascending split scans and strict replacement preserve the lexicographically
/// earliest threshold set when objectives tie.
fn optimal_threshold_bins(
    num_classes: usize,
    prefix_h: &[f64],
    prefix_m: &[f64],
    total_mu: f64,
    num_bins: usize,
) -> Vec<usize> {
    let row_width = num_bins + 1;
    let mut score = vec![f64::NEG_INFINITY; (num_classes + 1) * row_width];
    let mut split = vec![0_usize; (num_classes + 1) * row_width];
    score[0] = 0.0;

    for classes in 1..=num_classes {
        for upper in classes..=num_bins {
            let index = classes * row_width + upper;
            for lower in (classes - 1)..upper {
                let previous = score[(classes - 1) * row_width + lower];
                if !previous.is_finite() {
                    continue;
                }
                let candidate =
                    previous + class_variance(prefix_h, prefix_m, total_mu, lower, upper);
                if candidate > score[index] {
                    score[index] = candidate;
                    split[index] = lower;
                }
            }
        }
    }

    let mut thresholds = vec![0_usize; num_classes - 1];
    let mut upper = num_bins;
    for classes in (2..=num_classes).rev() {
        let lower = split[classes * row_width + upper];
        thresholds[classes - 2] = lower;
        upper = lower;
    }
    thresholds
}

fn class_variance(
    prefix_h: &[f64],
    prefix_m: &[f64],
    total_mu: f64,
    lower: usize,
    upper: usize,
) -> f64 {
    let probability = prefix_h[upper] - prefix_h[lower];
    if probability < super::PROB_ZERO_GUARD {
        return 0.0;
    }
    let mean = (prefix_m[upper] - prefix_m[lower]) / probability;
    probability * (mean - total_mu) * (mean - total_mu)
}

/// Compute the total between-class variance for a given set of threshold bin indices.
///
/// Uses prefix-sum arrays for O(K) evaluation per combination.
///
/// # Formula
/// ÏƒÂ²_B = Î£_{k=1}^{K} P_k Â· (Î¼_k âˆ’ Î¼_T)Â²
///
/// # Class boundaries
/// With thresholds \[tâ‚, tâ‚‚, â€¦, t_{Kâˆ’1}\] and boundaries tâ‚€ = 0, t_K = N:
/// Class k spans bins \[t_{kâˆ’1}, t_k âˆ’ 1\], evaluated as prefix arrays \[t_{kâˆ’1}, t_k\).
///
/// # Equivalence with Otsu for K = 2
/// For K = 2: Pâ‚Â·(Î¼â‚âˆ’Î¼_T)Â² + Pâ‚‚Â·(Î¼â‚‚âˆ’Î¼_T)Â² = Pâ‚Â·Pâ‚‚Â·(Î¼â‚âˆ’Î¼â‚‚)Â² (proven by substituting
/// Î¼_T = Pâ‚Â·Î¼â‚ + Pâ‚‚Â·Î¼â‚‚ and simplifying).
#[cfg(test)]
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

        sigma2 += class_variance(prefix_h, prefix_m, total_mu, a, b);
    }

    sigma2
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
