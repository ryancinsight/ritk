//! Nyúl-Udupa piecewise-linear histogram standardization.
//!
//! # Mathematical Specification
//!
//! Given K training images {I₁, …, I_K} and a set of target percentile ranks
//! P = {p₁, p₂, …, p_M} (e.g. {1, 10, 20, …, 90, 99}):
//!
//! ## Training phase (`learn_standard`)
//!
//! 1. For each training image Iₖ, compute the intensity landmarks:
//!
//!      Lₖ = [ Q(Iₖ, p₁), Q(Iₖ, p₂), …, Q(Iₖ, p_M) ]
//!
//!    where Q(I, p) is the p-th percentile of the image intensities, computed
//!    via linear interpolation on the sorted value array.
//!
//! 2. Compute the standard (average) landmark vector:
//!
//!      S_j = (1/K) · Σₖ Lₖⱼ,   j = 1, …, M
//!
//! ## Transform phase (`apply`)
//!
//! For a new image I with landmarks L = [ Q(I, p₁), …, Q(I, p_M) ]:
//!
//! 1. For each voxel intensity v, find the interval [Lⱼ, Lⱼ₊₁] containing v.
//! 2. Apply piecewise-linear interpolation to map v → v':
//!
//!      v' = Sⱼ + (v − Lⱼ) · (Sⱼ₊₁ − Sⱼ) / (Lⱼ₊₁ − Lⱼ)
//!
//! 3. Values below L₁ are clamped to S₁; values above L_M are clamped to S_M.
//!
//! ## Percentile computation
//!
//! The p-th percentile (p ∈ [0, 100]) of a sorted array V of length n is
//! computed via linear interpolation:
//!
//!   rank = p / 100 · (n − 1)
//!   lo   = ⌊rank⌋,  hi = ⌈rank⌉
//!   Q    = V[lo] + (rank − lo) · (V[hi] − V[lo])
//!
//! # References
//!
//! - Nyúl, L. G., Udupa, J. K., & Zhang, X. (2000). New variants of a method
//!   of MRI scale standardization. *IEEE Trans. Med. Imaging*, 19(2), 143–150.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

// ── Percentile Helper ─────────────────────────────────────────────────────────

/// Compute the p-th percentile of a sorted slice via linear interpolation.
///
/// # Arguments
/// * `sorted` – Non-empty slice sorted in non-decreasing order.
/// * `p`      – Percentile rank in \[0, 100\].
///
/// # Formula
/// ```text
/// rank = p / 100 · (n − 1)
/// Q    = sorted[⌊rank⌋] + (rank − ⌊rank⌋) · (sorted[⌈rank⌉] − sorted[⌊rank⌋])
/// ```
///
/// # Panics
/// Panics if `sorted` is empty.
fn compute_percentile(sorted: &[f32], p: f64) -> f32 {
    assert!(
        !sorted.is_empty(),
        "compute_percentile requires non-empty input"
    );
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil().min((n - 1) as f64) as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = (rank - lo as f64) as f32;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ── Sort Helper ───────────────────────────────────────────────────────────────

/// Sort a mutable slice of f32, treating NaN as greater than all finite values.
#[inline]
fn sort_f32(values: &mut [f32]) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

// ── Piecewise-Linear Mapping ──────────────────────────────────────────────────

/// Apply piecewise-linear mapping from `source_landmarks` to `target_landmarks`.
///
/// Values below `source_landmarks[0]` clamp to `target_landmarks[0]`.
/// Values above `source_landmarks[last]` clamp to `target_landmarks[last]`.
/// Within each interval \[Lⱼ, Lⱼ₊₁\], linearly interpolates to \[Sⱼ, Sⱼ₊₁\].
#[inline]
fn piecewise_linear_map(value: f32, source_landmarks: &[f32], target_landmarks: &[f32]) -> f32 {
    debug_assert_eq!(source_landmarks.len(), target_landmarks.len());
    let m = source_landmarks.len();

    // Clamp below first landmark.
    if value <= source_landmarks[0] {
        return target_landmarks[0];
    }
    // Clamp above last landmark.
    if value >= source_landmarks[m - 1] {
        return target_landmarks[m - 1];
    }

    // Find the interval [Lⱼ, Lⱼ₊₁] containing value via linear scan.
    // For typical landmark counts (≤ 11 entries), linear scan is faster
    // than binary search due to branch prediction and cache locality.
    for j in 0..m - 1 {
        if value <= source_landmarks[j + 1] {
            let denom = source_landmarks[j + 1] - source_landmarks[j];
            if denom.abs() < f32::EPSILON {
                // Degenerate interval: source landmarks coincide.
                return target_landmarks[j];
            }
            let t = (value - source_landmarks[j]) / denom;
            return target_landmarks[j] + t * (target_landmarks[j + 1] - target_landmarks[j]);
        }
    }

    // Fallback (unreachable for well-formed inputs).
    target_landmarks[m - 1]
}

// ── Nyúl-Udupa Normalizer ─────────────────────────────────────────────────────

/// Nyúl-Udupa piecewise-linear histogram standardization normalizer.
///
/// Two-phase normalizer:
/// 1. **Training** (`learn_standard`): learns average intensity landmarks from
///    a set of training images.
/// 2. **Application** (`apply`): maps a new image's intensity landmarks to the
///    learned standard via piecewise-linear interpolation.
///
/// # Reference
/// Nyúl, L. G., Udupa, J. K., & Zhang, X. (2000). New variants of a method
/// of MRI scale standardization. *IEEE Trans. Med. Imaging*, 19(2), 143–150.
pub struct NyulUdupaNormalizer {
    /// Percentile ranks used as landmarks (values in \[0, 100\]).
    /// Default: \[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99\].
    pub percentiles: Vec<f64>,
    /// Learned standard landmark intensities.
    /// `None` before `learn_standard` has been called.
    pub standard_landmarks: Option<Vec<f32>>,
}

impl NyulUdupaNormalizer {
    /// Create a normalizer with default percentile landmarks
    /// \[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99\].
    pub fn new() -> Self {
        Self {
            percentiles: vec![
                1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0,
            ],
            standard_landmarks: None,
        }
    }

    /// Create a normalizer with custom percentile landmarks.
    ///
    /// # Arguments
    /// * `percentiles` – Percentile ranks in \[0, 100\], must be sorted in
    ///   strictly ascending order with at least 2 entries.
    ///
    /// # Panics
    /// Panics if `percentiles` has fewer than 2 entries or is not strictly ascending.
    pub fn with_percentiles(percentiles: Vec<f64>) -> Self {
        assert!(
            percentiles.len() >= 2,
            "at least 2 percentile landmarks required, got {}",
            percentiles.len()
        );
        for i in 1..percentiles.len() {
            assert!(
                percentiles[i] > percentiles[i - 1],
                "percentiles must be strictly ascending: p[{}]={} <= p[{}]={}",
                i,
                percentiles[i],
                i - 1,
                percentiles[i - 1]
            );
        }
        Self {
            percentiles,
            standard_landmarks: None,
        }
    }

    /// Learn the standard intensity landmarks by averaging per-image landmarks
    /// across all training images.
    ///
    /// # Algorithm
    /// For each image Iₖ:
    /// 1. Extract voxel intensities and sort.
    /// 2. Compute landmarks Lₖ = \[Q(Iₖ, p₁), …, Q(Iₖ, p_M)\].
    ///
    /// Standard landmarks: Sⱼ = (1/K) · Σₖ Lₖⱼ.
    ///
    /// # Panics
    /// Panics if `images` is empty.
    pub fn learn_standard<B: Backend, const D: usize>(&mut self, images: &[&Image<B, D>]) {
        assert!(!images.is_empty(), "at least one training image required");

        let m = self.percentiles.len();
        let k = images.len();
        let mut sum_landmarks = vec![0.0f64; m];

        for image in images {
            let data = image.data().clone().into_data();
            let slice = data.as_slice::<f32>().expect("f32 image tensor data");
            let mut values: Vec<f32> = slice.to_vec();
            sort_f32(&mut values);

            for (j, &p) in self.percentiles.iter().enumerate() {
                sum_landmarks[j] += compute_percentile(&values, p) as f64;
            }
        }

        self.standard_landmarks = Some(
            sum_landmarks
                .iter()
                .map(|&s| (s / k as f64) as f32)
                .collect(),
        );
    }

    /// Apply the learned piecewise-linear mapping to a new image.
    ///
    /// Computes the input image's own landmarks, then maps each voxel intensity
    /// from the input landmark space to the standard landmark space via
    /// piecewise-linear interpolation.
    ///
    /// # Errors
    /// Returns `Err` if `learn_standard` has not been called (i.e.
    /// `standard_landmarks` is `None`).
    ///
    /// # Spatial metadata
    /// The output image preserves origin, spacing, and direction from the input.
    pub fn apply<B: Backend, const D: usize>(
        &self,
        image: &Image<B, D>,
    ) -> anyhow::Result<Image<B, D>> {
        let standard = self.standard_landmarks.as_ref().ok_or_else(|| {
            anyhow::anyhow!("standard landmarks not learned; call learn_standard before apply")
        })?;

        let device = image.data().device();
        let shape: [usize; D] = image.shape();

        // ── 1. Extract and sort voxel intensities ─────────────────────────────
        let img_data = image.data().clone().into_data();
        let img_slice = img_data.as_slice::<f32>().expect("f32 image tensor data");
        let mut sorted: Vec<f32> = img_slice.to_vec();
        sort_f32(&mut sorted);

        // ── 2. Compute input image landmarks ──────────────────────────────────
        let source_landmarks: Vec<f32> = self
            .percentiles
            .iter()
            .map(|&p| compute_percentile(&sorted, p))
            .collect();

        // ── 3. Apply piecewise-linear mapping ─────────────────────────────────
        let n_total: usize = shape.iter().product();
        let mut output: Vec<f32> = Vec::with_capacity(n_total);
        for &v in img_slice.iter() {
            output.push(piecewise_linear_map(v, &source_landmarks, standard));
        }

        // ── 4. Reconstruct image ──────────────────────────────────────────────
        let out_tensor =
            Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

        Ok(Image::new(
            out_tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

impl Default for NyulUdupaNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
        let n = data.len();
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

    fn get_values(image: &Image<TestBackend, 1>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── Percentile helper tests ───────────────────────────────────────────────

    #[test]
    fn test_percentile_min_and_max() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p0 = compute_percentile(&sorted, 0.0);
        let p100 = compute_percentile(&sorted, 100.0);
        assert!((p0 - 1.0).abs() < 1e-6, "p0 = min = 1.0, got {}", p0);
        assert!((p100 - 5.0).abs() < 1e-6, "p100 = max = 5.0, got {}", p100);
    }

    #[test]
    fn test_percentile_median() {
        // Odd length: [1, 2, 3, 4, 5] → p50 = 3.0.
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p50 = compute_percentile(&sorted, 50.0);
        assert!((p50 - 3.0).abs() < 1e-6, "p50 = 3.0, got {}", p50);
    }

    #[test]
    fn test_percentile_interpolation() {
        // [0, 10, 20, 30] (n=4). p50: rank = 0.5 * 3 = 1.5 → lerp(10, 20, 0.5) = 15.
        let sorted = vec![0.0, 10.0, 20.0, 30.0];
        let p50 = compute_percentile(&sorted, 50.0);
        assert!((p50 - 15.0).abs() < 1e-4, "p50 = 15.0, got {}", p50);
    }

    #[test]
    fn test_percentile_single_element() {
        let sorted = vec![42.0];
        let p = compute_percentile(&sorted, 50.0);
        assert!((p - 42.0).abs() < 1e-6, "single element → 42.0, got {}", p);
    }

    // ── Piecewise-linear mapping tests ────────────────────────────────────────

    #[test]
    fn test_piecewise_identity_mapping() {
        // When source = target, the mapping is the identity.
        let landmarks = vec![0.0, 50.0, 100.0];
        let val = piecewise_linear_map(25.0, &landmarks, &landmarks);
        assert!(
            (val - 25.0).abs() < 1e-5,
            "identity map: expected 25.0, got {}",
            val
        );
    }

    #[test]
    fn test_piecewise_clamp_below() {
        let src = vec![10.0, 50.0, 90.0];
        let tgt = vec![0.0, 0.5, 1.0];
        let val = piecewise_linear_map(5.0, &src, &tgt);
        assert!(
            (val - 0.0).abs() < 1e-5,
            "below min clamps to target[0], got {}",
            val
        );
    }

    #[test]
    fn test_piecewise_clamp_above() {
        let src = vec![10.0, 50.0, 90.0];
        let tgt = vec![0.0, 0.5, 1.0];
        let val = piecewise_linear_map(100.0, &src, &tgt);
        assert!(
            (val - 1.0).abs() < 1e-5,
            "above max clamps to target[last], got {}",
            val
        );
    }

    #[test]
    fn test_piecewise_midpoint() {
        // src = [0, 100], tgt = [0, 200]. v = 50 → 100.
        let src = vec![0.0, 100.0];
        let tgt = vec![0.0, 200.0];
        let val = piecewise_linear_map(50.0, &src, &tgt);
        assert!(
            (val - 100.0).abs() < 1e-4,
            "midpoint: expected 100.0, got {}",
            val
        );
    }

    // ── NyulUdupaNormalizer: positive tests ───────────────────────────────────

    #[test]
    fn test_learn_and_apply_single_image_roundtrip() {
        // Learning from a single image and applying to the same image:
        // source landmarks == standard landmarks, so the piecewise-linear
        // mapping is the identity *within* the [p1, p99] landmark range.
        // Values outside [p1, p99] are clamped to the boundary landmarks
        // per the Nyúl-Udupa algorithm specification.
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let image = make_image_1d(data.clone());

        let mut normalizer = NyulUdupaNormalizer::new();
        normalizer.learn_standard(&[&image]);

        let standard = normalizer.standard_landmarks.as_ref().unwrap();
        let lo = standard[0]; // p1 landmark value
        let hi = *standard.last().unwrap(); // p99 landmark value

        let result = normalizer.apply(&image).expect("apply must succeed");
        let result_vals = get_values(&result);

        // Interior values (between p1 and p99) are identity-mapped.
        for (i, (&original, &mapped)) in data.iter().zip(result_vals.iter()).enumerate() {
            if original >= lo && original <= hi {
                assert!(
                    (original - mapped).abs() < 1e-2,
                    "interior voxel {}: original = {}, mapped = {}, diff = {}",
                    i,
                    original,
                    mapped,
                    (original - mapped).abs()
                );
            }
        }

        // Extreme values are clamped to the boundary landmarks.
        assert!(
            (result_vals[0] - lo).abs() < 1e-2,
            "below-p1 voxel clamped to p1 landmark {}, got {}",
            lo,
            result_vals[0]
        );
        let last = *result_vals.last().unwrap();
        assert!(
            (last - hi).abs() < 1e-2,
            "above-p99 voxel clamped to p99 landmark {}, got {}",
            hi,
            last
        );
    }

    #[test]
    fn test_standardize_two_different_ranges_converge() {
        // Image A: intensities in [0, 100].
        // Image B: intensities in [1000, 2000].
        // After learning the standard from both and applying to each,
        // their intensity ranges must be closer together than before.
        let data_a: Vec<f32> = (0..200).map(|i| i as f32 * 0.5).collect();
        let data_b: Vec<f32> = (0..200).map(|i| 1000.0 + i as f32 * 5.0).collect();

        let image_a = make_image_1d(data_a);
        let image_b = make_image_1d(data_b);

        let mut normalizer = NyulUdupaNormalizer::new();
        normalizer.learn_standard(&[&image_a, &image_b]);

        let result_a = normalizer.apply(&image_a).expect("apply A");
        let result_b = normalizer.apply(&image_b).expect("apply B");

        let vals_a = get_values(&result_a);
        let vals_b = get_values(&result_b);

        // Compute the range of each standardized image.
        let min_a = vals_a.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_a = vals_a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_b = vals_b.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_b = vals_b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range_a = max_a - min_a;
        let range_b = max_b - min_b;

        // Original range difference: |100 − 1000| = 900.
        // After standardization the ranges must be closer.
        let original_range_diff = (100.0f32 - 1000.0f32).abs();
        let standardized_range_diff = (range_a - range_b).abs();

        assert!(
            standardized_range_diff < original_range_diff,
            "standardization must bring ranges closer: original diff = {}, standardized diff = {}",
            original_range_diff,
            standardized_range_diff
        );
    }

    #[test]
    fn test_preserves_spatial_metadata() {
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let image = make_image_1d(data);

        let mut normalizer = NyulUdupaNormalizer::new();
        normalizer.learn_standard(&[&image]);

        let result = normalizer.apply(&image).expect("apply");
        assert_eq!(result.origin(), image.origin(), "origin must be preserved");
        assert_eq!(
            result.spacing(),
            image.spacing(),
            "spacing must be preserved"
        );
        assert_eq!(
            result.direction(),
            image.direction(),
            "direction must be preserved"
        );
        assert_eq!(result.shape(), image.shape(), "shape must be preserved");
    }

    #[test]
    fn test_custom_percentiles() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let image = make_image_1d(data.clone());

        let mut normalizer =
            NyulUdupaNormalizer::with_percentiles(vec![5.0, 25.0, 50.0, 75.0, 95.0]);
        normalizer.learn_standard(&[&image]);

        let standard = normalizer.standard_landmarks.as_ref().unwrap();
        let lo = standard[0]; // p5 landmark value
        let hi = *standard.last().unwrap(); // p95 landmark value

        let result = normalizer.apply(&image).expect("apply");
        let result_vals = get_values(&result);

        // Interior values (between p5 and p95) are identity-mapped.
        for (i, (&original, &mapped)) in data.iter().zip(result_vals.iter()).enumerate() {
            if original >= lo && original <= hi {
                assert!(
                    (original - mapped).abs() < 1e-2,
                    "interior voxel {}: original = {}, mapped = {}, diff = {}",
                    i,
                    original,
                    mapped,
                    (original - mapped).abs()
                );
            }
        }

        // Values below p5 clamp to the p5 landmark.
        assert!(
            (result_vals[0] - lo).abs() < 1e-2,
            "below-p5 voxel clamped to p5 landmark {}, got {}",
            lo,
            result_vals[0]
        );
        // Values above p95 clamp to the p95 landmark.
        let last = *result_vals.last().unwrap();
        assert!(
            (last - hi).abs() < 1e-2,
            "above-p95 voxel clamped to p95 landmark {}, got {}",
            hi,
            last
        );
    }

    #[test]
    fn test_learn_from_multiple_images_averages_landmarks() {
        // Image A: [0..100), Image B: [100..200).
        // Standard landmarks should be the average of A and B landmarks.
        let data_a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..100).map(|i| 100.0 + i as f32).collect();
        let image_a = make_image_1d(data_a);
        let image_b = make_image_1d(data_b);

        let mut normalizer = NyulUdupaNormalizer::new();
        normalizer.learn_standard(&[&image_a, &image_b]);

        let standard = normalizer.standard_landmarks.as_ref().unwrap();

        // For uniform data [0..100): p50 ≈ 49.5. For [100..200): p50 ≈ 149.5.
        // Average p50 ≈ 99.5. Verify it's in a reasonable range.
        let p50_idx = normalizer
            .percentiles
            .iter()
            .position(|&p| (p - 50.0).abs() < 1e-9)
            .unwrap();
        let avg_p50 = standard[p50_idx];
        assert!(
            (avg_p50 - 99.5).abs() < 1.0,
            "average p50 ≈ 99.5, got {}",
            avg_p50
        );
    }

    // ── Negative tests ────────────────────────────────────────────────────────

    #[test]
    fn test_apply_before_learn_returns_error() {
        let normalizer = NyulUdupaNormalizer::new();
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let image = make_image_1d(data);

        let result = normalizer.apply(&image);
        assert!(
            result.is_err(),
            "apply before learn_standard must return Err"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("standard landmarks not learned"),
            "error message must mention missing training: {}",
            err_msg
        );
    }

    #[test]
    #[should_panic(expected = "at least one training image required")]
    fn test_learn_standard_empty_images_panics() {
        let mut normalizer = NyulUdupaNormalizer::new();
        let empty: Vec<&Image<TestBackend, 1>> = vec![];
        normalizer.learn_standard(&empty);
    }

    #[test]
    #[should_panic(expected = "at least 2 percentile landmarks required")]
    fn test_with_percentiles_too_few_panics() {
        let _ = NyulUdupaNormalizer::with_percentiles(vec![50.0]);
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn test_with_percentiles_not_ascending_panics() {
        let _ = NyulUdupaNormalizer::with_percentiles(vec![50.0, 30.0, 80.0]);
    }

    // ── Boundary tests ────────────────────────────────────────────────────────

    #[test]
    fn test_constant_image_maps_to_constant() {
        // Constant image: all landmarks are the same value.
        // After learning and applying, output should be constant.
        let data = vec![5.0f32; 100];
        let image = make_image_1d(data);

        let mut normalizer = NyulUdupaNormalizer::new();
        normalizer.learn_standard(&[&image]);

        let result = normalizer.apply(&image).expect("apply");
        let vals = get_values(&result);
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-4,
                "voxel {}: constant image must remain constant, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_default_percentiles() {
        let n = NyulUdupaNormalizer::new();
        assert_eq!(
            n.percentiles,
            vec![1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0]
        );
        assert!(n.standard_landmarks.is_none());
    }

    #[test]
    fn test_default_trait_matches_new() {
        let d = NyulUdupaNormalizer::default();
        let n = NyulUdupaNormalizer::new();
        assert_eq!(d.percentiles, n.percentiles);
        assert!(d.standard_landmarks.is_none());
    }
}
