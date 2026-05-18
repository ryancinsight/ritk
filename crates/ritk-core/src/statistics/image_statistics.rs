//! Image intensity statistics.
//!
//! Computes descriptive statistics over image intensities, either for the
//! full image or restricted to a foreground mask.
//!
//! # Mathematical Specification
//! Given a vector of intensity values V = {v₁, …, vₙ}:
//! - min     = min(V)
//! - max     = max(V)
//! - mean    = (1/n) · Σ vᵢ
//! - std     = √( (1/n) · Σ (vᵢ − mean)² )   (population std)
//! - p25     = V_sorted[⌊n/4⌋]
//! - p50     = V_sorted[⌊n/2⌋]
//! - p75     = V_sorted[⌊3n/4⌋]

use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Descriptive statistics over image intensities.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageStatistics {
    /// Minimum intensity value.
    pub min: f32,
    /// Maximum intensity value.
    pub max: f32,
    /// Arithmetic mean intensity.
    pub mean: f32,
    /// Population standard deviation.
    pub std: f32,
    /// Percentiles: \[p25, p50, p75\].
    pub percentiles: [f32; 3],
}

/// Compute statistics over **all** voxels in `image`.
///
/// Extraction path: `tensor.clone().into_data()` → `as_slice::<f32>()` → CPU arithmetic.
pub fn compute_statistics<B: Backend, const D: usize>(image: &Image<B, D>) -> ImageStatistics {
    let (vals, _) = extract_vec_infallible(image);
    let slice: &[f32] = &vals;
    compute_statistics_from_slice(slice)
}

/// Compute statistics from an immutable slice.
///
/// This is the zero-domain-logic public helper for callers that already have
/// borrowed f32 tensor storage. It clones the values once because percentile
/// computation sorts in-place.
pub fn compute_statistics_from_slice(slice: &[f32]) -> ImageStatistics {
    let values: Vec<f32> = slice.to_vec();
    compute_from_values(&values)
}

/// Compute statistics restricted to voxels where `mask` > 0.5 (foreground).
///
/// `mask` must have the same shape as `image` and contain 0.0 (background) or
/// 1.0 (foreground). Panics if the shapes differ or no foreground voxels exist.
pub fn masked_statistics<B: Backend, const D: usize>(
    image: &Image<B, D>,
    mask: &Image<B, D>,
) -> ImageStatistics {
    let (img_vals, _) = extract_vec_infallible(image);
    let image_slice: &[f32] = &img_vals;
    let (mask_vals, _) = extract_vec_infallible(mask);
    let mask_slice: &[f32] = &mask_vals;

    assert_eq!(
        image_slice.len(),
        mask_slice.len(),
        "image and mask must have identical element count"
    );

    let values: Vec<f32> = image_slice
        .iter()
        .zip(mask_slice.iter())
        .filter(|(_, &m)| m > 0.5)
        .map(|(&v, _)| v)
        .collect();

    assert!(!values.is_empty(), "mask contains no foreground voxels");
    compute_from_values(&values)
}

/// Core statistics computation.
///
/// # Invariants
/// - `values` is non-empty (caller enforced).
/// - Sorts `values` in-place; NaN propagates arithmetic and is ordered last by
///   the `partial_cmp` fallback.
///
/// # Precision
/// Mean and variance accumulate in f64 to avoid catastrophic f32 cancellation
/// for large arrays (n > ~10^7).  Sequential f32 summation of n ≈ 10^8 values
/// with mean ≈ −789 produces a running sum of ~−85 billion; at that scale the
/// f32 ULP (≈8192) exceeds individual element magnitudes, so additions are
/// rounded to zero and the sum saturates.  Two-pass f64 accumulation is the
/// algorithm's numerical contract requirement, not a convenience cast.
pub fn compute_from_values(values: &[f32]) -> ImageStatistics {
    let mut sorted_values = values.to_vec();
    let values = sorted_values.as_mut_slice();
    let n = values.len();
    debug_assert!(n > 0, "compute_from_values requires non-empty input");

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = values[0];
    let max = values[n - 1];

    // Accumulate in f64: sequential f32 sum saturates for n > ~10^7 elements
    // when the running total exceeds the f32 ULP of individual values.
    let sum_f64: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
    let mean_f64: f64 = sum_f64 / n as f64;
    let mean: f32 = mean_f64 as f32;

    // Two-pass f64 variance: squared deviations sum to ~10^13 for CT-scale
    // data, exceeding f32 representable precision per element at n > ~10^7.
    let var_f64: f64 = values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean_f64;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let std = var_f64.sqrt() as f32;

    // Floor-division percentile indices as specified in the module contract.
    let p25 = values[n / 4];
    let p50 = values[n / 2];
    let p75 = values[(3 * n) / 4];

    ImageStatistics {
        min,
        max,
        mean,
        std,
        percentiles: [p25, p50, p75],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
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

    fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    // ── Positive tests ────────────────────────────────────────────────────────

    #[test]
    fn test_uniform_image() {
        // All voxels = 5.0 → std = 0, all percentiles = 5.
        let image = make_image_3d(vec![5.0f32; 27], [3, 3, 3]);
        let s = compute_statistics(&image);

        assert_eq!(s.min, 5.0);
        assert_eq!(s.max, 5.0);
        assert!((s.mean - 5.0).abs() < 1e-6, "mean={}", s.mean);
        assert!(s.std < 1e-6, "std={}", s.std);
        assert_eq!(s.percentiles, [5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_known_sequence() {
        // Values [1,2,3,4,5,6,7,8] (n=8):
        //   mean     = 36/8 = 4.5
        //   variance = 42/8 = 5.25  →  std = √5.25 ≈ 2.2913
        //   p25 = values[8/4]   = values[2] = 3.0
        //   p50 = values[8/2]   = values[4] = 5.0
        //   p75 = values[24/4]  = values[6] = 7.0
        let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
        let image = make_image_1d(data);
        let s = compute_statistics(&image);

        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 8.0);
        assert!((s.mean - 4.5).abs() < 1e-5, "mean={}", s.mean);
        assert!(
            (s.std - 5.25f32.sqrt()).abs() < 1e-4,
            "std={} expected={}",
            s.std,
            5.25f32.sqrt()
        );
        assert_eq!(s.percentiles[0], 3.0, "p25");
        assert_eq!(s.percentiles[1], 5.0, "p50");
        assert_eq!(s.percentiles[2], 7.0, "p75");
    }

    #[test]
    fn test_single_voxel() {
        // n=1: all statistics collapse to the single value.
        let image = make_image_1d(vec![42.0]);
        let s = compute_statistics(&image);

        assert_eq!(s.min, 42.0);
        assert_eq!(s.max, 42.0);
        assert!((s.mean - 42.0).abs() < 1e-6);
        assert!(s.std < 1e-6, "std must be 0 for single voxel");
        assert_eq!(s.percentiles, [42.0, 42.0, 42.0]);
    }

    #[test]
    fn test_two_values() {
        // n=2, values=[1,2]:
        //   mean = 1.5, variance = 0.25, std = 0.5
        //   p25 = values[2/4=0] = 1.0
        //   p50 = values[2/2=1] = 2.0
        //   p75 = values[6/4=1] = 2.0
        let image = make_image_1d(vec![1.0, 2.0]);
        let s = compute_statistics(&image);

        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 2.0);
        assert!((s.mean - 1.5).abs() < 1e-5);
        assert!((s.std - 0.5).abs() < 1e-5, "std={}", s.std);
        assert_eq!(s.percentiles[0], 1.0, "p25");
        assert_eq!(s.percentiles[1], 2.0, "p50");
        assert_eq!(s.percentiles[2], 2.0, "p75");
    }

    #[test]
    fn test_reverse_order_input_matches_sorted() {
        // Sort order of input must not change the result.
        let sorted = make_image_1d(vec![1.0, 2.0, 3.0, 4.0]);
        let reversed = make_image_1d(vec![4.0, 3.0, 2.0, 1.0]);

        let s_sorted = compute_statistics(&sorted);
        let s_reversed = compute_statistics(&reversed);

        assert_eq!(s_sorted.min, s_reversed.min);
        assert_eq!(s_sorted.max, s_reversed.max);
        assert!((s_sorted.mean - s_reversed.mean).abs() < 1e-6);
        assert!((s_sorted.std - s_reversed.std).abs() < 1e-6);
        assert_eq!(s_sorted.percentiles, s_reversed.percentiles);
    }

    // ── Masked statistics ─────────────────────────────────────────────────────

    #[test]
    fn test_masked_statistics_subset() {
        // Values [1..8]; mask foreground at indices 2..=5 → [3,4,5,6] (n=4).
        //   mean     = 4.5
        //   variance = 5/4 = 1.25  →  std = √1.25 ≈ 1.1180
        //   p25 = values[4/4=1] = 4.0
        //   p50 = values[4/2=2] = 5.0
        //   p75 = values[12/4=3] = 6.0
        let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
        let mut mask_data = vec![0.0f32; 8];
        for v in mask_data.iter_mut().take(6).skip(2) {
            *v = 1.0;
        }

        let image = make_image_1d(data);
        let mask = make_image_1d(mask_data);
        let s = masked_statistics(&image, &mask);

        assert_eq!(s.min, 3.0);
        assert_eq!(s.max, 6.0);
        assert!((s.mean - 4.5).abs() < 1e-5, "mean={}", s.mean);
        assert!(
            (s.std - 1.25f32.sqrt()).abs() < 1e-4,
            "std={} expected={}",
            s.std,
            1.25f32.sqrt()
        );
        assert_eq!(s.percentiles[0], 4.0, "p25");
        assert_eq!(s.percentiles[1], 5.0, "p50");
        assert_eq!(s.percentiles[2], 6.0, "p75");
    }

    #[test]
    fn test_masked_statistics_all_foreground_matches_full() {
        // mask = all ones → identical result to compute_statistics.
        let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
        let mask_data = vec![1.0f32; 8];

        let image = make_image_1d(data);
        let mask = make_image_1d(mask_data);

        let s_full = compute_statistics(&image);
        let s_masked = masked_statistics(&image, &mask);

        assert_eq!(s_full.min, s_masked.min);
        assert_eq!(s_full.max, s_masked.max);
        assert!((s_full.mean - s_masked.mean).abs() < 1e-6);
        assert!((s_full.std - s_masked.std).abs() < 1e-6);
        assert_eq!(s_full.percentiles, s_masked.percentiles);
    }

    #[test]
    fn test_masked_statistics_single_foreground_voxel() {
        // Only one foreground voxel → std = 0, all percentiles = that value.
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let mut mask_data = vec![0.0f32; 4];
        mask_data[2] = 1.0; // foreground is value 30.0

        let image = make_image_1d(data);
        let mask = make_image_1d(mask_data);
        let s = masked_statistics(&image, &mask);

        assert_eq!(s.min, 30.0);
        assert_eq!(s.max, 30.0);
        assert!((s.mean - 30.0).abs() < 1e-6);
        assert!(s.std < 1e-6);
        assert_eq!(s.percentiles, [30.0, 30.0, 30.0]);
    }

    // ── Negative / boundary ───────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "mask contains no foreground voxels")]
    fn test_masked_statistics_empty_mask_panics() {
        let image = make_image_1d(vec![1.0, 2.0, 3.0]);
        let mask = make_image_1d(vec![0.0, 0.0, 0.0]);
        let _ = masked_statistics(&image, &mask);
    }

    #[test]
    #[should_panic(expected = "identical element count")]
    fn test_masked_statistics_shape_mismatch_panics() {
        let image = make_image_1d(vec![1.0, 2.0, 3.0]);
        let mask = make_image_1d(vec![1.0, 1.0]);
        let _ = masked_statistics(&image, &mask);
    }

    // ── Large-N f64-accumulation precision ───────────────────────────────────
    //
    // Sequential f32 summation of n ≈ 10^7 elements with CT-scale values
    // (-2048..=3071, mean ≈ -789) produces a running total of ~-85 billion.
    // At that magnitude the f32 ULP (≈8192) exceeds typical per-element values
    // so additions round to zero; the accumulated sum saturates.  The f64
    // accumulator path must return a mean within ε = 1.0 HU of the f64 reference.

    #[test]
    fn test_large_n_ct_scale_mean_precision() {
        // n = 10,485,760 (10 × 2^20) elements spanning a CT-like range.
        // Pattern: floor(i * 5120 / n) − 2048 produces values uniformly spaced
        // in [−2048, 3071] with mean = 511.5.
        // Analytical mean: (−2048 + 3071) / 2 = 511.5
        let n: usize = 10_485_760;
        let scale = 5120_f64;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f64 * scale / n as f64).floor() as f32) - 2048.0)
            .collect();

        let s = compute_statistics_from_slice(&data);

        let expected_mean = 511.5_f32;
        assert!(
            (s.mean - expected_mean).abs() < 1.0,
            "large-N mean={} expected≈{} (f64 accumulation required for precision)",
            s.mean,
            expected_mean
        );
        assert_eq!(s.min, -2048.0, "min");
        assert_eq!(s.max, 3071.0, "max");
    }

    #[test]
    fn test_large_n_negative_mean_precision() {
        // n = 10,485,760 elements all equal to a large negative CT value (−789).
        // Sequential f32 sum saturates at n ≈ 17M; at n = 10.5M the error is
        // ~200 HU without f64 accumulation and <0.01 HU with it.
        let n: usize = 10_485_760;
        let constant = -789.0_f32;
        let data = vec![constant; n];

        let s = compute_statistics_from_slice(&data);

        assert!(
            (s.mean - constant).abs() < 1.0,
            "large-N constant mean={} expected={} (precision lost without f64 accumulation)",
            s.mean,
            constant
        );
        assert_eq!(s.min, constant, "min");
        assert_eq!(s.max, constant, "max");
        assert!(
            s.std < 1e-3,
            "std of constant array must be ~0, got {}",
            s.std
        );
    }
}
