//! Per-label intensity statistics over a labeled segmentation map.
//!
//! # Mathematical Specification
//!
//! Given a label map L and intensity image I with identical shape [Z, Y, X]:
//!
//! For each non-zero label k in {1, 2, ..., K}:
//! - V_k = { I(x) : L(x) = k }
//! - count_k  = |V_k|
//! - min_k    = min(V_k)
//! - max_k    = max(V_k)
//! - mean_k   = (1/count_k) * sum_{v in V_k} v
//! - std_k    = sqrt( (1/count_k) * sum_{v in V_k} (v - mean_k)^2 )  (population std)
//!
//! Background (label 0) is excluded from results.
//!
//! # Complexity
//! Single O(N) parallel fold/reduce pass over (label, intensity) pairs.
//! HashMap per thread accumulates (min, max, sum, sum_sq, count).
//!
//! # Reference
//! ITK LabelStatisticsImageFilter -- per-label min, max, mean, sigma, count.

use crate::image::Image;
use burn::tensor::backend::Backend;
use rayon::prelude::*;
use std::collections::HashMap;

/// Intensity statistics for a single label region.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelIntensityStatistics {
    /// Integer label index (>= 1); background (0) is excluded.
    pub label: u32,
    /// Number of voxels carrying this label.
    pub count: usize,
    /// Minimum intensity value within this label region.
    pub min: f32,
    /// Maximum intensity value within this label region.
    pub max: f32,
    /// Arithmetic mean intensity.
    pub mean: f32,
    /// Population standard deviation of intensity.
    pub std: f32,
}

/// Compute per-label intensity statistics.
///
/// For each non-zero label in `label_image`, accumulates `min`, `max`, `mean`,
/// and `std` over all voxels of `intensity_image` where `label_image` carries
/// that label.  Background voxels (label = 0) are excluded from results.
///
/// Returns a `Vec<LabelIntensityStatistics>` sorted by label index.
///
/// # Panics
/// Panics if `label_image` and `intensity_image` have different shapes.
pub fn compute_label_intensity_statistics<B: Backend>(
    label_image: &Image<B, 3>,
    intensity_image: &Image<B, 3>,
) -> Vec<LabelIntensityStatistics> {
    assert_eq!(
        label_image.shape(),
        intensity_image.shape(),
        "label_image and intensity_image must have identical shapes"
    );

    let label_data = label_image.data().clone().into_data();
    let label_slice = label_data.as_slice::<f32>().expect("f32 label tensor data");

    let intensity_data = intensity_image.data().clone().into_data();
    let intensity_slice = intensity_data
        .as_slice::<f32>()
        .expect("f32 intensity tensor data");

    compute_label_intensity_statistics_from_slices(label_slice, intensity_slice)
}

/// Compute per-label intensity statistics from pre-extracted flat slices.
///
/// Zero-copy variant: accepts slices directly.  Label values cast to `u32` via
/// truncation; background (label 0) is excluded.
///
/// Returns a `Vec<LabelIntensityStatistics>` sorted by label index.
///
/// # Panics
/// Panics if `label_slice` and `intensity_slice` have different lengths.
pub fn compute_label_intensity_statistics_from_slices(
    label_slice: &[f32],
    intensity_slice: &[f32],
) -> Vec<LabelIntensityStatistics> {
    assert_eq!(
        label_slice.len(),
        intensity_slice.len(),
        "label_slice and intensity_slice must have equal length"
    );

    // Accumulator: (min, max, sum_f64, sum_sq_f64, count)
    type Acc = (f32, f32, f64, f64, usize);

    let combined: HashMap<u32, Acc> = label_slice
        .par_iter()
        .zip(intensity_slice.par_iter())
        .fold(
            || HashMap::<u32, Acc>::new(),
            |mut acc, (&label_f, &intensity)| {
                let label = label_f as u32;
                if label == 0 {
                    return acc;
                }
                let entry = acc
                    .entry(label)
                    .or_insert((f32::INFINITY, f32::NEG_INFINITY, 0.0_f64, 0.0_f64, 0_usize));
                let v_f64 = intensity as f64;
                entry.0 = entry.0.min(intensity);
                entry.1 = entry.1.max(intensity);
                entry.2 += v_f64;
                entry.3 += v_f64 * v_f64;
                entry.4 += 1;
                acc
            },
        )
        .reduce(
            || HashMap::<u32, Acc>::new(),
            |mut a, b| {
                for (k, (bmin, bmax, bsum, bsumsq, bcnt)) in b {
                    let e = a
                        .entry(k)
                        .or_insert((f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0, 0));
                    e.0 = e.0.min(bmin);
                    e.1 = e.1.max(bmax);
                    e.2 += bsum;
                    e.3 += bsumsq;
                    e.4 += bcnt;
                }
                a
            },
        );

    let mut result: Vec<LabelIntensityStatistics> = combined
        .into_iter()
        .map(|(label, (min, max, sum, sum_sq, count))| {
            let n = count as f64;
            let mean_f64 = sum / n;
            let mean = mean_f64 as f32;
            // Population variance: E[X^2] - E[X]^2; .max(0.0) absorbs f64 cancellation.
            let variance = ((sum_sq / n) - mean_f64 * mean_f64).max(0.0) as f32;
            let std = variance.sqrt();
            LabelIntensityStatistics { label, count, min, max, mean, std }
        })
        .collect();

    result.sort_by_key(|s| s.label);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data, Shape::new(dims)),
            &device,
        );
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    #[test]
    fn test_single_label_single_voxel() {
        // Label=1 at index 1, intensity 42.0; rest background.
        // Expected: count=1, min=max=mean=42.0, std=0.0
        let labels = vec![0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let intensities = vec![10.0_f32, 42.0, 5.0, 3.0, 7.0, 1.0, 9.0, 2.0];
        let stats = compute_label_intensity_statistics_from_slices(&labels, &intensities);

        assert_eq!(stats.len(), 1, "exactly one label");
        let s = &stats[0];
        assert_eq!(s.label, 1);
        assert_eq!(s.count, 1);
        assert_eq!(s.min, 42.0);
        assert_eq!(s.max, 42.0);
        assert!((s.mean - 42.0).abs() < 1e-6, "mean={}", s.mean);
        assert!(s.std < 1e-6, "std must be 0, got {}", s.std);
    }

    #[test]
    fn test_single_label_known_statistics() {
        // n=4, values [1,2,3,4]: mean=2.5, variance=1.25, std=sqrt(1.25)
        let labels = vec![1.0_f32, 1.0, 1.0, 1.0, 0.0, 0.0];
        let intensities = vec![1.0_f32, 2.0, 3.0, 4.0, 99.0, 99.0];
        let stats = compute_label_intensity_statistics_from_slices(&labels, &intensities);

        assert_eq!(stats.len(), 1);
        let s = &stats[0];
        assert_eq!(s.label, 1);
        assert_eq!(s.count, 4);
        assert_eq!(s.min, 1.0, "min");
        assert_eq!(s.max, 4.0, "max");
        assert!((s.mean - 2.5).abs() < 1e-5, "mean={}", s.mean);
        assert!(
            (s.std - 1.25_f32.sqrt()).abs() < 1e-4,
            "std={} expected={}",
            s.std,
            1.25_f32.sqrt()
        );
    }

    #[test]
    fn test_two_labels_independent_statistics() {
        // Label=1: [10,20,30] mean=20, variance=200/3, std=sqrt(200/3)
        // Label=2: [5,15] mean=10, variance=25, std=5.0
        let labels = vec![1.0_f32, 1.0, 1.0, 2.0, 2.0];
        let intensities = vec![10.0_f32, 20.0, 30.0, 5.0, 15.0];
        let stats = compute_label_intensity_statistics_from_slices(&labels, &intensities);

        assert_eq!(stats.len(), 2, "two labels");
        let s1 = &stats[0];
        assert_eq!(s1.label, 1);
        assert_eq!(s1.count, 3);
        assert_eq!(s1.min, 10.0);
        assert_eq!(s1.max, 30.0);
        assert!((s1.mean - 20.0).abs() < 1e-5, "label1 mean={}", s1.mean);
        let exp1 = (200.0_f64 / 3.0).sqrt() as f32;
        assert!((s1.std - exp1).abs() < 1e-4, "label1 std={} expected={}", s1.std, exp1);
        let s2 = &stats[1];
        assert_eq!(s2.label, 2);
        assert_eq!(s2.count, 2);
        assert_eq!(s2.min, 5.0);
        assert_eq!(s2.max, 15.0);
        assert!((s2.mean - 10.0).abs() < 1e-5, "label2 mean={}", s2.mean);
        assert!((s2.std - 5.0_f32).abs() < 1e-4, "label2 std={}", s2.std);
    }

    #[test]
    fn test_background_label_zero_excluded() {
        // All labels=0: result must be empty.
        let labels = vec![0.0_f32; 10];
        let intensities: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let stats = compute_label_intensity_statistics_from_slices(&labels, &intensities);
        assert_eq!(stats.len(), 0, "all background -> empty result");
    }

    #[test]
    fn test_uniform_intensity_within_label() {
        // Uniform intensity -> std = 0.
        let labels: Vec<f32> = vec![1.0; 20];
        let intensities: Vec<f32> = vec![7.5; 20];
        let stats = compute_label_intensity_statistics_from_slices(&labels, &intensities);

        assert_eq!(stats.len(), 1);
        let s = &stats[0];
        assert_eq!(s.count, 20);
        assert_eq!(s.min, 7.5);
        assert_eq!(s.max, 7.5);
        assert!((s.mean - 7.5).abs() < 1e-6);
        assert!(s.std < 1e-6, "std must be 0 for uniform intensity");
    }

    #[test]
    fn test_compute_from_image_api_matches_slice_api() {
        // 2x2x2: label=1 at positions 0..4 with [1,2,3,4], label=2 at 4..8 with [10,20,30,40]
        let label_data = vec![1.0_f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let intensity_data = vec![1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let label_image = make_image_3d(label_data.clone(), [2, 2, 2]);
        let intensity_image = make_image_3d(intensity_data.clone(), [2, 2, 2]);
        let stats_img = compute_label_intensity_statistics(&label_image, &intensity_image);
        let stats_slice =
            compute_label_intensity_statistics_from_slices(&label_data, &intensity_data);
        assert_eq!(stats_img.len(), stats_slice.len());
        for (a, b) in stats_img.iter().zip(stats_slice.iter()) {
            assert_eq!(a.label, b.label);
            assert_eq!(a.count, b.count);
            assert_eq!(a.min, b.min);
            assert_eq!(a.max, b.max);
            assert!((a.mean - b.mean).abs() < 1e-6, "mean mismatch");
            assert!((a.std - b.std).abs() < 1e-6, "std mismatch");
        }
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn test_length_mismatch_panics() {
        let _ = compute_label_intensity_statistics_from_slices(
            &vec![1.0_f32; 4],
            &vec![1.0_f32; 5],
        );
    }

    #[test]
    #[should_panic(expected = "identical shapes")]
    fn test_shape_mismatch_panics() {
        let _ = compute_label_intensity_statistics(
            &make_image_3d(vec![1.0; 8], [2, 2, 2]),
            &make_image_3d(vec![1.0; 12], [2, 2, 3]),
        );
    }

    #[test]
    fn test_results_sorted_by_label() {
        // Labels inserted out of order: 3, 1, 2 -> sorted output must be 1, 2, 3.
        let labels = vec![3.0_f32, 1.0, 2.0, 3.0, 1.0];
        let intensities = vec![30.0_f32, 10.0, 20.0, 30.0, 10.0];
        let stats = compute_label_intensity_statistics_from_slices(&labels, &intensities);
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].label, 1, "first must be label 1");
        assert_eq!(stats[1].label, 2, "second must be label 2");
        assert_eq!(stats[2].label, 3, "third must be label 3");
    }
}
