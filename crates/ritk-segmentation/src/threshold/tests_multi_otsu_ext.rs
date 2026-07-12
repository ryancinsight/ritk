//! Extracted tests: general invariants, edge cases, internal variance, negative, from_slice.
use super::*;
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support::{make_image, make_image_with};
use ritk_tensor_ops::extract_vec_infallible;

type TestBackend = NdArray<f32>;

fn assert_native_multi_otsu_conformance<const D: usize>(
    values: Vec<f32>,
    dimensions: [usize; D],
) -> (Vec<f32>, Vec<f32>) {
    let origin = Point::new([2.0; D]);
    let spacing = Spacing::new([0.5; D]);
    let direction = Direction::identity();
    let native = NativeImage::from_flat_on(
        values.clone(),
        dimensions,
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let legacy = make_image::<TestBackend, D>(values, dimensions);
    let filter = MultiOtsuThreshold::new(3);

    let native_thresholds = filter
        .compute_native(&native)
        .expect("native Multi-Otsu computation succeeds");
    let (native_labels, fused_thresholds) = filter
        .apply_native_with_thresholds(&native, &SequentialBackend)
        .expect("native Multi-Otsu application succeeds");
    let native_labels_only = filter
        .apply_native(&native, &SequentialBackend)
        .expect("native Multi-Otsu labels succeed");
    let legacy_thresholds = filter.compute(&legacy);
    let legacy_labels = filter.apply(&legacy);

    assert_eq!(native_thresholds, legacy_thresholds);
    assert_eq!(fused_thresholds, legacy_thresholds);
    assert_eq!(native_labels.shape(), dimensions);
    assert_eq!(*native_labels.origin(), origin);
    assert_eq!(*native_labels.spacing(), spacing);
    assert_eq!(*native_labels.direction(), direction);
    assert_eq!(
        native_labels.data_slice().expect("contiguous labels"),
        legacy_labels.data_slice().as_ref()
    );
    assert_eq!(
        native_labels_only.data_slice().expect("contiguous labels"),
        native_labels.data_slice().expect("contiguous labels")
    );
    (
        native_thresholds,
        native_labels
            .data_slice()
            .expect("contiguous labels")
            .to_vec(),
    )
}

#[test]
fn native_multi_otsu_conforms_across_dimensions_and_special_values() {
    assert_eq!(
        compute_multi_otsu_thresholds_from_slice(&[], 3, 256),
        [0.0, 0.0]
    );
    let _ = assert_native_multi_otsu_conformance(vec![0.0, 1.0, 2.0, 3.0], [4]);
    let _ = assert_native_multi_otsu_conformance(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3]);
    let _ = assert_native_multi_otsu_conformance(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1, 2, 3]);

    let (constant_thresholds, constant_labels) =
        assert_native_multi_otsu_conformance(vec![5.0; 8], [2, 2, 2]);
    assert_eq!(constant_thresholds, [5.0, 5.0]);
    assert_eq!(constant_labels, [2.0; 8]);

    let (_, mixed_labels) = assert_native_multi_otsu_conformance(
        vec![f32::NAN, f32::NEG_INFINITY, 1.0, 2.0, f32::INFINITY, 3.0],
        [1, 2, 3],
    );
    assert_eq!(mixed_labels[0], 0.0);
    assert_eq!(mixed_labels[1], 0.0);
    assert_eq!(mixed_labels[4], 0.0);

    let (nonfinite_thresholds, nonfinite_labels) = assert_native_multi_otsu_conformance(
        vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
        [1, 1, 3],
    );
    assert_eq!(nonfinite_thresholds, [0.0, 0.0]);
    assert_eq!(nonfinite_labels, [0.0; 3]);
}

#[test]
#[should_panic(expected = "num_bins must be ≥ 2")]
fn slice_thresholds_reject_invalid_bin_count() {
    let _ = compute_multi_otsu_thresholds_from_slice(&[0.0, 1.0], 2, 1);
}

fn assert_class_bin_boundary(num_classes: usize, num_bins: usize) {
    let values: Vec<f32> = (0..num_bins).map(|value| value as f32).collect();
    let thresholds = compute_multi_otsu_thresholds_from_slice(&values, num_classes, num_bins);
    let labels = apply_multi_otsu_to_slice(&values, num_classes, num_bins);

    assert_eq!(thresholds.len(), num_classes - 1);
    assert!(thresholds.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(labels
        .iter()
        .all(|&label| label >= 0.0 && label < num_classes as f32));
}

#[test]
fn coupled_class_bin_boundaries_are_exact_and_bounded() {
    assert_class_bin_boundary(2, 2);
    assert_class_bin_boundary(8, 8);
    assert_class_bin_boundary(256, 256);
}

#[test]
fn dynamic_program_matches_exhaustive_three_class_oracle() {
    let histogram = [1_u32, 3, 2, 7, 4, 1];
    let total = histogram.iter().map(|&count| count as f64).sum::<f64>();
    let probabilities: Vec<f64> = histogram
        .iter()
        .map(|&count| count as f64 / total)
        .collect();
    let mut prefix_h = vec![0.0; histogram.len() + 1];
    let mut prefix_m = vec![0.0; histogram.len() + 1];
    for (index, probability) in probabilities.iter().copied().enumerate() {
        prefix_h[index + 1] = prefix_h[index] + probability;
        prefix_m[index + 1] = prefix_m[index] + index as f64 * probability;
    }
    let total_mu = prefix_m[histogram.len()];
    let mut exhaustive = (f64::NEG_INFINITY, [0_usize; 2]);
    for first in 1..histogram.len() - 1 {
        for second in first + 1..histogram.len() {
            let score = between_class_variance(
                &[first, second],
                &prefix_h,
                &prefix_m,
                total_mu,
                histogram.len(),
            );
            if score > exhaustive.0 {
                exhaustive = (score, [first, second]);
            }
        }
    }

    assert_eq!(
        optimal_threshold_bins(3, &prefix_h, &prefix_m, total_mu, histogram.len()),
        exhaustive.1
    );
}

#[test]
#[should_panic(expected = "num_classes must not exceed num_bins")]
fn constructor_rejects_more_classes_than_bins() {
    let _ = MultiOtsuThreshold::with_bins(9, 8);
}

#[test]
#[should_panic(expected = "num_classes must not exceed num_bins")]
fn default_bin_constructor_rejects_more_than_256_classes() {
    let _ = MultiOtsuThreshold::new(257);
}

#[test]
#[should_panic(expected = "num_classes must not exceed num_bins")]
fn slice_thresholds_reject_more_classes_than_bins() {
    let _ = compute_multi_otsu_thresholds_from_slice(&[0.0, 1.0], 3, 2);
}

fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
    let n = data.len();
    make_image(data, [n])
}

fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    make_image(data, dims)
}

fn get_values_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

// ── General invariants ─────────────────────────────────────────────────────

#[test]
fn test_threshold_count_equals_k_minus_1_for_k_2_3_4() {
    let data: Vec<f32> = (0u8..=200).map(|x| x as f32).collect();
    let image = make_image_1d(data);
    for k in 2usize..=4 {
        let thresholds = MultiOtsuThreshold::new(k).compute(&image);
        assert_eq!(
            thresholds.len(),
            k - 1,
            "K={} must return {} thresholds, got {}",
            k,
            k - 1,
            thresholds.len()
        );
    }
}

#[test]
fn test_thresholds_are_strictly_increasing() {
    let mut data = vec![20.0f32; 50];
    data.extend(vec![120.0f32; 50]);
    data.extend(vec![220.0f32; 50]);
    let image = make_image_1d(data);
    let thresholds = MultiOtsuThreshold::new(3).compute(&image);
    for i in 0..thresholds.len().saturating_sub(1) {
        assert!(
            thresholds[i] < thresholds[i + 1],
            "thresholds must be strictly increasing: t[{}]={} >= t[{}]={}",
            i,
            thresholds[i],
            i + 1,
            thresholds[i + 1]
        );
    }
}

#[test]
fn test_thresholds_within_intensity_range() {
    let data: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();
    let image = make_image_1d(data);
    let thresholds = MultiOtsuThreshold::new(3).compute(&image);
    for &t in &thresholds {
        assert!(t >= 0.0, "threshold must be ≥ x_min (0.0), got {}", t);
        assert!(t <= 100.0, "threshold must be ≤ x_max (100.0), got {}", t);
    }
}

#[test]
fn test_computation_is_deterministic() {
    let data: Vec<f32> = (0u8..=200).map(|x| x as f32).collect();
    let image = make_image_1d(data);
    let t1 = MultiOtsuThreshold::new(3).compute(&image);
    let t2 = MultiOtsuThreshold::new(3).compute(&image);
    assert_eq!(t1, t2, "threshold computation must be deterministic");
}

#[test]
fn test_apply_preserves_spatial_metadata_volumetric() {
    let data: Vec<f32> = (0u8..27)
        .map(|x| {
            if x < 9 {
                10.0
            } else if x < 18 {
                128.0
            } else {
                240.0
            }
        })
        .collect();
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.5, 0.5]);
    let direction = Direction::<3>::identity();
    let image: Image<TestBackend, 3> =
        make_image_with(data, [3, 3, 3], Some(origin), Some(spacing), None);
    let labels = MultiOtsuThreshold::new(3).apply(&image);
    assert_eq!(labels.origin(), &origin, "origin must be preserved");
    assert_eq!(labels.spacing(), &spacing, "spacing must be preserved");
    assert_eq!(
        labels.direction(),
        &direction,
        "direction must be preserved"
    );
    assert_eq!(labels.shape(), [3, 3, 3], "shape must be preserved");
}

#[test]
fn test_apply_3d_trimodal_labels_correct() {
    let data: Vec<f32> = (0u8..27)
        .map(|x| {
            if x < 9 {
                10.0
            } else if x < 18 {
                128.0
            } else {
                250.0
            }
        })
        .collect();
    let image = make_image_3d(data, [3, 3, 3]);
    let labels = MultiOtsuThreshold::new(3).apply(&image);
    let (values, _) = extract_vec_infallible(&labels);
    for (i, &v) in values[..9].iter().enumerate() {
        assert_eq!(v, 0.0, "voxel {} (10.0) must have label 0, got {}", i, v);
    }
    for (i, &v) in values[9..18].iter().enumerate() {
        assert_eq!(v, 1.0, "voxel {} (128.0) must have label 1, got {}", i, v);
    }
    for (i, &v) in values[18..].iter().enumerate() {
        assert_eq!(v, 2.0, "voxel {} (250.0) must have label 2, got {}", i, v);
    }
}

#[test]
fn test_apply_label_ordering_monotone_for_monotone_input() {
    let data: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(3).apply(&image);
    let values = get_values_1d(&labels);
    for i in 0..values.len().saturating_sub(1) {
        assert!(
            values[i] <= values[i + 1],
            "labels must be non-decreasing for monotone input: values[{}]={} > values[{}]={}",
            i,
            values[i],
            i + 1,
            values[i + 1]
        );
    }
}

#[test]
fn test_convenience_fn_matches_struct_compute() {
    let data: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();
    let image = make_image_1d(data);
    let t_fn = multi_otsu_threshold(&image, 3);
    let t_struct = MultiOtsuThreshold::new(3).compute(&image);
    assert_eq!(
        t_fn, t_struct,
        "convenience fn and struct::compute must produce identical results"
    );
}

// ── Edge / boundary cases ──────────────────────────────────────────────────

#[test]
fn test_constant_image_all_thresholds_equal_constant_value() {
    let image = make_image_1d(vec![42.0f32; 64]);
    let thresholds = MultiOtsuThreshold::new(3).compute(&image);
    for &t in &thresholds {
        assert_eq!(t, 42.0, "constant image → threshold = 42.0, got {}", t);
    }
}

#[test]
fn test_single_voxel_returns_single_threshold_at_pixel_value() {
    let image = make_image_1d(vec![7.0f32]);
    let thresholds = MultiOtsuThreshold::new(2).compute(&image);
    assert_eq!(thresholds.len(), 1);
    assert_eq!(
        thresholds[0], 7.0,
        "single-voxel threshold must equal the pixel value"
    );
}

#[test]
fn test_custom_bin_count_still_separates_modes() {
    let mut data = vec![30.0f32; 50];
    data.extend(vec![220.0f32; 50]);
    let image = make_image_1d(data);
    let thresholds = MultiOtsuThreshold::with_bins(2, 64).compute(&image);
    assert_eq!(thresholds.len(), 1);
    assert!(
        thresholds[0] > 30.0 && thresholds[0] < 220.0,
        "64-bin threshold must lie between modes, got {}",
        thresholds[0]
    );
}

#[test]
fn test_default_is_3_classes_256_bins() {
    let mot = MultiOtsuThreshold::default();
    assert_eq!(mot.num_classes(), 3, "default num_classes must be 3");
    assert_eq!(mot.num_bins(), 256, "default num_bins must be 256");
}

// ── Internal: between_class_variance known values ─────────────────────────
//
// Uniform histogram over 4 bins: h = [0.25, 0.25, 0.25, 0.25].
// prefix_h = [0.00, 0.25, 0.50, 0.75, 1.00]
// prefix_m = [0.00, 0.00, 0.25, 0.75, 1.50]  (M[t+1] = M[t] + t·h[t])
// total_mu = 1.50
//
// At t = 2: Class 1 [0,1]: P = 0.5, μ = 0.5; Class 2 [2,3]: P = 0.5, μ = 2.5
//   σ²_B = 0.5·(0.5−1.5)² + 0.5·(2.5−1.5)² = 0.5 + 0.5 = 1.0
//
// At t = 1: Class 1 [0]: P = 0.25, μ = 0.0; Class 2 [1,3]: P = 0.75, μ = 2.0
//   σ²_B = 0.25·(0−1.5)² + 0.75·(2−1.5)² = 0.5625 + 0.1875 = 0.75

#[test]
fn test_between_class_variance_symmetric_split_uniform() {
    let prefix_h = vec![0.0, 0.25, 0.50, 0.75, 1.00];
    let prefix_m = vec![0.0, 0.00, 0.25, 0.75, 1.50];
    let total_mu = 1.50;
    let num_bins = 4;
    let sigma2_t2 = between_class_variance(&[2], &prefix_h, &prefix_m, total_mu, num_bins);
    assert!(
        (sigma2_t2 - 1.0).abs() < 1e-9,
        "σ²_B(t=2) must equal 1.0, got {}",
        sigma2_t2
    );
}

#[test]
fn test_between_class_variance_asymmetric_split_uniform() {
    let prefix_h = vec![0.0, 0.25, 0.50, 0.75, 1.00];
    let prefix_m = vec![0.0, 0.00, 0.25, 0.75, 1.50];
    let total_mu = 1.50;
    let num_bins = 4;
    let sigma2_t1 = between_class_variance(&[1], &prefix_h, &prefix_m, total_mu, num_bins);
    assert!(
        (sigma2_t1 - 0.75).abs() < 1e-9,
        "σ²_B(t=1) must equal 0.75, got {}",
        sigma2_t1
    );
}

#[test]
fn test_between_class_variance_symmetric_exceeds_asymmetric() {
    let prefix_h = vec![0.0, 0.25, 0.50, 0.75, 1.00];
    let prefix_m = vec![0.0, 0.00, 0.25, 0.75, 1.50];
    let total_mu = 1.50;
    let num_bins = 4;
    let s2 = between_class_variance(&[2], &prefix_h, &prefix_m, total_mu, num_bins);
    let s1 = between_class_variance(&[1], &prefix_h, &prefix_m, total_mu, num_bins);
    assert!(
        s2 > s1,
        "symmetric split (t=2, σ²={}) must exceed asymmetric (t=1, σ²={})",
        s2,
        s1
    );
}

#[test]
fn test_between_class_variance_k3_three_equal_classes() {
    // Class 1 [0]: P=0.25, μ=0.0; Class 2 [1]: P=0.25, μ=1.0; Class 3 [2,3]: P=0.50, μ=2.5
    // σ²_B = 0.25·(0−1.5)² + 0.25·(1−1.5)² + 0.50·(2.5−1.5)² = 0.5625 + 0.0625 + 0.5 = 1.125
    let prefix_h = vec![0.0, 0.25, 0.50, 0.75, 1.00];
    let prefix_m = vec![0.0, 0.00, 0.25, 0.75, 1.50];
    let total_mu = 1.50;
    let num_bins = 4;
    let sigma2 = between_class_variance(&[1, 2], &prefix_h, &prefix_m, total_mu, num_bins);
    let expected = 0.25 * (0.0 - 1.5_f64).powi(2)
        + 0.25 * (1.0 - 1.5_f64).powi(2)
        + 0.50 * (2.5 - 1.5_f64).powi(2);
    assert!(
        (sigma2 - expected).abs() < 1e-9,
        "K=3 σ²_B([1,2]) must equal {}, got {}",
        expected,
        sigma2
    );
}

// ── Negative tests ─────────────────────────────────────────────────────────

#[test]
#[should_panic(expected = "num_classes must be ≥ 2")]
fn test_new_with_k1_panics() {
    let _ = MultiOtsuThreshold::new(1);
}

#[test]
#[should_panic(expected = "num_classes must be ≥ 2")]
fn test_convenience_fn_k1_panics() {
    let data: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();
    let image = make_image_1d(data);
    let _ = multi_otsu_threshold(&image, 1);
}

#[test]
#[should_panic(expected = "num_bins must be ≥ 2")]
fn test_with_bins_zero_panics() {
    let _ = MultiOtsuThreshold::with_bins(2, 0);
}

// ── from_slice parity ──────────────────────────────────────────────────────

#[test]
fn test_compute_multi_otsu_from_slice_matches_filter() {
    let mut data = vec![20.0_f32; 100];
    data.extend(vec![120.0_f32; 100]);
    data.extend(vec![220.0_f32; 100]);
    let image = make_image_1d(data.clone());
    let t_filter = MultiOtsuThreshold::new(3).compute(&image);
    let t_slice = compute_multi_otsu_thresholds_from_slice(&data, 3, 256);
    assert_eq!(
        t_filter, t_slice,
        "from_slice must match filter: filter={:?} slice={:?}",
        t_filter, t_slice
    );
}
