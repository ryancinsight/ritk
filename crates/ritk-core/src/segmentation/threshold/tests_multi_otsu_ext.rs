//! Extracted tests: general invariants, edge cases, internal variance, negative, from_slice.
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
fn test_apply_preserves_spatial_metadata_3d() {
    let device: <TestBackend as Backend>::Device = Default::default();
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
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(data, Shape::new([3, 3, 3])),
        &device,
    );
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.5, 0.5]);
    let direction = Direction::identity();
    let image: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);
    let labels = MultiOtsuThreshold::new(3).apply(&image);
    assert_eq!(labels.origin(), &origin, "origin must be preserved");
    assert_eq!(labels.spacing(), &spacing, "spacing must be preserved");
    assert_eq!(labels.direction(), &direction, "direction must be preserved");
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
    let result_data = labels.data().clone().into_data();
    let values = result_data.as_slice::<f32>().unwrap();
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
    assert_eq!(mot.num_classes, 3, "default num_classes must be 3");
    assert_eq!(mot.num_bins, 256, "default num_bins must be 256");
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
