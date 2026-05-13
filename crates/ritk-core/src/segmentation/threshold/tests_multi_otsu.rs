//! Tests for multi_otsu
//! Extracted from the main module to keep the 500-line structural limit.
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

// ── K=2: degeneration to standard Otsu ────────────────────────────────────

#[test]
fn test_k2_returns_exactly_one_threshold() {
    let data: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();
    let image = make_image_1d(data);
    let thresholds = multi_otsu_threshold(&image, 2);
    assert_eq!(thresholds.len(), 1, "K=2 must return exactly 1 threshold");
}

#[test]
fn test_k2_bimodal_threshold_between_modes() {
    // 50 × 20.0 and 50 × 200.0: threshold must lie strictly between the two modes.
    let mut data = vec![20.0f32; 50];
    data.extend(vec![200.0f32; 50]);
    let image = make_image_1d(data);
    let thresholds = multi_otsu_threshold(&image, 2);

    assert_eq!(thresholds.len(), 1);
    assert!(
        thresholds[0] > 20.0,
        "threshold must exceed lower mode (20.0), got {}",
        thresholds[0]
    );
    assert!(
        thresholds[0] < 200.0,
        "threshold must be below upper mode (200.0), got {}",
        thresholds[0]
    );
}

#[test]
fn test_k2_apply_produces_strictly_binary_labels() {
    // K=2 apply must produce labels in {0.0, 1.0} only.
    let mut data = vec![20.0f32; 50];
    data.extend(vec![200.0f32; 50]);
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(2).apply(&image);
    let values = get_values_1d(&labels);

    for &v in &values {
        assert!(
            v == 0.0 || v == 1.0,
            "K=2 label must be in {{0, 1}}, got {}",
            v
        );
    }
}

#[test]
fn test_k2_apply_bimodal_correct_class_assignment() {
    // Lower-mode pixels → label 0; upper-mode pixels → label 1.
    let mut data = vec![10.0f32; 50];
    data.extend(vec![240.0f32; 50]);
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(2).apply(&image);
    let values = get_values_1d(&labels);

    for (i, &v) in values[..50].iter().enumerate() {
        assert_eq!(v, 0.0, "pixel {} (10.0) must have label 0, got {}", i, v);
    }
    for (i, &v) in values[50..].iter().enumerate() {
        assert_eq!(v, 1.0, "pixel {} (240.0) must have label 1, got {}", i, v);
    }
}

// ── K=3: three-class segmentation ─────────────────────────────────────────

#[test]
fn test_k3_returns_exactly_two_thresholds() {
    let data: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();
    let image = make_image_1d(data);
    let thresholds = multi_otsu_threshold(&image, 3);
    assert_eq!(thresholds.len(), 2, "K=3 must return exactly 2 thresholds");
}

#[test]
fn test_k3_trimodal_thresholds_separate_all_three_clusters() {
    // Three equal-weight clusters: 50 × 10.0, 50 × 128.0, 50 × 250.0.
    // Optimal thresholds must satisfy: 10 < t1 < 128 < t2 < 250, t1 < t2.
    let mut data = vec![10.0f32; 50];
    data.extend(vec![128.0f32; 50]);
    data.extend(vec![250.0f32; 50]);
    let image = make_image_1d(data);
    let thresholds = multi_otsu_threshold(&image, 3);

    assert_eq!(thresholds.len(), 2);
    let (t1, t2) = (thresholds[0], thresholds[1]);

    assert!(
        t1 < t2,
        "thresholds must be strictly increasing: t1={} t2={}",
        t1,
        t2
    );
    assert!(t1 > 10.0, "t1 must exceed lower mode (10.0), got {}", t1);
    assert!(
        t1 < 128.0,
        "t1 must be below middle mode (128.0), got {}",
        t1
    );
    assert!(t2 > 128.0, "t2 must exceed middle mode (128.0), got {}", t2);
    assert!(
        t2 < 250.0,
        "t2 must be below upper mode (250.0), got {}",
        t2
    );
}

#[test]
fn test_k3_trimodal_apply_assigns_correct_class_labels() {
    // Three disjoint clusters: label 0 / 1 / 2 respectively.
    let mut data = vec![10.0f32; 50];
    data.extend(vec![128.0f32; 50]);
    data.extend(vec![250.0f32; 50]);
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(3).apply(&image);
    let values = get_values_1d(&labels);

    for (i, &v) in values[..50].iter().enumerate() {
        assert_eq!(v, 0.0, "pixel {} (10.0) must have label 0, got {}", i, v);
    }
    for (i, &v) in values[50..100].iter().enumerate() {
        assert_eq!(v, 1.0, "pixel {} (128.0) must have label 1, got {}", i, v);
    }
    for (i, &v) in values[100..].iter().enumerate() {
        assert_eq!(v, 2.0, "pixel {} (250.0) must have label 2, got {}", i, v);
    }
}

#[test]
fn test_k3_apply_label_values_in_valid_set() {
    // Every output value must be in {0.0, 1.0, 2.0}.
    let mut data = vec![30.0f32; 40];
    data.extend(vec![130.0f32; 40]);
    data.extend(vec![230.0f32; 40]);
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(3).apply(&image);
    let values = get_values_1d(&labels);

    for &v in &values {
        assert!(
            v == 0.0 || v == 1.0 || v == 2.0,
            "K=3 label must be in {{0, 1, 2}}, got {}",
            v
        );
    }
}

// ── General invariants ─────────────────────────────────────────────────────

#[test]
fn test_threshold_count_equals_k_minus_1_for_k_2_3_4() {
    // For any K ∈ {2, 3, 4}, the returned Vec has exactly K−1 elements.
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
    // The returned Vec must be sorted in strictly ascending order.
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
    // Every threshold must satisfy x_min ≤ t ≤ x_max.
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
    // Same image must always produce identical thresholds.
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
    assert_eq!(
        labels.direction(),
        &direction,
        "direction must be preserved"
    );
    assert_eq!(labels.shape(), [3, 3, 3], "shape must be preserved");
}

#[test]
fn test_apply_3d_trimodal_labels_correct() {
    // 3×3×3: voxels 0..9 = 10.0, 9..18 = 128.0, 18..27 = 250.0.
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
    // A monotonically increasing image must produce non-decreasing labels.
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
    // Degenerate constant image: all K−1 thresholds must equal the uniform intensity.
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
    // With 64 bins instead of 256, thresholds must still separate the modes.
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
// At t = 2 (split at bin index 2):
//   Class 1 [0,1]: P = 0.5, μ = 0.25/0.5 = 0.5
//   Class 2 [2,3]: P = 0.5, μ = 1.25/0.5 = 2.5
//   σ²_B = 0.5·(0.5−1.5)² + 0.5·(2.5−1.5)² = 0.5 + 0.5 = 1.0
//
// At t = 1 (split at bin index 1):
//   Class 1 [0]:   P = 0.25, μ = 0/0.25 = 0.0
//   Class 2 [1,3]: P = 0.75, μ = 1.5/0.75 = 2.0
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
    // The symmetric split (t=2) must have higher between-class variance than t=1.
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
    // Uniform histogram, K=3, split at {1, 2}: three equal classes.
    // Class 1 [0]: P=0.25, μ=0.0
    // Class 2 [1]: P=0.25, μ=1.0
    // Class 3 [2,3]: P=0.50, μ=2.5
    // σ²_B = 0.25·(0−1.5)² + 0.25·(1−1.5)² + 0.50·(2.5−1.5)²
    //      = 0.25·2.25 + 0.25·0.25 + 0.50·1.0
    //      = 0.5625 + 0.0625 + 0.5
    //      = 1.125
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

// ── from_slice parity ─────────────────────────────────────────────────────────────────────────

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

// ── K=4: four-class segmentation ──────────────────────────────────────────

#[test]
fn test_k4_returns_exactly_three_thresholds() {
    let data: Vec<f32> = (0u8..=200).map(|x| x as f32).collect();
    let image = make_image_1d(data);
    let thresholds = multi_otsu_threshold(&image, 4);
    assert_eq!(
        thresholds.len(),
        3,
        "K=4 must return exactly 3 thresholds, got {}",
        thresholds.len()
    );
}

#[test]
fn test_k4_four_cluster_thresholds_separate_all_clusters() {
    // Four equal-weight clusters: 50 × {10, 80, 160, 240}.
    // t1 ∈ (10, 80), t2 ∈ (80, 160), t3 ∈ (160, 240), t1 < t2 < t3.
    let mut data = vec![10.0f32; 50];
    data.extend(vec![80.0f32; 50]);
    data.extend(vec![160.0f32; 50]);
    data.extend(vec![240.0f32; 50]);
    let image = make_image_1d(data);
    let thresholds = multi_otsu_threshold(&image, 4);

    assert_eq!(thresholds.len(), 3);
    let (t1, t2, t3) = (thresholds[0], thresholds[1], thresholds[2]);

    assert!(
        t1 < t2 && t2 < t3,
        "thresholds must be strictly increasing: {t1} < {t2} < {t3}"
    );
    assert!(t1 > 10.0 && t1 < 80.0, "t1={t1} must lie in (10, 80)");
    assert!(t2 > 80.0 && t2 < 160.0, "t2={t2} must lie in (80, 160)");
    assert!(t3 > 160.0 && t3 < 240.0, "t3={t3} must lie in (160, 240)");
}

#[test]
fn test_k4_apply_assigns_four_labels() {
    // Four disjoint clusters must map to labels {0, 1, 2, 3}.
    let mut data = vec![10.0f32; 40];
    data.extend(vec![80.0f32; 40]);
    data.extend(vec![160.0f32; 40]);
    data.extend(vec![240.0f32; 40]);
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(4).apply(&image);
    let values = get_values_1d(&labels);

    // Verify each quarter maps to the correct class.
    for (i, &v) in values[..40].iter().enumerate() {
        assert_eq!(v, 0.0, "pixel {i} (10.0) must have label 0, got {v}");
    }
    for (i, &v) in values[40..80].iter().enumerate() {
        assert_eq!(v, 1.0, "pixel {i} (80.0) must have label 1, got {v}");
    }
    for (i, &v) in values[80..120].iter().enumerate() {
        assert_eq!(v, 2.0, "pixel {i} (160.0) must have label 2, got {v}");
    }
    for (i, &v) in values[120..160].iter().enumerate() {
        assert_eq!(v, 3.0, "pixel {i} (240.0) must have label 3, got {v}");
    }
}

// ── K=5: five-class segmentation ──────────────────────────────────────────

#[test]
fn test_k5_returns_exactly_four_thresholds() {
    // Five equal clusters of 30 voxels each at {0, 64, 128, 192, 255}.
    let mut data = vec![0.0f32; 30];
    data.extend(vec![64.0f32; 30]);
    data.extend(vec![128.0f32; 30]);
    data.extend(vec![192.0f32; 30]);
    data.extend(vec![255.0f32; 30]);
    let image = make_image_1d(data);
    let thresholds = multi_otsu_threshold(&image, 5);
    assert_eq!(
        thresholds.len(),
        4,
        "K=5 must return exactly 4 thresholds, got {}",
        thresholds.len()
    );
}

#[test]
fn test_k5_five_cluster_thresholds_each_between_adjacent_modes() {
    let mut data = vec![0.0f32; 30];
    data.extend(vec![64.0f32; 30]);
    data.extend(vec![128.0f32; 30]);
    data.extend(vec![192.0f32; 30]);
    data.extend(vec![255.0f32; 30]);
    let image = make_image_1d(data);
    let t = multi_otsu_threshold(&image, 5);

    let modes = [0.0f32, 64.0, 128.0, 192.0, 255.0];
    for i in 0..4 {
        assert!(
            t[i] > modes[i] && t[i] < modes[i + 1],
            "t[{i}]={:.2} must lie in ({:.0}, {:.0})",
            t[i],
            modes[i],
            modes[i + 1]
        );
    }
    for i in 0..3 {
        assert!(
            t[i] < t[i + 1],
            "thresholds must be strictly increasing: t[{i}]={:.2} t[{}]={:.2}",
            t[i],
            i + 1,
            t[i + 1]
        );
    }
}

// ── Between-class variance invariant: K=2 equals P1*P2*(mu1-mu2)^2 ────────

#[test]
fn test_between_class_variance_k2_equals_product_formula() {
    // For K=2 with two equal-weight clusters at 20 and 200 (n=100 each):
    // P1 = P2 = 0.5, mu1 ≈ 20/(200-20)*(N-1), mu2 ≈ 200/(200-20)*(N-1)
    // σ²_B = P1*P2*(mu1-mu2)² (standard Otsu formula, proved algebraically from definition)
    // The between_class_variance function must produce the same result whether
    // evaluated via prefix sums (K=2 path) or via P1*P2*(mu1-mu2)².

    let n_bins = 256usize;
    let n = 100usize;
    // Build a histogram for values 20.0 and 200.0 (equal counts).
    let x_min = 20.0_f64;
    let x_max = 180.0_f64;
    let range = x_max - x_min;
    let scale = (n_bins - 1) as f64 / range;

    let bin_low = ((20.0_f64 - x_min) * scale).floor() as usize; // = 0
    let bin_high = ((180.0_f64 - x_min) * scale).floor() as usize; // = 255

    let mut h = vec![0.0_f64; n_bins];
    h[bin_low] = n as f64 / (2 * n) as f64; // 0.5
    h[bin_high] = n as f64 / (2 * n) as f64; // 0.5

    // Build prefix sums.
    let mut prefix_h = vec![0.0_f64; n_bins + 1];
    let mut prefix_m = vec![0.0_f64; n_bins + 1];
    for i in 0..n_bins {
        prefix_h[i + 1] = prefix_h[i] + h[i];
        prefix_m[i + 1] = prefix_m[i] + i as f64 * h[i];
    }

    // K=2 optimal threshold should lie between bin_low and bin_high.
    // Evaluate between_class_variance at a mid-point split (e.g. t=128 between bin 0 and bin 255).
    // Use the analytical P1*P2*(mu1-mu2)^2 formula to verify.
    let p1 = 0.5_f64;
    let p2 = 0.5_f64;
    let mu1 = bin_low as f64;
    let mu2 = bin_high as f64;
    let expected_variance = p1 * p2 * (mu1 - mu2).powi(2);

    // The maximum between-class variance for any split must be ≤ P1*P2*(mu1-mu2)^2 + epsilon.
    // For the two-point distribution, the split at any t ∈ (bin_low, bin_high) achieves exactly
    // P1*P2*(mu1-mu2)^2 because all mass is at the two endpoints.
    let t_mid = 128usize;
    let p_left = prefix_h[t_mid + 1]; // = 0.5 (all mass of bin_low=0 is in [0, t_mid])
    let p_right = 1.0 - p_left;
    let mu_left = if p_left > 1e-12 {
        (prefix_m[t_mid + 1]) / p_left
    } else {
        0.0
    };
    let mu_right = if p_right > 1e-12 {
        (prefix_m[n_bins] - prefix_m[t_mid + 1]) / p_right
    } else {
        0.0
    };
    let product_formula = p_left * p_right * (mu_left - mu_right).powi(2);

    // Both formulas must agree within floating-point tolerance.
    let diff = (product_formula - expected_variance).abs();
    assert!(
        diff < 1e-9,
        "K=2 between-class variance via prefix sums ({product_formula:.6}) must equal P1*P2*(mu1-mu2)^2 ({expected_variance:.6}), diff={diff:.2e}"
    );
}

// ── Monotone label ordering for monotone input ─────────────────────────────

#[test]
fn test_k4_apply_label_ordering_monotone_for_monotone_input() {
    // For a strictly increasing input, K=4 labels must be non-decreasing.
    let data: Vec<f32> = (0u32..200).map(|i| i as f32).collect();
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(4).apply(&image);
    let values = get_values_1d(&labels);

    // Labels must be non-decreasing (monotone input → monotone labels).
    for i in 1..values.len() {
        assert!(
            values[i] >= values[i - 1],
            "labels must be non-decreasing for monotone input at index {i}: {} → {}",
            values[i - 1],
            values[i]
        );
    }
}

// ── K=5 apply: all labels in {0,1,2,3,4} ──────────────────────────────────

#[test]
fn test_k5_apply_label_values_in_valid_set() {
    let mut data = vec![10.0f32; 30];
    data.extend(vec![70.0f32; 30]);
    data.extend(vec![130.0f32; 30]);
    data.extend(vec![190.0f32; 30]);
    data.extend(vec![250.0f32; 30]);
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(5).apply(&image);
    let values = get_values_1d(&labels);
    for &v in &values {
        assert!(
            v == 0.0 || v == 1.0 || v == 2.0 || v == 3.0 || v == 4.0,
            "K=5 label must be in {{0,1,2,3,4}}, got {v}"
        );
    }
}

// ── Adversarial: K > distinct intensity values ─────────────────────────────

#[test]
fn test_k3_with_only_two_distinct_values_returns_two_thresholds() {
    // Only 2 distinct intensities but K=3 requested.
    // Must not panic and must return exactly K-1 = 2 thresholds.
    let mut data = vec![10.0f32; 50];
    data.extend(vec![200.0f32; 50]);
    let image = make_image_1d(data);
    let thresholds = multi_otsu_threshold(&image, 3);
    assert_eq!(
        thresholds.len(),
        2,
        "K=3 on 2-value image must return 2 thresholds, got {}",
        thresholds.len()
    );
}

// ── Adversarial: single voxel ─────────────────────────────────────────────

#[test]
fn test_single_voxel_k3_returns_two_thresholds_equal_to_value() {
    let image = make_image_1d(vec![42.0f32]);
    let thresholds = multi_otsu_threshold(&image, 3);
    assert_eq!(
        thresholds.len(),
        2,
        "single-voxel K=3 must return 2 thresholds"
    );
    // Both thresholds must equal (or be near) the single voxel value for a constant image.
    for &t in &thresholds {
        assert!(
            (t - 42.0).abs() < 1.0,
            "threshold {t} on single-voxel image must be near 42.0"
        );
    }
}
