//! Extracted tests: K=4, K=5, between-class variance K=2 product formula, adversarial.
use super::*;
use ritk_core::spatial::{Direction, Point, Spacing};
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

fn get_values_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
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
    let mut data = vec![10.0f32; 40];
    data.extend(vec![80.0f32; 40]);
    data.extend(vec![160.0f32; 40]);
    data.extend(vec![240.0f32; 40]);
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(4).apply(&image);
    let values = get_values_1d(&labels);
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

#[test]
fn test_k4_apply_label_ordering_monotone_for_monotone_input() {
    let data: Vec<f32> = (0u32..200).map(|i| i as f32).collect();
    let image = make_image_1d(data);
    let labels = MultiOtsuThreshold::new(4).apply(&image);
    let values = get_values_1d(&labels);
    for i in 1..values.len() {
        assert!(
            values[i] >= values[i - 1],
            "labels must be non-decreasing for monotone input at index {i}: {} → {}",
            values[i - 1],
            values[i]
        );
    }
}

// ── K=5: five-class segmentation ──────────────────────────────────────────

#[test]
fn test_k5_returns_exactly_four_thresholds() {
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

// ── Between-class variance invariant: K=2 equals P1*P2*(mu1-mu2)^2 ────────
//
// For a two-point distribution with all mass at bin_low and bin_high,
// the prefix-sum formula must produce P1*P2*(mu1-mu2)^2 (proved by algebraic
// substitution into the definition of between-class variance).

#[test]
fn test_between_class_variance_k2_equals_product_formula() {
    let n_bins = 256usize;
    let n = 100usize;
    let x_min = 20.0_f64;
    let x_max = 180.0_f64;
    let range = x_max - x_min;
    let scale = (n_bins - 1) as f64 / range;
    let bin_low = ((20.0_f64 - x_min) * scale).floor() as usize;
    let bin_high = ((180.0_f64 - x_min) * scale).floor() as usize;
    let mut h = vec![0.0_f64; n_bins];
    h[bin_low] = n as f64 / (2 * n) as f64;
    h[bin_high] = n as f64 / (2 * n) as f64;
    let mut prefix_h = vec![0.0_f64; n_bins + 1];
    let mut prefix_m = vec![0.0_f64; n_bins + 1];
    for i in 0..n_bins {
        prefix_h[i + 1] = prefix_h[i] + h[i];
        prefix_m[i + 1] = prefix_m[i] + i as f64 * h[i];
    }
    let p1 = 0.5_f64;
    let p2 = 0.5_f64;
    let mu1 = bin_low as f64;
    let mu2 = bin_high as f64;
    let expected_variance = p1 * p2 * (mu1 - mu2).powi(2);
    let t_mid = 128usize;
    let p_left = prefix_h[t_mid + 1];
    let p_right = 1.0 - p_left;
    let mu_left = if p_left > 1e-12 {
        prefix_m[t_mid + 1] / p_left
    } else {
        0.0
    };
    let mu_right = if p_right > 1e-12 {
        (prefix_m[n_bins] - prefix_m[t_mid + 1]) / p_right
    } else {
        0.0
    };
    let product_formula = p_left * p_right * (mu_left - mu_right).powi(2);
    let diff = (product_formula - expected_variance).abs();
    assert!(
        diff < 1e-9,
        "K=2 between-class variance via prefix sums ({product_formula:.6}) must equal \
         P1*P2*(mu1-mu2)^2 ({expected_variance:.6}), diff={diff:.2e}"
    );
}

// ── Adversarial: K > distinct intensity values ─────────────────────────────

#[test]
fn test_k3_with_only_two_distinct_values_returns_two_thresholds() {
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

#[test]
fn test_single_voxel_k3_returns_two_thresholds_equal_to_value() {
    let image = make_image_1d(vec![42.0f32]);
    let thresholds = multi_otsu_threshold(&image, 3);
    assert_eq!(
        thresholds.len(),
        2,
        "single-voxel K=3 must return 2 thresholds"
    );
    for &t in &thresholds {
        assert!(
            (t - 42.0).abs() < 1.0,
            "threshold {t} on single-voxel image must be near 42.0"
        );
    }
}
