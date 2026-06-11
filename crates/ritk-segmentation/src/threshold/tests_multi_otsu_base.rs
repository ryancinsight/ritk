//! Tests for multi_otsu
//! Extracted from the main module to keep the 500-line structural limit.
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
