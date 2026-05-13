//! Tests for kmeans
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn make_image_1d(data: Vec<f32>) -> Image<B, 1> {
    let n = data.len();
    let device = Default::default();
    let tensor = Tensor::<B, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
    Image::new(
        tensor,
        Point::new([0.0]),
        Spacing::new([1.0]),
        Direction::identity(),
    )
}

fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn get_slice_1d(image: &Image<B, 1>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

// ── Degenerate / constant image ────────────────────────────────────────────

#[test]
fn test_constant_image_all_same_label() {
    let data = vec![42.0_f32; 100];
    let image = make_image_1d(data);
    let result = KMeansSegmentation::new(3).apply(&image);
    let labels = get_slice_1d(&result);

    // All voxels must have the same label (cluster).
    let first = labels[0];
    for &l in &labels {
        assert!(
            (l - first).abs() < f32::EPSILON,
            "constant image must yield uniform labels, found {} and {}",
            first,
            l
        );
    }
}

// ── Bimodal image produces two clusters ────────────────────────────────────

#[test]
fn test_bimodal_two_clusters() {
    // 50 voxels at 10.0 and 50 voxels at 200.0, k=2.
    // All low-intensity voxels must share one label, all high another.
    let mut data = vec![10.0_f32; 50];
    data.extend(vec![200.0_f32; 50]);
    let image = make_image_1d(data);
    let result = KMeansSegmentation::new(2).apply(&image);
    let labels = get_slice_1d(&result);

    // Labels in [0, K-1].
    for &l in &labels {
        assert!(
            (0.0..2.0).contains(&l),
            "label must be in [0, 2), got {}",
            l
        );
    }

    // The first 50 must share a label, the last 50 must share a different label.
    let low_label = labels[0];
    let high_label = labels[50];
    assert!(
        (low_label - high_label).abs() > 0.5,
        "two distinct modes must get different labels: {} vs {}",
        low_label,
        high_label
    );

    for (i, &lbl) in labels.iter().enumerate().take(50) {
        assert!(
            (lbl - low_label).abs() < f32::EPSILON,
            "low-mode voxel {} has inconsistent label {} (expected {})",
            i,
            lbl,
            low_label
        );
    }
    for (i, &lbl) in labels.iter().enumerate().skip(50).take(50) {
        assert!(
            (lbl - high_label).abs() < f32::EPSILON,
            "high-mode voxel {} has inconsistent label {} (expected {})",
            i,
            lbl,
            high_label
        );
    }
}

// ── Output shape matches input shape ───────────────────────────────────────

#[test]
fn test_apply_output_shape_matches_input() {
    let dims = [4, 5, 6];
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i % 3) as f32 * 50.0).collect();
    let image = make_image_3d(data, dims);
    let result = KMeansSegmentation::new(3).apply(&image);
    assert_eq!(result.shape(), dims);
}

// ── Labels in valid range ──────────────────────────────────────────────────

#[test]
fn test_labels_in_valid_range() {
    let k = 4;
    let mut data = Vec::new();
    for c in 0..k {
        data.extend(vec![c as f32 * 80.0; 25]);
    }
    let image = make_image_1d(data);
    let result = KMeansSegmentation::new(k).apply(&image);
    let labels = get_slice_1d(&result);

    for &l in &labels {
        let li = l as usize;
        assert!(li < k, "label {} must be in [0, {})", l, k);
        assert!(
            (l - li as f32).abs() < f32::EPSILON,
            "label must be an integer, got {}",
            l
        );
    }
}

// ── Spatial metadata preserved ─────────────────────────────────────────────

#[test]
fn test_apply_preserves_spatial_metadata() {
    let dims = [2, 3, 4];
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 10.0).collect();
    let image = make_image_3d(data, dims);
    let result = KMeansSegmentation::new(2).apply(&image);

    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}

// ── Convenience function ───────────────────────────────────────────────────

#[test]
fn test_convenience_fn_produces_valid_output() {
    let mut data = vec![10.0_f32; 30];
    data.extend(vec![200.0_f32; 30]);
    let image = make_image_1d(data);
    let result = kmeans_segment(&image, 2);
    let labels = get_slice_1d(&result);
    assert_eq!(labels.len(), 60);
    for &l in &labels {
        assert!(
            l == 0.0 || l == 1.0,
            "must produce binary labels, got {}",
            l
        );
    }
}

// ── Determinism ────────────────────────────────────────────────────────────

#[test]
fn test_deterministic_with_same_seed() {
    let mut data = vec![10.0_f32; 50];
    data.extend(vec![200.0_f32; 50]);
    let image = make_image_1d(data);

    let r1 = KMeansSegmentation::new(2).apply(&image);
    let r2 = KMeansSegmentation::new(2).apply(&image);

    let l1 = get_slice_1d(&r1);
    let l2 = get_slice_1d(&r2);
    assert_eq!(l1, l2, "same seed must produce identical results");
}

// ── k=1 assigns all label 0 ────────────────────────────────────────────────

#[test]
fn test_k1_all_zero() {
    let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let image = make_image_1d(data);
    let result = KMeansSegmentation::new(1).apply(&image);
    let labels = get_slice_1d(&result);
    for &l in &labels {
        assert!(
            (l - 0.0).abs() < f32::EPSILON,
            "k=1 must assign all label 0, got {}",
            l
        );
    }
}

// ── Default trait ──────────────────────────────────────────────────────────

#[test]
fn test_default_k2() {
    let d = KMeansSegmentation::default();
    assert_eq!(d.k, 2);
    assert_eq!(d.max_iterations, 100);
    assert!((d.tolerance - 1e-6).abs() < 1e-15);
    assert_eq!(d.seed, 42);
}
