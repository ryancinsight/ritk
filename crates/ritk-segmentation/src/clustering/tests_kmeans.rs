//! Tests for kmeans
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support::make_image;

type B = NdArray<f32>;

fn make_image_1d(data: Vec<f32>) -> Image<B, 1> {
    let n = data.len();
    make_image(data, [n])
}
fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    make_image(data, dims)
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
    let result = KMeansSegmentation::new(3).unwrap().apply(&image).unwrap();
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
    let result = KMeansSegmentation::new(2).unwrap().apply(&image).unwrap();
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
    let result = KMeansSegmentation::new(3).unwrap().apply(&image).unwrap();
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
    let result = KMeansSegmentation::new(k).unwrap().apply(&image).unwrap();
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
    let result = KMeansSegmentation::new(2).unwrap().apply(&image).unwrap();

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
    let result = kmeans_segment(&image, 2).unwrap();
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

    let r1 = KMeansSegmentation::new(2).unwrap().apply(&image).unwrap();
    let r2 = KMeansSegmentation::new(2).unwrap().apply(&image).unwrap();

    let l1 = get_slice_1d(&r1);
    let l2 = get_slice_1d(&r2);
    assert_eq!(l1, l2, "same seed must produce identical results");
}

// ── Zero seed is valid (no panic) ──────────────────────────────────────────

#[test]
fn test_seed_zero_does_not_panic() {
    // Regression: a zero seed reached xorshift64's fixed point and panicked
    // ("seed must be non-zero"). `seed = 0` is a natural user default and must
    // segment normally; it is remapped to a fixed nonzero state internally.
    let mut data = vec![10.0_f32; 50];
    data.extend(vec![200.0_f32; 50]);
    let image = make_image_1d(data);

    let seg = KMeansSegmentation::new(2).unwrap().with_seed(0);
    let labels = get_slice_1d(&seg.apply(&image).unwrap());

    // Valid two-cluster partition: the two modes land in different clusters.
    for &l in &labels {
        assert!((0.0..2.0).contains(&l), "label out of range: {l}");
    }
    assert!(
        (labels[0] - labels[50]).abs() > 0.5,
        "seed=0 must still separate the two modes: {} vs {}",
        labels[0],
        labels[50]
    );
}

// ── k=1 assigns all label 0 ────────────────────────────────────────────────

#[test]
fn test_k1_all_zero() {
    let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let image = make_image_1d(data);
    let result = KMeansSegmentation::new(1).unwrap().apply(&image).unwrap();
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
    assert_eq!(d.k(), 2);
    assert_eq!(d.max_iterations(), 100);
    assert_eq!(d.tolerance(), 1e-6);
    assert_eq!(d.seed(), 42);
}

#[test]
fn native_and_legacy_boundaries_are_exactly_equivalent() {
    let dimensions = [2, 2, 3];
    let values = vec![
        -3.0, -2.0, -1.0, 10.0, 11.0, 12.0, 100.0, 101.0, 102.0, 9.0, 0.0, 99.0,
    ];
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
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
    let legacy = make_image_3d(values, dimensions);
    let segmentation = KMeansSegmentation::new(3)
        .unwrap()
        .with_max_iterations(20)
        .unwrap()
        .with_tolerance(0.0)
        .unwrap()
        .with_seed(7);

    let native_output = segmentation
        .apply_native(&native, &SequentialBackend)
        .unwrap();
    let legacy_output = segmentation.apply(&legacy).unwrap();

    assert_eq!(
        native_output.data_slice().unwrap(),
        legacy_output.data_slice().as_ref()
    );
    assert_eq!(native_output.shape(), dimensions);
    assert_eq!(*native_output.origin(), origin);
    assert_eq!(*native_output.spacing(), spacing);
    assert_eq!(*native_output.direction(), direction);
}

#[test]
fn invalid_configuration_reports_exact_contract_errors() {
    assert_eq!(
        KMeansSegmentation::new(0).unwrap_err().to_string(),
        "k must be at least 1, got 0"
    );
    assert_eq!(
        KMeansSegmentation::new(MAX_EXACT_LABELS + 1)
            .unwrap_err()
            .to_string(),
        format!(
            "k must not exceed {MAX_EXACT_LABELS} because output labels are f32, got {}",
            MAX_EXACT_LABELS + 1
        )
    );
    assert_eq!(
        KMeansSegmentation::default()
            .with_max_iterations(0)
            .unwrap_err()
            .to_string(),
        "k-means maximum iterations must be at least 1, got 0"
    );
    for (value, expected) in [
        (
            -1.0,
            "k-means tolerance must be finite and nonnegative, got -1",
        ),
        (
            f32::INFINITY,
            "k-means tolerance must be finite and nonnegative, got inf",
        ),
        (
            f32::NEG_INFINITY,
            "k-means tolerance must be finite and nonnegative, got -inf",
        ),
        (
            f32::NAN,
            "k-means tolerance must be finite and nonnegative, got NaN",
        ),
    ] {
        assert_eq!(
            KMeansSegmentation::default()
                .with_tolerance(value)
                .unwrap_err()
                .to_string(),
            expected
        );
    }
}

#[test]
fn nonfinite_samples_are_rejected_at_both_boundaries() {
    for (value, rendered) in [
        (f32::NAN, "NaN"),
        (f32::INFINITY, "inf"),
        (f32::NEG_INFINITY, "-inf"),
    ] {
        let values = vec![0.0, value, 1.0];
        let native = NativeImage::from_flat_on(
            values.clone(),
            [3],
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
            &SequentialBackend,
        )
        .unwrap();
        let expected = format!("k-means sample at flat index 1 must be finite, got {rendered}");
        assert_eq!(
            KMeansSegmentation::default()
                .apply(&make_image_1d(values))
                .unwrap_err()
                .to_string(),
            expected
        );
        assert_eq!(
            KMeansSegmentation::default()
                .apply_native(&native, &SequentialBackend)
                .unwrap_err()
                .to_string(),
            expected
        );
    }
}

#[test]
fn finite_overflow_range_is_rejected_at_both_boundaries() {
    let values = vec![f32::MIN, f32::MAX];
    let native = NativeImage::from_flat_on(
        values.clone(),
        [2],
        Point::new([0.0]),
        Spacing::new([1.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    let expected = format!(
        "k-means sample range must be representable in f32, got [{}, {}]",
        f32::MIN,
        f32::MAX
    );
    assert_eq!(
        KMeansSegmentation::default()
            .apply(&make_image_1d(values))
            .unwrap_err()
            .to_string(),
        expected
    );
    assert_eq!(
        KMeansSegmentation::default()
            .apply_native(&native, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        expected
    );
}

#[test]
fn large_same_sign_samples_compute_without_overflow() {
    let scale = f32::MAX / 4.0;
    let values = vec![scale, scale, 2.0 * scale, 2.0 * scale];
    let labels = KMeansSegmentation::new(2)
        .unwrap()
        .with_seed(7)
        .apply(&make_image_1d(values))
        .unwrap();
    assert_eq!(labels.data_slice().as_ref(), &[1.0, 1.0, 0.0, 0.0]);
}
