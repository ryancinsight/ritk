//! Tests for nyul_udupa
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;

type TestBackend = NdArray<f32>;

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
    let image: Image<TestBackend, 1> = make_image(data.clone(), [100]);

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
fn test_apply_preserves_original_voxel_order_after_landmark_sort() {
    let data = vec![8.0, 1.0, 6.0, 3.0, 5.0, 2.0, 7.0, 4.0];
    let image: Image<TestBackend, 1> = make_image(data.clone(), [8]);

    let mut normalizer = NyulUdupaNormalizer::with_percentiles(vec![0.0, 50.0, 100.0]);
    normalizer.learn_standard(&[&image]);

    let result = normalizer.apply(&image).expect("apply");
    let result_vals = get_values(&result);

    assert_eq!(
        result_vals, data,
        "identity mapping must preserve original voxel order after sorting landmarks"
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

    let image_a: Image<TestBackend, 1> = make_image(data_a, [200]);
    let image_b: Image<TestBackend, 1> = make_image(data_b, [200]);

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
    let image: Image<TestBackend, 1> = make_image(data, [50]);

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
    let image: Image<TestBackend, 1> = make_image(data.clone(), [100]);

    let mut normalizer = NyulUdupaNormalizer::with_percentiles(vec![5.0, 25.0, 50.0, 75.0, 95.0]);
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
    let image_a: Image<TestBackend, 1> = make_image(data_a, [100]);
    let image_b: Image<TestBackend, 1> = make_image(data_b, [100]);

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
    let image: Image<TestBackend, 1> = make_image(data, [10]);

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
    let image: Image<TestBackend, 1> = make_image(data, [100]);

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
