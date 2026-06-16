use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::test_support;
use ritk_spatial::{Direction, Point, Spacing};

type TestBackend = NdArray<f32>;

fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
    test_support::make_image_1d(data)
}



#[test]
fn test_zscore_zero_mean() {
    // Values [1, 2, 3, 4, 5]: mean = 3.0
    // After normalization the mean of the output must be ≈ 0.
    let image = make_image_1d(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let normalizer = ZScoreNormalizer::new();
    let result = normalizer.normalize(&image);

    let stats = crate::image_statistics::compute_statistics(&result);
    assert!(
        stats.mean.abs() < 1e-5,
        "output mean must be ≈ 0, got {}",
        stats.mean
    );
}

#[test]
fn test_zscore_unit_variance() {
    // Values [1, 2, 3, 4, 5]:
    //   variance = Σ(xᵢ−3)²/5 = (4+1+0+1+4)/5 = 2  →  std = √2
    //   After division by (√2 + ε), output std ≈ 1.
    let image = make_image_1d(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let normalizer = ZScoreNormalizer::new();
    let result = normalizer.normalize(&image);

    let stats = crate::image_statistics::compute_statistics(&result);
    assert!(
        (stats.std - 1.0).abs() < 1e-3,
        "output std must be ≈ 1, got {}",
        stats.std
    );
}

#[test]
fn test_zscore_preserves_metadata() {
    let device = Default::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0f32; 27], Shape::new([3, 3, 3])),
        &device,
    );
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.5, 0.5]);
    let direction = Direction::identity();
    let image: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

    let normalizer = ZScoreNormalizer::new();
    let result = normalizer.normalize(&image);

    assert_eq!(result.origin(), &origin, "origin must be preserved");
    assert_eq!(result.spacing(), &spacing, "spacing must be preserved");
    assert_eq!(
        result.direction(),
        &direction,
        "direction must be preserved"
    );
    assert_eq!(result.shape(), [3, 3, 3], "shape must be preserved");
}

#[test]
fn test_zscore_known_values() {
    // Values [1, 3]:
    //   mean = 2.0
    //   variance = ((1−2)² + (3−2)²) / 2 = 1.0 → std = 1.0
    //   z(1) = (1−2)/(1+1e-8) ≈ −1.0
    //   z(3) = (3−2)/(1+1e-8) ≈  1.0
    let image = make_image_1d(vec![1.0, 3.0]);
    let normalizer = ZScoreNormalizer::new();
    let result = normalizer.normalize(&image);

    let (slice, _) = extract_vec_infallible(&result);

    assert!(
        (slice[0] - (-1.0)).abs() < 1e-4,
        "z(1) ≈ -1.0, got {}",
        slice[0]
    );
    assert!(
        (slice[1] - 1.0).abs() < 1e-4,
        "z(3) ≈ 1.0, got {}",
        slice[1]
    );
}

// ── Boundary / edge cases ──────────────────────────────────────────────────

#[test]
fn test_zscore_constant_image_does_not_panic() {
    // Constant image: std = 0. Division by (0 + 1e-8) must not panic.
    // All output values = (5 − 5) / 1e-8 = 0.
    let image = make_image_1d(vec![5.0; 8]);
    let normalizer = ZScoreNormalizer::new();
    let result = normalizer.normalize(&image);

    let (slice, _) = extract_vec_infallible(&result);
    for &v in slice.iter() {
        assert!(
            v.abs() < 1e-3,
            "constant image must normalize to ≈ 0, got {}",
            v
        );
    }
}

#[test]
fn test_zscore_single_voxel() {
    // Single voxel: mean = value, std = 0 → output = 0.
    let image = make_image_1d(vec![100.0]);
    let normalizer = ZScoreNormalizer::new();
    let result = normalizer.normalize(&image);

    let (slice, _) = extract_vec_infallible(&result);
    assert!(
        slice[0].abs() < 1e-3,
        "single voxel output = 0, got {}",
        slice[0]
    );
}

// ── normalize_masked tests ─────────────────────────────────────────────────

#[test]
fn test_zscore_masked_uses_mask_statistics() {
    // Image: [1.0, 2.0, 3.0, 100.0, 200.0]
    // Mask:  [1.0, 1.0, 1.0,   0.0,   0.0]
    //
    // Masked values: [1, 2, 3]
    //   mean     = 2.0
    //   variance = ((1−2)² + (2−2)² + (3−2)²) / 3 = 2/3
    //   std      = √(2/3) ≈ 0.81650
    //   denom    = std + 1e-8
    //
    // z(1.0)   = (1 − 2) / denom ≈ −1.2247
    // z(2.0)   = (2 − 2) / denom =  0.0
    // z(3.0)   = (3 − 2) / denom ≈  1.2247
    // z(100.0) = (100 − 2) / denom ≈ 120.04  (background voxels also transformed)
    // z(200.0) = (200 − 2) / denom ≈ 242.28
    let image = make_image_1d(vec![1.0, 2.0, 3.0, 100.0, 200.0]);
    let mask = make_image_1d(vec![1.0, 1.0, 1.0, 0.0, 0.0]);
    let normalizer = ZScoreNormalizer::new();
    let result = normalizer.normalize_masked(&image, &mask);

    let (slice, _) = extract_vec_infallible(&result);

    let expected_std = (2.0_f32 / 3.0_f32).sqrt();
    let denom = expected_std + crate::normalization::NORMALIZER_EPSILON;
    let expected_z3 = 1.0_f32 / denom; // z(3.0) = (3 − 2) / denom ≈ 1.2247

    assert!(
        (slice[2] - expected_z3).abs() < 1e-4,
        "z(3.0) must be ≈ {expected_z3}, got {}",
        slice[2]
    );
    // Ordering within masked region.
    assert!(
        slice[0] < slice[1] && slice[1] < slice[2],
        "ordering violated: slice[0]={} slice[1]={} slice[2]={}",
        slice[0],
        slice[1],
        slice[2]
    );
    // Background voxels are transformed — not left at original values.
    let expected_z100 = (100.0_f32 - 2.0_f32) / denom;
    assert!(
        (slice[3] - expected_z100).abs() < 1e-2,
        "z(100.0) must be ≈ {expected_z100}, got {}",
        slice[3]
    );
    // Ordering continues into background region.
    assert!(
        slice[3] < slice[4],
        "z(100) < z(200) must hold: got {} and {}",
        slice[3],
        slice[4]
    );
}

#[test]
fn test_zscore_masked_empty_mask_falls_back_to_full_image() {
    // All-zero mask: masked_statistics would panic; normalize_masked must
    // fall back to full-image statistics and produce output identical to
    // normalize.
    let image = make_image_1d(vec![1.0, 2.0, 3.0, 100.0, 200.0]);
    let mask = make_image_1d(vec![0.0; 5]);
    let normalizer = ZScoreNormalizer::new();

    let result_masked = normalizer.normalize_masked(&image, &mask);
    let result_full = normalizer.normalize(&image);

    let (masked_slice, _) = extract_vec_infallible(&result_masked);
    let (full_slice, _) = extract_vec_infallible(&result_full);

    for (i, (&m, &f)) in masked_slice.iter().zip(full_slice.iter()).enumerate() {
        assert!(
            (m - f).abs() < 1e-6,
            "fallback mismatch at index {i}: masked={m} full={f}"
        );
    }
}

#[test]
fn test_zscore_masked_preserves_metadata() {
    // Verify origin, spacing, direction, and shape are carried through
    // normalize_masked unchanged.
    let device = Default::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0f32; 27], Shape::new([3, 3, 3])),
        &device,
    );
    // Single foreground voxel so masked_statistics path is exercised.
    let mut mask_data = vec![0.0f32; 27];
    mask_data[0] = 1.0;
    let mask_tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(mask_data, Shape::new([3, 3, 3])),
        &device,
    );
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.5, 0.5]);
    let direction = Direction::identity();
    let image: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);
    let mask: Image<TestBackend, 3> = Image::new(mask_tensor, origin, spacing, direction);

    let normalizer = ZScoreNormalizer::new();
    let result = normalizer.normalize_masked(&image, &mask);

    assert_eq!(result.origin(), &origin, "origin must be preserved");
    assert_eq!(result.spacing(), &spacing, "spacing must be preserved");
    assert_eq!(
        result.direction(),
        &direction,
        "direction must be preserved"
    );
    assert_eq!(result.shape(), [3, 3, 3], "shape must be preserved");
}

#[test]
fn test_zscore_negative_values_preserved_sign() {
    // Values [−2, −1, 0, 1, 2]: mean = 0, std = √2 ≈ 1.4142
    // z(−2) < z(−1) < z(0) == 0 < z(1) < z(2): ordering preserved.
    let image = make_image_1d(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let normalizer = ZScoreNormalizer::new();
    let result = normalizer.normalize(&image);

    let (slice, _) = extract_vec_infallible(&result);

    for i in 0..4 {
        assert!(
            slice[i] < slice[i + 1],
            "ordering violated at index {}: {} >= {}",
            i,
            slice[i],
            slice[i + 1]
        );
    }
    // Central value (0) maps to 0.
    assert!(slice[2].abs() < 1e-5, "z(0) must be 0, got {}", slice[2]);
}
