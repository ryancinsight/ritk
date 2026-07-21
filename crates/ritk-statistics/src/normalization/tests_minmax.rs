use super::*;
use coeus_core::SequentialBackend;
use ritk_image::test_support::{make_image, make_image_with};
use ritk_image::Image as NativeImage;

type TestBackend = SequentialBackend;

fn get_values(image: &Image<f32, TestBackend, 1>) -> Vec<f32> {
    ritk_tensor_ops::extract_vec_infallible(image).0
}

fn get_values_3d(image: &Image<f32, TestBackend, 3>) -> Vec<f32> {
    ritk_tensor_ops::extract_vec_infallible(image).0
}

// ── Positive tests ─────────────────────────────────────────────────────────

#[test]
fn test_minmax_known_values_unit_range() {
    // Values [0, 5, 10]: min=0, max=10, range=10
    //   N(0)  = 0.0
    //   N(5)  = 0.5
    //   N(10) = (10−0)/(10+1e-8) ≈ 1.0
    let image: Image<f32, TestBackend, 1> = make_image(vec![0.0, 5.0, 10.0], [3]);
    let normalizer = MinMaxNormalizer::new();
    let result = normalizer.normalize(&image);
    let values = get_values(&result);

    assert!(values[0].abs() < 1e-5, "N(0) ≈ 0.0, got {}", values[0]);
    assert!(
        (values[1] - 0.5).abs() < 1e-4,
        "N(5) ≈ 0.5, got {}",
        values[1]
    );
    assert!(
        (values[2] - 1.0).abs() < 1e-4,
        "N(10) ≈ 1.0, got {}",
        values[2]
    );
}

#[test]
fn native_minmax_maps_endpoints_and_preserves_metadata() {
    let image = NativeImage::from_flat_on(
        vec![0.0, 5.0, 10.0],
        [1, 1, 3],
        ritk_spatial::Point::new([1.0, 2.0, 3.0]),
        ritk_spatial::Spacing::new([0.5, 1.0, 2.0]),
        ritk_spatial::Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = MinMaxNormalizer::with_range(-1.0, 1.0)
        .normalize_native(&image)
        .expect("native min-max normalization succeeds");
    let values = output.data_slice().expect("contiguous output");
    assert_eq!(values[0], -1.0);
    assert!((values[1]).abs() < 1e-6);
    assert!((values[2] - 1.0).abs() < 1e-6);
    assert_eq!(output.origin(), image.origin());
    assert_eq!(output.spacing(), image.spacing());
    assert_eq!(output.direction(), image.direction());
}

#[test]
fn test_minmax_min_is_zero_after_normalization() {
    // Minimum value must map to 0 in [0,1] range.
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let image: Image<f32, TestBackend, 1> = make_image(data, [8]);
    let normalizer = MinMaxNormalizer::new();
    let result = normalizer.normalize(&image);

    let stats = crate::image_statistics::compute_statistics(&result);
    assert!(
        stats.min.abs() < 1e-5,
        "min after normalization must be ≈ 0.0, got {}",
        stats.min
    );
}

#[test]
fn test_minmax_max_is_one_after_normalization() {
    // Maximum value must map to ≈ 1 in [0,1] range.
    let data: Vec<f32> = (1u8..=8).map(|x| x as f32).collect();
    let image: Image<f32, TestBackend, 1> = make_image(data, [8]);
    let normalizer = MinMaxNormalizer::new();
    let result = normalizer.normalize(&image);

    let stats = crate::image_statistics::compute_statistics(&result);
    assert!(
        (stats.max - 1.0).abs() < 1e-4,
        "max after normalization must be ≈ 1.0, got {}",
        stats.max
    );
}

#[test]
fn test_minmax_ordering_preserved() {
    // Monotone input order must be preserved in the output.
    let data: Vec<f32> = (0u8..8).map(|x| x as f32).collect();
    let image: Image<f32, TestBackend, 1> = make_image(data, [8]);
    let normalizer = MinMaxNormalizer::new();
    let result = normalizer.normalize(&image);
    let values = get_values(&result);

    for i in 0..7 {
        assert!(
            values[i] < values[i + 1],
            "ordering violated: values[{}]={} >= values[{}]={}",
            i,
            values[i],
            i + 1,
            values[i + 1]
        );
    }
}

#[test]
fn test_minmax_with_custom_range() {
    // Values [0, 5, 10] normalized to [−1, 1]:
    //   N(0)  = 0.0  → −1.0 + 0   * 2 = −1.0
    //   N(5)  = 0.5  → −1.0 + 0.5 * 2 =  0.0
    //   N(10) ≈ 1.0  → −1.0 + 1   * 2 =  1.0
    let image: Image<f32, TestBackend, 1> = make_image(vec![0.0, 5.0, 10.0], [3]);
    let normalizer = MinMaxNormalizer::with_range(-1.0, 1.0);
    let result = normalizer.normalize(&image);
    let values = get_values(&result);

    assert!(
        (values[0] - (-1.0)).abs() < 1e-4,
        "R(0) ≈ -1.0, got {}",
        values[0]
    );
    assert!(values[1].abs() < 1e-4, "R(5) ≈ 0.0, got {}", values[1]);
    assert!(
        (values[2] - 1.0).abs() < 1e-4,
        "R(10) ≈ 1.0, got {}",
        values[2]
    );
}

#[test]
fn test_minmax_with_positive_shift() {
    // Values [2, 4, 6] normalized to [100, 200]:
    //   N(2) = 0.0 → 100.0
    //   N(4) = 0.5 → 150.0
    //   N(6) ≈ 1.0 → 200.0
    let image: Image<f32, TestBackend, 1> = make_image(vec![2.0, 4.0, 6.0], [3]);
    let normalizer = MinMaxNormalizer::with_range(100.0, 200.0);
    let result = normalizer.normalize(&image);
    let values = get_values(&result);

    assert!(
        (values[0] - 100.0).abs() < 1e-3,
        "R(2) ≈ 100.0, got {}",
        values[0]
    );
    assert!(
        (values[1] - 150.0).abs() < 1e-3,
        "R(4) ≈ 150.0, got {}",
        values[1]
    );
    assert!(
        (values[2] - 200.0).abs() < 1e-3,
        "R(6) ≈ 200.0, got {}",
        values[2]
    );
}

#[test]
fn test_minmax_negative_input_values() {
    // Values [−10, 0, 10]:
    //   range = 20,  N(−10)=0, N(0)=0.5, N(10)≈1
    let image: Image<f32, TestBackend, 1> = make_image(vec![-10.0, 0.0, 10.0], [3]);
    let normalizer = MinMaxNormalizer::new();
    let result = normalizer.normalize(&image);
    let values = get_values(&result);

    assert!(values[0].abs() < 1e-5, "N(−10) ≈ 0.0, got {}", values[0]);
    assert!(
        (values[1] - 0.5).abs() < 1e-4,
        "N(0) ≈ 0.5, got {}",
        values[1]
    );
    assert!(
        (values[2] - 1.0).abs() < 1e-4,
        "N(10) ≈ 1.0, got {}",
        values[2]
    );
}

#[test]
fn test_minmax_preserves_metadata() {
    let origin = ritk_spatial::Point::new([1.0, 2.0, 3.0]);
    let spacing = ritk_spatial::Spacing::new([0.5, 0.5, 0.5]);
    let direction = ritk_spatial::Direction::<3>::identity();
    let image: Image<f32, TestBackend, 3> = make_image_with(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2, 2, 2],
        Some(origin),
        Some(spacing),
        None,
    );

    let normalizer = MinMaxNormalizer::new();
    let result = normalizer.normalize(&image);

    assert_eq!(result.origin(), &origin, "origin must be preserved");
    assert_eq!(result.spacing(), &spacing, "spacing must be preserved");
    assert_eq!(
        result.direction(),
        &direction,
        "direction must be preserved"
    );
    assert_eq!(result.shape(), [2, 2, 2], "shape must be preserved");
}

#[test]
fn test_minmax_3d_shape_preserved() {
    let image: Image<f32, TestBackend, 3> =
        make_image((0..27).map(|x| x as f32).collect(), [3, 3, 3]);
    let normalizer = MinMaxNormalizer::new();
    let result = normalizer.normalize(&image);
    assert_eq!(result.shape(), [3, 3, 3]);

    let values = get_values_3d(&result);
    // First element (min) maps to ≈ 0.
    assert!(
        values[0].abs() < 1e-5,
        "first element ≈ 0.0, got {}",
        values[0]
    );
    // Last element (max) maps to ≈ 1.
    let last = *values.last().expect("infallible: validated precondition");
    assert!(
        (last - 1.0).abs() < 1e-4,
        "last element ≈ 1.0, got {}",
        last
    );
}

// ── Boundary / edge cases ──────────────────────────────────────────────────

#[test]
fn test_minmax_constant_image_does_not_panic() {
    // Constant image: range = 0. Division by (0 + 1e-8) must not panic.
    // N(c) = (c − c) / ε = 0.  Remapped: range.min() + 0 * range.span() = range.min().
    let image: Image<f32, TestBackend, 1> = make_image(vec![7.0; 8], [8]);
    let normalizer = MinMaxNormalizer::new();
    let result = normalizer.normalize(&image);
    let values = get_values(&result);

    for &v in &values {
        assert!(
            v.abs() < 1e-3,
            "constant image → all outputs ≈ 0, got {}",
            v
        );
    }
}

#[test]
fn test_minmax_constant_image_custom_range_maps_to_target_min() {
    // Constant image with custom range [5, 10]:
    //   N(c) = 0 → 5 + 0 * 5 = 5 = range.min().
    let image: Image<f32, TestBackend, 1> = make_image(vec![7.0; 4], [4]);
    let normalizer = MinMaxNormalizer::with_range(5.0, 10.0);
    let result = normalizer.normalize(&image);
    let values = get_values(&result);

    for &v in &values {
        assert!(
            (v - 5.0).abs() < 1e-3,
            "constant image with range [5,10] → all outputs ≈ 5.0, got {}",
            v
        );
    }
}

#[test]
fn test_minmax_single_voxel() {
    // Single voxel: range = 0, N = 0, output = range.min() = 0.
    let image: Image<f32, TestBackend, 1> = make_image(vec![42.0], [1]);
    let normalizer = MinMaxNormalizer::new();
    let result = normalizer.normalize(&image);
    let values = get_values(&result);

    assert!(
        values[0].abs() < 1e-3,
        "single voxel → output ≈ 0.0, got {}",
        values[0]
    );
}

#[test]
fn test_default_is_unit_range() {
    let n1 = MinMaxNormalizer::default();
    assert_eq!(n1.range.min(), 0.0);
    assert_eq!(n1.range.max(), 1.0);
}
