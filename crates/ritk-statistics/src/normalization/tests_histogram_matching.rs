use super::*;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support::{make_image, make_image_with};
use ritk_image::Image;

type TestBackend = SequentialBackend;

fn get_values(image: &Image<f32, TestBackend, 1>) -> Vec<f32> {
    ritk_tensor_ops::extract_vec_infallible(image).0
}

// ── Positive tests ──────────────────────────────────────────────────────────

#[test]
fn test_self_match_is_approximately_identity() {
    // Matching a source against itself: T(v) = F_src⁻¹(F_src(v)) ≈ v.
    // Due to discrete LUT quantisation, the tolerance is one LUT step.
    let data: Vec<f32> = (0u16..256).map(|x| x as f32 * 32.0 / 255.0).collect();
    let image: Image<f32, TestBackend, 1> = make_image(data.clone(), [data.len()]);
    let matcher = HistogramMatcher::new(256);
    let result = matcher.match_histograms(&image, &image);
    let values = get_values(&result);

    let step = 32.0_f32 / 255.0; // one LUT bin width = reference quantization step for n=256
    for (i, (&orig, &out)) in data.iter().zip(values.iter()).enumerate() {
        assert!(
            (orig - out).abs() <= step + 1e-3,
            "self-match diverged at index {}: orig={} out={} tol={}",
            i,
            orig,
            out,
            step + 1e-3
        );
    }
}

#[test]
fn native_histogram_match_maps_endpoints_and_preserves_source_geometry() {
    let source = NativeImage::from_flat_on(
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        [1, 1, 5],
        ritk_spatial::Point::new([1.0, 2.0, 3.0]),
        ritk_spatial::Spacing::new([0.5, 1.0, 2.0]),
        ritk_spatial::Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native source image");
    let reference = NativeImage::from_flat_on(
        vec![10.0, 11.0, 12.0, 13.0, 14.0],
        [1, 1, 5],
        ritk_spatial::Point::new([4.0, 5.0, 6.0]),
        ritk_spatial::Spacing::new([3.0, 4.0, 5.0]),
        ritk_spatial::Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native reference image");

    let output = HistogramMatcher::new(64)
        .match_histograms_native(&source, &reference)
        .expect("native histogram matching succeeds");
    let values = output.data_slice().expect("contiguous native output");

    assert_eq!(values.first(), Some(&10.0));
    assert_eq!(values.last(), Some(&14.0));
    assert!(values.iter().all(|&value| (10.0..=14.0).contains(&value)));
    assert_eq!(output.origin(), source.origin());
    assert_eq!(output.spacing(), source.spacing());
    assert_eq!(output.direction(), source.direction());
}

#[test]
fn test_self_match_preserves_unsorted_source_order() {
    let data = vec![7.0, 1.0, 6.0, 2.0, 5.0, 3.0, 4.0, 0.0];
    let image: Image<f32, TestBackend, 1> = make_image(data.clone(), [data.len()]);

    let matcher = HistogramMatcher::new(8)
        .with_match_points(0)
        .with_threshold_at_mean(false);
    let result = matcher.match_histograms(&image, &image);
    let values = get_values(&result);

    assert_eq!(
        values, data,
        "self-match must preserve original voxel order when landmarks are estimated"
    );
}

#[test]
fn test_match_shifts_mean_toward_reference() {
    // Source in [0, 10]; reference in [90, 100].
    // After matching, output mean must be close to reference mean ≈ 95.
    let source: Vec<f32> = (0u8..=10).map(|x| x as f32).collect();
    let reference: Vec<f32> = (90u8..=100).map(|x| x as f32).collect();

    let src_image: Image<f32, TestBackend, 1> = make_image(source, [11]);
    let ref_image: Image<f32, TestBackend, 1> = make_image(reference, [11]);

    let matcher = HistogramMatcher::new(64);
    let result = matcher.match_histograms(&src_image, &ref_image);
    let values = get_values(&result);

    let out_mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    let ref_mean = 95.0_f32;
    assert!(
        (out_mean - ref_mean).abs() < 5.0,
        "output mean {} not close to reference mean {}",
        out_mean,
        ref_mean
    );
}

#[test]
fn test_output_values_bounded_by_reference_range() {
    // All output values must lie within [ref_min, ref_max] plus one LUT step.
    let source: Vec<f32> = (0u8..=20).map(|x| x as f32).collect();
    let reference: Vec<f32> = (50u8..=70).map(|x| x as f32).collect();

    let src_image: Image<f32, TestBackend, 1> = make_image(source, [21]);
    let ref_image: Image<f32, TestBackend, 1> = make_image(reference, [21]);

    let matcher = HistogramMatcher::new(64);
    let result = matcher.match_histograms(&src_image, &ref_image);
    let values = get_values(&result);

    let (ref_min, ref_max) = (50.0_f32, 70.0_f32);
    let tol = 1.0_f32; // one reference bin step

    for &v in &values {
        assert!(
            v >= ref_min - tol && v <= ref_max + tol,
            "output value {} outside reference range [{}, {}] ± {}",
            v,
            ref_min,
            ref_max,
            tol
        );
    }
}

#[test]
fn test_output_shape_matches_source() {
    // Output shape must equal source shape regardless of reference shape.
    let source: Vec<f32> = (0u8..16).map(|x| x as f32).collect();
    let reference: Vec<f32> = (0u8..64).map(|x| x as f32).collect();

    let src_image: Image<f32, TestBackend, 1> = make_image(source, [16]);
    let ref_image: Image<f32, TestBackend, 1> = make_image(reference, [64]);

    let matcher = HistogramMatcher::default();
    let result = matcher.match_histograms(&src_image, &ref_image);

    assert_eq!(
        result.shape(),
        src_image.shape(),
        "output shape must match source shape"
    );
}

#[test]
fn test_preserves_spatial_metadata() {
    let origin = ritk_spatial::Point::new([1.0, 2.0, 3.0]);
    let spacing = ritk_spatial::Spacing::new([0.5, 0.5, 0.5]);
    let make_3d = |vals: Vec<f32>| -> Image<f32, TestBackend, 3> {
        make_image_with(vals, [3, 3, 3], Some(origin), Some(spacing), None)
    };

    let src_vals: Vec<f32> = (0u16..27).map(|x| x as f32).collect();
    let ref_vals: Vec<f32> = (10u16..37).map(|x| x as f32).collect();
    let source = make_3d(src_vals);
    let reference = make_3d(ref_vals);

    let matcher = HistogramMatcher::default();
    let result = matcher.match_histograms(&source, &reference);

    assert_eq!(result.origin(), source.origin(), "origin must be preserved");
    assert_eq!(
        result.spacing(),
        source.spacing(),
        "spacing must be preserved"
    );
    assert_eq!(
        result.direction(),
        source.direction(),
        "direction must be preserved"
    );
    assert_eq!(result.shape(), [3, 3, 3], "shape must be preserved");
}

#[test]
fn test_monotone_output_for_monotone_input() {
    // A monotonically increasing source matched against a monotonically
    // increasing reference must produce a monotonically non-decreasing output.
    let source: Vec<f32> = (0u8..=16).map(|x| x as f32).collect();
    let reference: Vec<f32> = (0u8..=100).map(|x| x as f32).collect();

    let src_image: Image<f32, TestBackend, 1> = make_image(source, [17]);
    let ref_image: Image<f32, TestBackend, 1> = make_image(reference, [101]);

    let matcher = HistogramMatcher::new(128);
    let result = matcher.match_histograms(&src_image, &ref_image);
    let values = get_values(&result);

    for i in 0..values.len().saturating_sub(1) {
        assert!(
            values[i] <= values[i + 1] + 1e-4,
            "monotonicity violated: values[{}]={} > values[{}]={}",
            i,
            values[i],
            i + 1,
            values[i + 1]
        );
    }
}

#[test]
fn test_lut_endpoints_clamp_correctly() {
    // src_min must map to ≈ ref_min; src_max must map to ≈ ref_max.
    // With a 1:1 uniform mapping, the first/last source values map to
    // the first/last reference values.
    let n = 32usize;
    let source: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let reference: Vec<f32> = (100..100 + n).map(|x| x as f32).collect();

    let src_image: Image<f32, TestBackend, 1> = make_image(source, [n]);
    let ref_image: Image<f32, TestBackend, 1> = make_image(reference, [n]);

    let matcher = HistogramMatcher::new(256);
    let result = matcher.match_histograms(&src_image, &ref_image);
    let values = get_values(&result);

    // First source pixel (src_min=0) → should map close to ref_min=100.
    assert!(
        (values[0] - 100.0).abs() < 1.0,
        "src_min should map near ref_min=100, got {}",
        values[0]
    );
    // Last source pixel (src_max=31) → should map close to ref_max=131.
    let last = *values.last().unwrap();
    assert!(
        (last - 131.0).abs() < 1.0,
        "src_max should map near ref_max=131, got {}",
        last
    );
}

// ── Boundary / edge cases ────────────────────────────────────────────────────

#[test]
fn test_constant_source_returns_unchanged() {
    // Constant source: CDF slope is undefined → source returned unchanged.
    let source: Image<f32, TestBackend, 1> = make_image(vec![5.0; 8], [8]);
    let reference: Vec<f32> = (0u8..8).map(|x| x as f32).collect();
    let ref_image: Image<f32, TestBackend, 1> = make_image(reference, [8]);

    let matcher = HistogramMatcher::new(64);
    let result = matcher.match_histograms(&source, &ref_image);
    let values = get_values(&result);

    for &v in &values {
        assert!(
            (v - 5.0).abs() < 1e-5,
            "constant source must be returned unchanged, got {}",
            v
        );
    }
}

#[test]
fn test_single_source_voxel_does_not_panic() {
    // Single source voxel must produce a single output value without panic.
    let source: Image<f32, TestBackend, 1> = make_image(vec![0.5], [1]);
    let reference: Vec<f32> = (0u8..16).map(|x| x as f32).collect();
    let ref_image: Image<f32, TestBackend, 1> = make_image(reference, [16]);

    let matcher = HistogramMatcher::new(64);
    let result = matcher.match_histograms(&source, &ref_image);

    assert_eq!(result.shape(), [1]);
}

#[test]
fn test_single_reference_voxel() {
    // Single reference value: all output values must equal that reference value.
    let source: Vec<f32> = (0u8..8).map(|x| x as f32).collect();
    let src_image: Image<f32, TestBackend, 1> = make_image(source, [8]);
    let ref_image: Image<f32, TestBackend, 1> = make_image(vec![42.0], [1]);

    let matcher = HistogramMatcher::new(64);
    let result = matcher.match_histograms(&src_image, &ref_image);
    let values = get_values(&result);

    for &v in &values {
        assert!(
            (v - 42.0).abs() < 1e-4,
            "single reference → all outputs must equal 42.0, got {}",
            v
        );
    }
}

#[test]
fn test_default_uses_256_bins() {
    let m = HistogramMatcher::default();
    assert_eq!(m.num_bins, 256);
}
