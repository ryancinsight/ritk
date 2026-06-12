use super::*;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use burn::tensor::backend::Backend;
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

fn get_values(image: &Image<TestBackend, 1>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

// ── Positive tests ──────────────────────────────────────────────────────────

#[test]
fn test_self_match_is_approximately_identity() {
    // Matching a source against itself: T(v) = F_src⁻¹(F_src(v)) ≈ v.
    // Due to discrete LUT quantisation, the tolerance is one LUT step.
    let data: Vec<f32> = (0u16..256).map(|x| x as f32 * 32.0 / 255.0).collect();
    let image = make_image_1d(data.clone());
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
fn test_match_shifts_mean_toward_reference() {
    // Source in [0, 10]; reference in [90, 100].
    // After matching, output mean must be close to reference mean ≈ 95.
    let source: Vec<f32> = (0u8..=10).map(|x| x as f32).collect();
    let reference: Vec<f32> = (90u8..=100).map(|x| x as f32).collect();

    let src_image = make_image_1d(source);
    let ref_image = make_image_1d(reference);

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

    let src_image = make_image_1d(source);
    let ref_image = make_image_1d(reference);

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

    let src_image = make_image_1d(source);
    let ref_image = make_image_1d(reference);

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
    let device: <TestBackend as Backend>::Device = Default::default();
    let make_3d = |vals: Vec<f32>| {
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vals, Shape::new([3, 3, 3])),
            &device,
        );
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.5, 0.5]);
        let direction = Direction::<3>::identity();
        Image::new(tensor, origin, spacing, direction)
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

    let src_image = make_image_1d(source);
    let ref_image = make_image_1d(reference);

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

    let src_image = make_image_1d(source);
    let ref_image = make_image_1d(reference);

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
    let source = make_image_1d(vec![5.0; 8]);
    let reference: Vec<f32> = (0u8..8).map(|x| x as f32).collect();
    let ref_image = make_image_1d(reference);

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
    let source = make_image_1d(vec![0.5]);
    let reference: Vec<f32> = (0u8..16).map(|x| x as f32).collect();
    let ref_image = make_image_1d(reference);

    let matcher = HistogramMatcher::new(64);
    let result = matcher.match_histograms(&source, &ref_image);

    assert_eq!(result.shape(), [1]);
}

#[test]
fn test_single_reference_voxel() {
    // Single reference value: all output values must equal that reference value.
    let source: Vec<f32> = (0u8..8).map(|x| x as f32).collect();
    let src_image = make_image_1d(source);
    let ref_image = make_image_1d(vec![42.0]);

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
