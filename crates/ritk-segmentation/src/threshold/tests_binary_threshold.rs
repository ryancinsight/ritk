//! Tests for binary_threshold
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn_ndarray::NdArray;
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

fn get_slice_3d(image: &Image<B, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

// ── Positive: all inside band ─────────────────────────────────────────────

#[test]
fn test_all_voxels_inside_band_become_inside_value() {
    // Every voxel is 100.0; band [50, 150] → all inside.
    let image = make_image_1d(vec![100.0_f32; 20]);
    let result = BinaryThreshold::new(50.0, 150.0).apply(&image);
    let vals = get_slice_1d(&result);
    assert!(
        vals.iter().all(|&v| v == 1.0),
        "all voxels in band must be inside_value=1.0, got {:?}",
        vals
    );
}

// ── Positive: all outside band ───────────────────────────────────────────

#[test]
fn test_all_voxels_outside_band_become_outside_value() {
    // Every voxel is 200.0; band [0, 100] → all outside.
    let image = make_image_1d(vec![200.0_f32; 20]);
    let result = BinaryThreshold::new(0.0, 100.0).apply(&image);
    let vals = get_slice_1d(&result);
    assert!(
        vals.iter().all(|&v| v == 0.0),
        "all voxels outside band must be outside_value=0.0, got {:?}",
        vals
    );
}

// ── Positive: exact band boundary (inclusive) ─────────────────────────────

#[test]
fn test_lower_bound_voxel_is_inside() {
    // Voxel exactly at lower bound must map to inside.
    let image = make_image_1d(vec![50.0_f32]);
    let result = BinaryThreshold::new(50.0, 150.0).apply(&image);
    let vals = get_slice_1d(&result);
    assert_eq!(
        vals[0], 1.0,
        "voxel at lower bound must be inside_value=1.0"
    );
}

#[test]
fn test_upper_bound_voxel_is_inside() {
    // Voxel exactly at upper bound must map to inside.
    let image = make_image_1d(vec![150.0_f32]);
    let result = BinaryThreshold::new(50.0, 150.0).apply(&image);
    let vals = get_slice_1d(&result);
    assert_eq!(
        vals[0], 1.0,
        "voxel at upper bound must be inside_value=1.0"
    );
}

#[test]
fn test_voxel_just_below_lower_is_outside() {
    let image = make_image_1d(vec![49.9_f32]);
    let result = BinaryThreshold::new(50.0, 150.0).apply(&image);
    let vals = get_slice_1d(&result);
    assert_eq!(
        vals[0], 0.0,
        "voxel just below lower bound must be outside_value=0.0"
    );
}

#[test]
fn test_voxel_just_above_upper_is_outside() {
    let image = make_image_1d(vec![150.1_f32]);
    let result = BinaryThreshold::new(50.0, 150.0).apply(&image);
    let vals = get_slice_1d(&result);
    assert_eq!(
        vals[0], 0.0,
        "voxel just above upper bound must be outside_value=0.0"
    );
}

// ── Positive: split band ──────────────────────────────────────────────────

#[test]
fn test_band_selects_correct_subset() {
    // Values: [10, 50, 100, 150, 200]; band [50, 150].
    // Expected: [0, 1, 1, 1, 0].
    let image = make_image_1d(vec![10.0, 50.0, 100.0, 150.0, 200.0]);
    let result = BinaryThreshold::new(50.0, 150.0).apply(&image);
    let vals = get_slice_1d(&result);
    assert_eq!(
        vals,
        vec![0.0, 1.0, 1.0, 1.0, 0.0],
        "band [50,150] must select {{50,100,150}}"
    );
}

// ── Positive: custom inside/outside values ───────────────────────────────

#[test]
fn test_custom_inside_outside_values() {
    let image = make_image_1d(vec![10.0, 100.0, 200.0]);
    let result = BinaryThreshold::new(50.0, 150.0)
        .with_values(255.0, 128.0)
        .apply(&image);
    let vals = get_slice_1d(&result);
    assert_eq!(
        vals[0], 128.0,
        "voxel 10.0 outside band → outside_value=128.0"
    );
    assert_eq!(
        vals[1], 255.0,
        "voxel 100.0 inside band → inside_value=255.0"
    );
    assert_eq!(
        vals[2], 128.0,
        "voxel 200.0 outside band → outside_value=128.0"
    );
}

// ── Positive: half-open intervals using infinity ──────────────────────────

#[test]
fn test_upper_only_threshold_via_neg_infinity_lower() {
    // lower = NEG_INFINITY → any value ≤ upper = 100 → inside.
    let image = make_image_1d(vec![-1000.0, 0.0, 50.0, 100.0, 100.1, 200.0]);
    let result = BinaryThreshold::new(f32::NEG_INFINITY, 100.0).apply(&image);
    let vals = get_slice_1d(&result);
    assert_eq!(vals[0], 1.0, "-1000 ≤ 100 → inside");
    assert_eq!(vals[1], 1.0, "0 ≤ 100 → inside");
    assert_eq!(vals[2], 1.0, "50 ≤ 100 → inside");
    assert_eq!(vals[3], 1.0, "100 = 100 → inside");
    assert_eq!(vals[4], 0.0, "100.1 > 100 → outside");
    assert_eq!(vals[5], 0.0, "200 > 100 → outside");
}

#[test]
fn test_lower_only_threshold_via_infinity_upper() {
    // upper = INFINITY → any value ≥ lower = 100 → inside.
    let image = make_image_1d(vec![50.0, 99.9, 100.0, 1000.0]);
    let result = BinaryThreshold::new(100.0, f32::INFINITY).apply(&image);
    let vals = get_slice_1d(&result);
    assert_eq!(vals[0], 0.0, "50 < 100 → outside");
    assert_eq!(vals[1], 0.0, "99.9 < 100 → outside");
    assert_eq!(vals[2], 1.0, "100 ≥ 100 → inside");
    assert_eq!(vals[3], 1.0, "1000 ≥ 100 → inside");
}

// ── Positive: single-point band ──────────────────────────────────────────

#[test]
fn test_single_point_band_lower_eq_upper() {
    // lower == upper: only voxels exactly equal to that value → inside.
    let image = make_image_1d(vec![99.9, 100.0, 100.0, 100.1]);
    let result = BinaryThreshold::new(100.0, 100.0).apply(&image);
    let vals = get_slice_1d(&result);
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[1], 1.0);
    assert_eq!(vals[2], 1.0);
    assert_eq!(vals[3], 0.0);
}

// ── Positive: spatial metadata preserved ─────────────────────────────────

#[test]
fn test_spatial_metadata_preserved() {
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let td = TensorData::new(vec![100.0_f32; 24], Shape::new([2, 3, 4]));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let image: Image<B, 3> = Image::new(tensor, origin, spacing, direction);

    let result = BinaryThreshold::new(50.0, 150.0).apply(&image);
    assert_eq!(result.origin(), &origin);
    assert_eq!(result.spacing(), &spacing);
    assert_eq!(result.direction(), &direction);
}

// ── Positive: output shape matches input ──────────────────────────────────

#[test]
fn test_output_shape_matches_input() {
    let dims = [4, 5, 6];
    let n: usize = dims.iter().product();
    let image = make_image_3d((0..n).map(|i| i as f32).collect(), dims);
    let result = BinaryThreshold::new(0.0, 100.0).apply(&image);
    assert_eq!(result.shape(), dims, "output shape must match input shape");
}

// ── Positive: struct and function agree ───────────────────────────────────

#[test]
fn test_struct_and_function_produce_identical_results() {
    let data: Vec<f32> = (0..30).map(|i| i as f32 * 10.0).collect();
    let image = make_image_1d(data);

    let via_struct = BinaryThreshold::new(50.0, 200.0).apply(&image);
    let via_fn = binary_threshold(&image, 50.0, 200.0, 1.0, 0.0);

    let s = get_slice_1d(&via_struct);
    let f = get_slice_1d(&via_fn);
    assert_eq!(s, f, "struct and function must produce identical results");
}

// ── Positive: slice function parity ──────────────────────────────────────

#[test]
fn test_slice_fn_matches_filter() {
    let data: Vec<f32> = (0..50).map(|i| i as f32 * 5.0).collect();
    let image = make_image_1d(data.clone());
    let via_filter = BinaryThreshold::new(50.0, 150.0).apply(&image);
    let via_slice = apply_binary_threshold_to_slice(&data, 50.0, 150.0, 1.0, 0.0);
    let filter_vals = get_slice_1d(&via_filter);
    assert_eq!(filter_vals, via_slice, "slice fn must match filter");
}

// ── Negative: lower > upper panics ────────────────────────────────────────

#[test]
#[should_panic(expected = "lower bound 200 must be ≤ upper bound 100")]
fn test_lower_gt_upper_panics_new() {
    BinaryThreshold::new(200.0, 100.0);
}

#[test]
#[should_panic(expected = "lower bound 200 must be ≤ upper bound 100")]
fn test_lower_gt_upper_panics_function() {
    let image = make_image_1d(vec![100.0_f32]);
    binary_threshold(&image, 200.0, 100.0, 1.0, 0.0);
}

// ── Negative: non-finite inside/outside panics ────────────────────────────

#[test]
#[should_panic(expected = "inside_value must be finite")]
fn test_infinite_inside_value_panics() {
    BinaryThreshold::new(0.0, 100.0).with_values(f32::INFINITY, 0.0);
}

#[test]
#[should_panic(expected = "outside_value must be finite")]
fn test_nan_outside_value_panics() {
    BinaryThreshold::new(0.0, 100.0).with_values(1.0, f32::NAN);
}

// ── Boundary: default construction ───────────────────────────────────────

#[test]
fn test_default_construction() {
    let d = BinaryThreshold::default();
    assert_eq!(d.lower, f32::NEG_INFINITY);
    assert_eq!(d.upper, f32::INFINITY);
    assert_eq!(d.inside_value, 1.0);
    assert_eq!(d.outside_value, 0.0);
}

// ── Adversarial: 3D analytical correctness ────────────────────────────────

#[test]
fn test_3d_band_select_correct_voxel_count() {
    // 4×4×4 image with values 0..64; band [16, 32] → voxels with value in [16,32].
    // Analytically: values 16,17,...,32 → 17 voxels.
    let data: Vec<f32> = (0u32..64).map(|i| i as f32).collect();
    let image = make_image_3d(data, [4, 4, 4]);
    let result = BinaryThreshold::new(16.0, 32.0).apply(&image);
    let inside_count = get_slice_3d(&result).iter().filter(|&&v| v == 1.0).count();
    assert_eq!(
        inside_count, 17,
        "band [16,32] on 0..63 must select exactly 17 voxels, got {}",
        inside_count
    );
}
