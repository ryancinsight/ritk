//! Tests for triangle thresholding.
//! Extracted to keep the 500-line structural limit.

use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support::{make_image, make_image_1d};

type TestBackend = NdArray<f32>;

fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    make_image(data, dims)
}

fn get_slice_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

// ── Degenerate case ────────────────────────────────────────────────────────

#[test]
fn test_constant_image_returns_constant_value() {
    let data = vec![42.0f32; 100];
    let image = make_image_1d(data);
    let t = triangle_threshold(&image);
    assert!(
        (t - 42.0).abs() < f32::EPSILON,
        "constant image must return its value, got {}",
        t
    );
}

// ── Bimodal distribution ───────────────────────────────────────────────────

#[test]
fn test_bimodal_threshold_between_modes() {
    // Bimodal: strong peak at 30.0 (200 voxels), smaller peak at 200.0 (20 voxels).
    // Triangle algorithm should place the threshold between the two modes.
    let mut data = vec![30.0f32; 200];
    data.extend(vec![200.0f32; 20]);
    let image = make_image_1d(data);
    let t = TriangleThreshold::new().compute(&image);

    assert!(
        t > 30.0,
        "threshold must exceed lower mode (30.0), got {}",
        t
    );
    assert!(
        t < 200.0,
        "threshold must be below upper mode (200.0), got {}",
        t
    );
}

// ── Output shape preservation ──────────────────────────────────────────────

#[test]
fn test_apply_output_shape_matches_input() {
    let dims = [2, 3, 4];
    let n: usize = dims.iter().product();
    let mut data = vec![10.0f32; n / 2];
    data.extend(vec![200.0f32; n - n / 2]);
    let image = make_image_3d(data, dims);
    let result = TriangleThreshold::new().apply(&image);
    assert_eq!(result.shape(), dims);
}

// ── Apply produces binary mask ─────────────────────────────────────────────

#[test]
fn test_apply_output_is_binary() {
    let mut data = vec![10.0f32; 60];
    data.extend(vec![200.0f32; 40]);
    let image = make_image_1d(data);
    let result = TriangleThreshold::new().apply(&image);
    let out = get_slice_1d(&result);
    for &v in &out {
        assert!(
            v == 0.0 || v == 1.0,
            "output must be binary (0 or 1), got {}",
            v
        );
    }
}

// ── Convenience function matches struct ────────────────────────────────────

#[test]
fn test_convenience_fn_matches_struct() {
    let mut data = vec![25.0f32; 80];
    data.extend(vec![180.0f32; 20]);
    let image = make_image_1d(data);
    let t_fn = triangle_threshold(&image);
    let t_struct = TriangleThreshold::new().compute(&image);
    assert!(
        (t_fn - t_struct).abs() < f32::EPSILON,
        "convenience fn and struct must agree"
    );
}

// ── Spatial metadata preserved ─────────────────────────────────────────────

#[test]
fn test_apply_preserves_spatial_metadata() {
    let dims = [2, 3, 4];
    let n: usize = dims.iter().product();
    let mut data = vec![10.0f32; n / 2];
    data.extend(vec![200.0f32; n - n / 2]);
    let image = make_image_3d(data, dims);
    let result = TriangleThreshold::new().apply(&image);
    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}

// ── Default is 256 bins ────────────────────────────────────────────────────

#[test]
fn test_default_is_256_bins() {
    let t = TriangleThreshold::default();
    assert_eq!(t.num_bins, 256);
}

// ── Panics on invalid bin count ────────────────────────────────────────────

#[test]
#[should_panic(expected = "num_bins must be ≥ 2")]
fn test_with_bins_one_panics() {
    TriangleThreshold::with_bins(1);
}
