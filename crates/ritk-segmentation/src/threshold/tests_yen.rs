use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support::{make_image, make_image_1d};

type B = NdArray<f32>;

fn make_image_1d(data: Vec<f32>) -> Image<B, 1> {
    make_image_1d(data)
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
fn test_constant_image_returns_constant_value() {
    let data = vec![42.0_f32; 100];
    let image = make_image_1d(data);
    let t = yen_threshold(&image);
    assert!(
        (t - 42.0).abs() < f32::EPSILON,
        "constant image threshold must equal the constant value, got {}",
        t
    );
}

// ── Bimodal image with known threshold ─────────────────────────────────────

#[test]
fn test_bimodal_threshold_separates_modes() {
    let mut data = vec![10.0_f32; 200];
    data.extend(vec![200.0_f32; 200]);
    let image = make_image_1d(data);
    let t = YenThreshold::new().compute(&image);

    assert!(
        t > 10.0,
        "threshold must exceed lower mode (10.0), got {}",
        t
    );
    assert!(
        t < 200.0,
        "threshold must be below upper mode (200.0), got {}",
        t
    );
}

// ── Output shape matches input shape ───────────────────────────────────────

#[test]
fn test_apply_preserves_shape_and_metadata() {
    let dims = [2, 3, 4];
    let n = dims[0] * dims[1] * dims[2];
    let data: Vec<f32> = (0..n)
        .map(|i| if i < n / 2 { 10.0 } else { 200.0 })
        .collect();
    let image = make_image_3d(data, dims);
    let result = YenThreshold::new().apply(&image);

    assert_eq!(result.shape(), dims, "output shape must match input shape");
    assert_eq!(result.origin(), image.origin());
    assert_eq!(result.spacing(), image.spacing());
    assert_eq!(result.direction(), image.direction());
}

// ── Apply output is strictly binary ────────────────────────────────────────

#[test]
fn test_apply_output_is_binary() {
    let mut data = vec![5.0_f32; 80];
    data.extend(vec![250.0_f32; 80]);
    let image = make_image_1d(data);
    let result = YenThreshold::new().apply(&image);

    let out = get_slice_1d(&result);
    for &v in &out {
        assert!(
            v == 0.0 || v == 1.0,
            "apply output must be binary, got {}",
            v
        );
    }
}

// ── Convenience function matches struct compute ────────────────────────────

#[test]
fn test_convenience_fn_matches_struct_compute() {
    let mut data = vec![30.0_f32; 100];
    data.extend(vec![220.0_f32; 100]);
    let image = make_image_1d(data);

    let t_fn = yen_threshold(&image);
    let t_struct = YenThreshold::new().compute(&image);
    assert!(
        (t_fn - t_struct).abs() < f32::EPSILON,
        "convenience function and struct compute must agree"
    );
}

// ── Default trait ──────────────────────────────────────────────────────────

#[test]
fn test_default_is_256_bins() {
    let yt = YenThreshold::default();
    assert_eq!(yt.num_bins, 256);
}

// ── Custom bins ────────────────────────────────────────────────────────────

#[test]
#[should_panic(expected = "num_bins must be ≥ 2")]
fn test_with_bins_one_panics() {
    YenThreshold::with_bins(1);
}

#[test]
#[should_panic(expected = "num_bins must be ≥ 2")]
fn test_with_bins_zero_panics() {
    YenThreshold::with_bins(0);
}
