use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support::burn_compat::make_image;

type TestBackend = NdArray<f32>;

fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
    let n = data.len();
    make_image(data, [n])
}

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

// ── Degenerate / constant image ────────────────────────────────────────────

#[test]
fn test_constant_image_returns_constant_value() {
    let image = make_image_1d(vec![42.0; 100]);
    let t = LiThreshold::new().compute(&image);
    assert!(
        (t - 42.0).abs() < 1e-3,
        "constant image threshold must equal the constant value, got {}",
        t
    );
}

// ── Bimodal image with known threshold ─────────────────────────────────────

#[test]
fn test_bimodal_threshold_between_modes() {
    // 50 voxels at 20.0 and 50 voxels at 200.0.
    // The converged threshold must lie strictly between the two modes.
    let mut data = vec![20.0_f32; 50];
    data.extend(vec![200.0_f32; 50]);
    let image = make_image_1d(data);

    let t = li_threshold(&image);

    assert!(
        t > 20.0,
        "threshold must exceed lower mode (20.0), got {}",
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
fn test_apply_output_shape_matches_input() {
    let dims = [4, 5, 6];
    let total = dims.iter().product::<usize>();
    let data: Vec<f32> = (0..total).map(|i| (i % 2) as f32 * 100.0).collect();
    let image = make_image_3d(data, dims);

    let mask = LiThreshold::new().apply(&image);
    assert_eq!(mask.shape(), dims);
}

// ── Apply produces binary output ───────────────────────────────────────────

#[test]
fn test_apply_output_is_binary() {
    let mut data = vec![10.0_f32; 50];
    data.extend(vec![180.0_f32; 50]);
    let image = make_image_1d(data);

    let mask = LiThreshold::new().apply(&image);
    let vals = get_slice_1d(&mask);
    for &v in &vals {
        assert!(v == 0.0 || v == 1.0, "mask must be binary, found {}", v);
    }
}

// ── Spatial metadata preserved ─────────────────────────────────────────────

#[test]
fn test_apply_preserves_spatial_metadata() {
    let data = vec![10.0_f32; 30];
    let mut data2 = data.clone();
    data2.extend(vec![200.0_f32; 30]);
    let image = make_image_1d(data2);

    let mask = LiThreshold::new().apply(&image);
    assert_eq!(mask.origin(), image.origin());
    assert_eq!(mask.spacing(), image.spacing());
    assert_eq!(mask.direction(), image.direction());
}

// ── Convenience function agrees with struct ────────────────────────────────

#[test]
fn test_convenience_fn_matches_struct_compute() {
    let mut data = vec![30.0_f32; 40];
    data.extend(vec![170.0_f32; 60]);
    let image = make_image_1d(data);

    let t_fn = li_threshold(&image);
    let t_struct = LiThreshold::new().compute(&image);
    assert!(
        (t_fn - t_struct).abs() < 1e-9,
        "convenience fn and struct must agree: {} vs {}",
        t_fn,
        t_struct
    );
}

// ── Default delegates to new ───────────────────────────────────────────────

#[test]
fn test_default_is_256_bins_1000_iters() {
    let d = LiThreshold::default();
    assert_eq!(d.num_bins, 256);
    assert_eq!(d.max_iterations, 1000);
}
