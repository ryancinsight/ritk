use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

#[test]
fn mask_filter_passes_image_values_where_mask_active() {
    let img = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
    let mask = make_image(vec![1.0, 0.0, 1.0, 0.0], [1, 2, 2]);
    let out = MaskImageFilter::new().apply(&img, &mask).unwrap();
    let v = voxels(&out);
    assert!(
        (v[0] - 10.0).abs() < 1e-5,
        "mask-active: expected 10, got {}",
        v[0]
    );
    assert!(
        (v[1] - 0.0).abs() < 1e-5,
        "mask-inactive: expected 0, got {}",
        v[1]
    );
    assert!(
        (v[2] - 30.0).abs() < 1e-5,
        "mask-active: expected 30, got {}",
        v[2]
    );
    assert!(
        (v[3] - 0.0).abs() < 1e-5,
        "mask-inactive: expected 0, got {}",
        v[3]
    );
}

#[test]
fn mask_filter_full_mask_is_identity() {
    let vals = vec![5.0f32, 6.0, 7.0, 8.0];
    let img = make_image(vals.clone(), [1, 2, 2]);
    let mask = make_image(vec![1.0; 4], [1, 2, 2]);
    let out = MaskImageFilter::new().apply(&img, &mask).unwrap();
    let v = voxels(&out);
    for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "[{}] expected {}, got {}", i, b, a);
    }
}

#[test]
fn mask_filter_custom_outside_value() {
    let img = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
    let mask = make_image(vec![0.0, 1.0, 0.0, 1.0], [1, 2, 2]);
    let out = MaskImageFilter::new()
        .with_outside_value(-1.0)
        .apply(&img, &mask)
        .unwrap();
    let v = voxels(&out);
    assert!(
        (v[0] - (-1.0)).abs() < 1e-5,
        "outside: expected -1, got {}",
        v[0]
    );
    assert!(
        (v[1] - 20.0).abs() < 1e-5,
        "inside: expected 20, got {}",
        v[1]
    );
}

#[test]
fn mask_filter_shape_mismatch_returns_error() {
    let img = make_image(vec![1.0; 4], [1, 2, 2]);
    let mask = make_image(vec![1.0; 8], [2, 2, 2]);
    assert!(MaskImageFilter::new().apply(&img, &mask).is_err());
}

#[test]
fn mask_filter_preserves_spatial_metadata() {
    let img = make_image(vec![1.0; 8], [2, 2, 2]);
    let mask = make_image(vec![1.0; 8], [2, 2, 2]);
    let out = MaskImageFilter::new().apply(&img, &mask).unwrap();
    assert_eq!(out.shape(), img.shape());
    assert_eq!(out.spacing(), img.spacing());
}

#[test]
fn mask_negated_filter_passes_values_where_mask_inactive() {
    let img = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
    let mask = make_image(vec![1.0, 0.0, 1.0, 0.0], [1, 2, 2]);
    let out = MaskNegatedImageFilter::new().apply(&img, &mask).unwrap();
    let v = voxels(&out);
    // mask active → outside (0); mask inactive → pass through
    assert!(
        (v[0] - 0.0).abs() < 1e-5,
        "mask-active zeroed: got {}",
        v[0]
    );
    assert!(
        (v[1] - 20.0).abs() < 1e-5,
        "mask-inactive pass: got {}",
        v[1]
    );
    assert!(
        (v[2] - 0.0).abs() < 1e-5,
        "mask-active zeroed: got {}",
        v[2]
    );
    assert!(
        (v[3] - 40.0).abs() < 1e-5,
        "mask-inactive pass: got {}",
        v[3]
    );
}

#[test]
fn mask_negated_full_mask_zeros_everything() {
    let img = make_image(vec![5.0, 6.0, 7.0, 8.0], [1, 2, 2]);
    let mask = make_image(vec![1.0; 4], [1, 2, 2]);
    let out = MaskNegatedImageFilter::new().apply(&img, &mask).unwrap();
    let v = voxels(&out);
    for &val in &v {
        assert!((val - 0.0).abs() < 1e-5, "expected 0, got {}", val);
    }
}
