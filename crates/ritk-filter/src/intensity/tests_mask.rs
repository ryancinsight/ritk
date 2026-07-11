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
fn native_threshold_mask_retains_only_strictly_greater_values() {
    use coeus_core::SequentialBackend;
    use ritk_image::native::Image as NativeImage;
    use ritk_spatial::{Direction, Point, Spacing};

    let image = NativeImage::from_flat_on(
        vec![0.5, 0.5001, 2.0],
        [1, 1, 3],
        Point::new([2.0, 3.0, 5.0]),
        Spacing::new([1.0, 2.0, 4.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = MaskImageFilter::apply_threshold_native(
        &image,
        BinarizationThreshold::DEFAULT,
        &SequentialBackend,
    )
    .expect("native threshold masking succeeds");

    assert_eq!(
        output.data_slice().expect("contiguous output"),
        &[0.0, 0.5001, 2.0]
    );
    assert_eq!(output.shape(), [1, 1, 3]);
    assert_eq!(
        [output.origin()[0], output.origin()[1], output.origin()[2]],
        [2.0, 3.0, 5.0]
    );
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

// ── MaskedAssignImageFilter ───────────────────────────────────────────────────

#[test]
fn masked_assign_writes_constant_where_mask_active() {
    let img = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 4]);
    let mask = make_image(vec![0.0, 1.0, 0.0, 1.0], [1, 1, 4]);
    let out = MaskedAssignImageFilter::new(99.0)
        .apply(&img, &mask)
        .unwrap();
    // mask active at 1,3 → 99; keep image at 0,2.
    assert_eq!(voxels(&out), vec![1.0, 99.0, 3.0, 99.0]);
}

#[test]
fn masked_assign_all_inactive_is_identity() {
    let img = make_image(vec![5.0, 6.0, 7.0], [1, 1, 3]);
    let mask = make_image(vec![0.0, 0.0, 0.0], [1, 1, 3]);
    let out = MaskedAssignImageFilter::new(-1.0)
        .apply(&img, &mask)
        .unwrap();
    assert_eq!(voxels(&out), voxels(&img));
}

#[test]
fn native_mask_family_matches_each_contract() {
    use coeus_core::SequentialBackend;
    use ritk_image::native::Image as NativeImage;
    use ritk_spatial::{Direction, Point, Spacing};

    let image = NativeImage::from_flat_on(
        vec![10.0, 20.0, 30.0],
        [1, 1, 3],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let mask = NativeImage::from_flat_on(
        vec![0.0, 0.5, 1.0],
        [1, 1, 3],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native mask");

    let retained = MaskImageFilter::new()
        .apply_native(&image, &mask, &SequentialBackend)
        .expect("native masking succeeds");
    let negated = MaskNegatedImageFilter::new()
        .apply_native(&image, &mask, &SequentialBackend)
        .expect("native negated masking succeeds");
    let assigned = MaskedAssignImageFilter::new(-1.0)
        .apply_native(&image, &mask, &SequentialBackend)
        .expect("native masked assignment succeeds");

    assert_eq!(
        retained.data_slice().expect("contiguous retained"),
        &[0.0, 0.0, 30.0]
    );
    assert_eq!(
        negated.data_slice().expect("contiguous negated"),
        &[10.0, 20.0, 0.0]
    );
    assert_eq!(
        assigned.data_slice().expect("contiguous assigned"),
        &[10.0, 20.0, -1.0]
    );
    assert_eq!(retained.origin(), image.origin());
    assert_eq!(retained.spacing(), image.spacing());
    assert_eq!(retained.direction(), image.direction());
}
