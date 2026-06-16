use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, shape)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

#[test]
fn shift_scale_identity_zero_shift_unit_scale() {
    // shift=0, scale=1 → out = in
    let img = make_image(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
    let out = ShiftScaleImageFilter::new(0.0, 1.0).apply(&img).unwrap();
    let v = voxels(&out);
    let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for (i, (&a, &b)) in v.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "voxel {} expected {} got {}", i, b, a);
    }
}

#[test]
fn shift_scale_shift_only_adds_constant() {
    // shift=10, scale=1 → out = in + 10
    let img = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
    let out = ShiftScaleImageFilter::new(10.0, 1.0).apply(&img).unwrap();
    let v = voxels(&out);
    for (i, (&a, &b)) in v.iter().zip([11.0f32, 12.0, 13.0, 14.0].iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "voxel {}: expected {} got {}",
            i,
            b,
            a
        );
    }
}

#[test]
fn shift_scale_scale_only_multiplies() {
    // shift=0, scale=2 → out = in * 2
    let img = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
    let out = ShiftScaleImageFilter::new(0.0, 2.0).apply(&img).unwrap();
    let v = voxels(&out);
    for (i, (&a, &b)) in v.iter().zip([2.0f32, 4.0, 6.0, 8.0].iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "voxel {}: expected {} got {}",
            i,
            b,
            a
        );
    }
}

#[test]
fn shift_scale_combined_shift_then_scale() {
    // shift=-1024, scale=0.001 → HU to fractional
    // Input: 1024 → (1024 - 1024) * 0.001 = 0.0
    // Input: 0 → (0 - 1024) * 0.001 = -1.024
    let img = make_image(vec![1024.0, 0.0], [1, 1, 2]);
    let out = ShiftScaleImageFilter::new(-1024.0, 0.001)
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    assert!((v[0] - 0.0_f32).abs() < 1e-5, "expected 0.0 got {}", v[0]);
    assert!(
        (v[1] - (-1.024_f32)).abs() < 1e-4,
        "expected -1.024 got {}",
        v[1]
    );
}

#[test]
fn shift_scale_preserves_spatial_metadata() {
    let img = make_image(vec![1.0; 8], [2, 2, 2]);
    let out = ShiftScaleImageFilter::new(5.0, 3.0).apply(&img).unwrap();
    assert_eq!(out.shape(), img.shape());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.origin(), img.origin());
    assert_eq!(out.direction(), img.direction());
}

#[test]
fn shift_scale_zero_scale_gives_zero() {
    let img = make_image(vec![5.0, 10.0, 15.0, 20.0], [1, 2, 2]);
    let out = ShiftScaleImageFilter::new(100.0, 0.0).apply(&img).unwrap();
    let v = voxels(&out);
    for (i, &x) in v.iter().enumerate() {
        assert!((x - 0.0).abs() < 1e-5, "voxel {} expected 0 got {}", i, x);
    }
}
