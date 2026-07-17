//! Native-image contracts for pixelwise intensity filters.

use coeus_core::SequentialBackend;
use ritk_filter::{
    BlendImageFilter, TernaryAddImageFilter, TernaryMagnitudeImageFilter,
    TernaryMagnitudeSquaredImageFilter,
};
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

fn image(values: Vec<f32>, origin: [f64; 3]) -> Image<f32, B, 3> {
    Image::from_flat_on(
        values,
        [1, 1, 3],
        Point::new(origin),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &B::default(),
    )
    .expect("invariant: valid native test image")
}

fn values(image: &Image<f32, B, 3>) -> &[f32] {
    image
        .data_slice()
        .expect("invariant: contiguous native test image")
}

#[test]
fn blend_preserves_values_and_first_image_metadata() {
    let a = image(vec![0.0, 10.0, 20.0], [1.0, 2.0, 3.0]);
    let b = image(vec![100.0, 100.0, 100.0], [9.0, 8.0, 7.0]);
    let output = BlendImageFilter::new(0.5)
        .apply_native(&a, &b, &B::default())
        .expect("native blend succeeds");

    assert_eq!(values(&output), &[50.0, 55.0, 60.0]);
    assert_eq!(output.origin(), a.origin());
    assert_eq!(output.spacing(), a.spacing());
    assert_eq!(output.direction(), a.direction());
}

#[test]
fn blend_endpoints_select_the_corresponding_image() {
    let a = image(vec![1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
    let b = image(vec![10.0, 20.0, 30.0], [9.0, 8.0, 7.0]);

    let first = BlendImageFilter::new(0.0)
        .apply_native(&a, &b, &B::default())
        .expect("native zero-alpha blend succeeds");
    let second = BlendImageFilter::new(1.0)
        .apply_native(&a, &b, &B::default())
        .expect("native one-alpha blend succeeds");

    assert_eq!(values(&first), values(&a));
    assert_eq!(values(&second), values(&b));
}

#[test]
fn ternary_add_preserves_values_and_first_image_metadata() {
    let a = image(vec![1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
    let b = image(vec![10.0, 20.0, 30.0], [9.0, 8.0, 7.0]);
    let c = image(vec![100.0, 200.0, 300.0], [4.0, 5.0, 6.0]);
    let output = TernaryAddImageFilter::new()
        .apply_native(&a, &b, &c, &B::default())
        .expect("native ternary addition succeeds");

    assert_eq!(values(&output), &[111.0, 222.0, 333.0]);
    assert_eq!(output.origin(), a.origin());
    assert_eq!(output.spacing(), a.spacing());
    assert_eq!(output.direction(), a.direction());
}

#[test]
fn ternary_magnitude_contracts_match_integer_triples() {
    let a = image(vec![2.0, 1.0, 0.0], [1.0, 2.0, 3.0]);
    let b = image(vec![3.0, 2.0, 0.0], [9.0, 8.0, 7.0]);
    let c = image(vec![6.0, 2.0, 5.0], [4.0, 5.0, 6.0]);

    let magnitude = TernaryMagnitudeImageFilter::new()
        .apply_native(&a, &b, &c, &B::default())
        .expect("native ternary magnitude succeeds");
    let squared = TernaryMagnitudeSquaredImageFilter::new()
        .apply_native(&a, &b, &c, &B::default())
        .expect("native ternary squared magnitude succeeds");

    assert_eq!(values(&magnitude), &[7.0, 3.0, 5.0]);
    assert_eq!(values(&squared), &[49.0, 9.0, 25.0]);
}
