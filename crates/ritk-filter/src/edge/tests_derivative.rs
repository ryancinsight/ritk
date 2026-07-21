use super::*;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

fn vals(image: &Image<f32, B, 3>) -> Vec<f32> {
    image
        .data_slice()
        .expect("invariant: contiguous host storage")
        .to_vec()
}

/// Order-1 derivative of a ramp along X (axis 2), unit spacing: central
/// difference with edge-clamp. Verified against the `sitk.Derivative` probe:
/// `[0,10,20,30,40] → [5,10,10,10,5]`.
#[test]
fn derivative_order1_ramp_matches_sitk_probe() {
    let img = ts::make_image::<f32, B, 3>(vec![0.0, 10.0, 20.0, 30.0, 40.0], [1, 1, 5]);
    let out = DerivativeImageFilter::new(2, 1, true)
        .apply(&img)
        .expect("infallible: validated precondition");
    assert_eq!(vals(&out), vec![5.0, 10.0, 10.0, 10.0, 5.0]);
}

/// Order-2 derivative of a linear ramp is zero in the interior (constant slope).
#[test]
fn derivative_order2_of_ramp_is_zero_interior() {
    let img = ts::make_image::<f32, B, 3>(vec![0.0, 10.0, 20.0, 30.0, 40.0], [1, 1, 5]);
    let out = vals(
        &DerivativeImageFilter::new(2, 2, true)
            .apply(&img)
            .expect("infallible: validated precondition"),
    );
    // interior (indices 1..4) of d²/dx² of a line = 0.
    for &v in &out[1..4] {
        assert!(
            v.abs() < 1e-5,
            "second derivative of a ramp must be 0 in the interior, got {v}"
        );
    }
}

/// Order-2 of a parabola x² gives the constant 2 in the interior.
#[test]
fn derivative_order2_of_parabola_is_two() {
    let f: Vec<f32> = (0..6).map(|i| (i * i) as f32).collect(); // 0,1,4,9,16,25
    let img = ts::make_image::<f32, B, 3>(f, [1, 1, 6]);
    let out = vals(
        &DerivativeImageFilter::new(2, 2, true)
            .apply(&img)
            .expect("infallible: validated precondition"),
    );
    for &v in &out[1..5] {
        assert!((v - 2.0).abs() < 1e-4, "d²/dx²(x²) = 2; got {v}");
    }
}

/// Image spacing scales the derivative: spacing 2 halves the order-1 result.
#[test]
fn derivative_respects_image_spacing() {
    use ritk_spatial::{Direction, Point, Spacing};
    let t = ritk_image::tensor::Tensor::<f32, B>::from_slice(
        [1, 1, 5],
        &[0.0_f32, 10.0, 20.0, 30.0, 40.0],
    );
    let img = Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 2.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank");
    let out = vals(
        &DerivativeImageFilter::new(2, 1, true)
            .apply(&img)
            .expect("infallible: validated precondition"),
    );
    // central diff / (2*spacing) = 20/(2*2) = 5 in the interior.
    assert!(
        (out[2] - 5.0).abs() < 1e-5,
        "spacing-scaled derivative, got {}",
        out[2]
    );
}
